# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (function-calling, optional streaming)
# All comments remain in English – per user request
from __future__ import annotations

import os
import sys
import json
import time
import queue
import asyncio
import logging
import functools
import threading
from collections.abc import Generator
from typing import Any, Optional

import httpx                      # fast HTTP client, async-friendly
import pandas as pd
import openai                     # ≥ 1.26.0 (required for Responses API)
from flask import (
    Flask, Response, flash, jsonify, redirect,
    render_template, request, session, copy_current_request_context
)
from flask_sqlalchemy import SQLAlchemy
import configparser

# ──────────────────────────────────────────────────────────────────────────────
# 0.  LOGGING (ISO-8601 → stdout)
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ENV & OpenAI client (Responses API)
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL = (cfg.get("DEFAULT", "model",
                 fallback=os.getenv("OPENAI_MODEL", "")).strip() or "gpt-4o")

if not OPENAI_API_KEY:
    log.critical("❌  OPENAI_API_KEY is not configured.")
    sys.exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)
log.info("OpenAI client ready (model=%s, API=Responses)", MODEL)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  FLASK & DB
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs("/app/data", exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Client(db.Model):
    __tablename__ = "client"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128))
    invoices = db.relationship(
        "Invoice", backref="client", lazy=True, cascade="all, delete-orphan"
    )


class Invoice(db.Model):
    __tablename__ = "invoice"
    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_due = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)


with app.app_context():
    db.create_all()
log.info("DB ready")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  TOOL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)


@functools.lru_cache(maxsize=1024)
def _coords_for(city: str) -> Optional[tuple[float, float]]:
    """Cached geo-lookup via open-meteo endpoint."""
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = resp.json().get("results")
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict:
    """Async weather fetcher."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        resp = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    return resp.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Return a short weather blurb for *location*."""
    coord = _coords_for(location)
    if not coord:
        return f"Location '{location}' not found."
    cw = asyncio.run(_weather_for(*coord))
    return (
        f"The temperature in {location} is {cw.get('temperature')} °C "
        f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
    )


def get_invoice_by_id(invoice_id: str) -> dict:
    """Return invoice data as dict; used by the LLM."""
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    if not inv:
        return {"error": f"Invoice {invoice_id} not found"}
    return {
        "invoice_id": inv.invoice_id,
        "amount": inv.amount,
        "date_due": inv.date_due,
        "status": inv.status,
        "client_name": inv.client.name,
        "client_email": inv.client.email,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "default": "celsius"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_invoice_by_id",
            "description": "Return invoice data by invoice_id",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 4.  RESPONSES-API HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _call_local_tool(call: Any) -> str:
    """Execute requested function and return stringified result."""
    fn_name: str = call.name
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"⛔ invalid JSON: {exc}"

    try:
        if fn_name == "get_current_weather":
            return get_current_weather(**args)
        if fn_name == "get_invoice_by_id":
            return json.dumps(get_invoice_by_id(**args))
        return f"⛔ unknown function '{fn_name}'"
    except Exception as exc:
        log.exception("tool %s raised", fn_name)
        return f"⛔ tool failure: {exc}"


def _sse_stream(tokens: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    """Convert pushed tokens into Server-Sent-Events."""
    keepalive = time.time() + 20
    while True:
        try:
            tok = tokens.get(timeout=1)
            if tok is None:
                yield b"event: done\ndata: [DONE]\n\n"
                break
            yield f"data: {tok}\n\n".encode()
            keepalive = time.time() + 20
        except queue.Empty:
            if time.time() > keepalive:
                yield b": keep-alive\n\n"
                keepalive = time.time() + 20


def _final_stream_to_queue(resp_stream, q: queue.Queue):
    """Forward `response.text.delta` events to queue."""
    for ev in resp_stream:
        if ev.type == "response.text.delta":
            q.put(ev.delta)
    q.put(None)


def _chat_round(user_msg: str,
                prev_resp_id: Optional[str]) -> tuple[openai.types.Response, list[dict]]:
    """Single non-streaming round to capture any tool calls."""
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": user_msg}],
        previous_response_id=prev_resp_id,
        tools=TOOLS,
        stream=False,
    )

    tool_outputs: list[dict] = []
    for item in response.output or []:
        if item.type == "function_call":
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": _call_local_tool(item),
                }
            )
    return response, tool_outputs


def _chat_until_no_tools(user_msg: str) -> openai.types.Stream:
    """
    Resolve all requested tool calls (loop, non-streaming),
    then open one streaming call for the final answer.
    """
    prev_id = session.get("prev_response_id")
    response, tool_outputs = _chat_round(user_msg, prev_id)

    while tool_outputs:
        log.info("LLM requested %d tool call(s)", len(tool_outputs))
        response = client.responses.create(
            model=MODEL,
            input=tool_outputs,
            previous_response_id=response.id,
            tools=TOOLS,
            stream=False,
        )
        tool_outputs = []
        for item in response.output or []:
            if item.type == "function_call":
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": _call_local_tool(item),
                    }
                )

    stream = client.responses.create(
        model=MODEL,
        previous_response_id=response.id,
        stream=True,
    )
    session["prev_response_id"] = response.id
    return stream

# ──────────────────────────────────────────────────────────────────────────────
# 5.  CHAT ENDPOINT (Responses API + SSE)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    q: queue.Queue[str | None] = queue.Queue()

    @copy_current_request_context          # preserves request/session inside thread
    def worker():
        try:
            stream = _chat_until_no_tools(user_msg)
            _final_stream_to_queue(stream, q)
        except Exception as exc:
            log.exception("chat_stream error")
            q.put(f"⚠️ Error: {exc}")
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return Response(
        _sse_stream(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6.  CSV / CRUD (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        df = pd.read_csv(f)
        new_clients: dict[str, Client] = {}
        invoices: list[Invoice] = []
        for _, row in df.iterrows():
            name = row["client_name"]
            cl = Client.query.filter_by(name=name).first() or new_clients.get(name)
            if not cl:
                cl = Client(name=name, email=row.get("client_email") or "")
                new_clients[name] = cl
            invoices.append(
                Invoice(
                    invoice_id=row["invoice_id"],
                    amount=row["amount"],
                    date_due=row["date_due"],
                    status=row["status"],
                    client=cl,
                )
            )
        db.session.add_all(new_clients.values())
        db.session.add_all(invoices)
        db.session.commit()
        flash("Invoices uploaded successfully.")
        return redirect("/")
    return render_template("index.html", invoices=Invoice.query.all())


@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount = request.form["amount"]
    inv.date_due = request.form["date_due"]
    inv.status = request.form["status"]
    inv.client.email = request.form["client_email"]
    db.session.commit()
    return jsonify(
        {
            "client_name": inv.client.name,
            "invoice_id": inv.invoice_id,
            "amount": inv.amount,
            "date_due": inv.date_due,
            "status": inv.status,
        }
    )


@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
    flash("Invoice deleted.")
    return redirect("/")


@app.route("/export")
@app.route("/export/<int:invoice_id>")
def export_invoice(invoice_id: Optional[int] = None):
    rows = [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
    csv_data = pd.DataFrame(
        [
            {
                "client_name": r.client.name,
                "client_email": r.client.email,
                "invoice_id": r.invoice_id,
                "amount": r.amount,
                "date_due": r.date_due,
                "status": r.status,
            }
            for r in rows
        ]
    ).to_csv(index=False)

    fname = f"invoice_{invoice_id or 'all'}.csv"
    return (
        csv_data,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )

# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    log.info("Flask serving on 0.0.0.0:5005 (Responses API, log-level=%s)", LOG_LEVEL)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
