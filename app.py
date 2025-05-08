# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (blocking + streaming)
from __future__ import annotations

import os
import sys
import time
import json
import queue
import threading
import asyncio
import functools
import logging
from collections.abc import Generator
from typing import Any

import httpx
from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session, Response, copy_current_request_context
)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import openai
import configparser

# ─────────────────────────────────────────────────────────────
# 0. LOGGING (ISO-8601 → stdout)
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────
# 1. ENV & OPENAI
# ─────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL = (cfg.get("DEFAULT", "model",
                 fallback=os.getenv("OPENAI_MODEL", "")).strip() or "gpt-4o-mini")

if not OPENAI_API_KEY:
    log.critical("❌  OPENAI_API_KEY is not configured.")
    sys.exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)
log.info("OpenAI client ready (model=%s)", MODEL)

# ─────────────────────────────────────────────────────────────
# 2. FLASK & DB
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 3. TOOLS (cached + async)
# ─────────────────────────────────────────────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)


@functools.lru_cache(maxsize=2048)
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a city using Open-Meteo geocoder."""
    t0 = time.perf_counter()
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = resp.json().get("results")
    log.debug("geocode '%s' %.3f s", city, time.perf_counter() - t0)
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict:
    """Async call to current-weather endpoint."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        t0 = time.perf_counter()
        resp = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    log.debug("weather API %.3f s", time.perf_counter() - t0)
    return resp.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    t0 = time.perf_counter()
    try:
        coord = _coords_for(location)
        if not coord:
            return f"Location '{location}' not found."
        cw = asyncio.run(_weather_for(*coord))
        temp = cw.get("temperature")
        if unit == "fahrenheit":
            temp = round(temp * 9 / 5 + 32, 1)
            unit_sign = "°F"
        else:
            unit_sign = "°C"
        return (
            f"The temperature in {location} is {temp} {unit_sign} "
            f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
        )
    finally:
        log.info("tool:get_current_weather('%s') %.3f s", location, time.perf_counter() - t0)


def get_invoice_by_id(invoice_id: str) -> dict:
    t0 = time.perf_counter()
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    log.info("tool:get_invoice_by_id '%s' DB %.3f s", invoice_id, time.perf_counter() - t0)
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
        "name": "get_current_weather",
        "type": "function",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_invoice_by_id",
        "type": "function",
        "description": "Return invoice data by invoice_id",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "Invoice identifier"}
            },
            "required": ["invoice_id"],
        },
    },
]

# ─────────────────────────────────────────────────────────────
# 4. STREAM HELPERS
# ─────────────────────────────────────────────────────────────
def _run_tool(call: dict[str, Any]) -> str:
    fn = call["name"]
    args = call["args"]
    t0 = time.perf_counter()
    log.info("▶ tool %s %s", fn, args)
    try:
        if fn == "get_current_weather":
            return get_current_weather(**args)
        if fn == "get_invoice_by_id":
            return json.dumps(get_invoice_by_id(**args))
        return f"Unknown tool {fn}"
    finally:
        log.info("▲ tool %s done %.3f s", fn, time.perf_counter() - t0)


def _pipe_events(events, q: queue.Queue[str | None]) -> str | None:
    """
    Push Responses-API stream into SSE queue.
    Executes tools and resumes streaming after submit_tool_outputs.
    Returns final response_id for conversation state.
    """
    response_id: str | None = None
    pending: dict[str, dict] = {}   # tool_call_id → {name, args_str, parsed?}
    finished: set[str] = set()

    for ev in events:
        typ = getattr(ev, "type", None)

        if typ == "response.created":
            response_id = ev.response.id

        elif typ == "response.output_text.delta":
            if ev.delta:
                q.put(ev.delta)

        elif typ == "response.tool_calls":
            for tc in ev.tool_calls or []:
                pending[tc.id] = {"name": tc.function.name, "args": ""}

        elif typ == "response.tool_calls.arguments.delta":
            tc_id = ev.tool_call_id
            if tc_id in pending:
                pending[tc_id]["args"] += ev.delta or ""
                try:
                    pending[tc_id]["parsed"] = json.loads(pending[tc_id]["args"])
                except json.JSONDecodeError:
                    pass  # wait for more chunks

        elif typ == "response.tool_calls.done":
            tool_outputs = []
            for tc in ev.tool_calls or []:
                if tc.id in finished:
                    continue
                meta = pending.get(tc.id)
                if not meta or "parsed" not in meta:
                    continue
                finished.add(tc.id)
                out = _run_tool({"name": meta["name"], "args": meta["parsed"]})
                tool_outputs.append({"tool_call_id": tc.id, "output": out})

            if tool_outputs and response_id:
                follow = client.responses.submit_tool_outputs(
                    response_id=response_id,
                    tool_outputs=tool_outputs,
                    stream=True,
                )
                response_id = _pipe_events(follow, q) or response_id

        elif typ == "response.done":
            q.put(None)
            return response_id

        else:
            log.debug("⏺  unhandled event %s", typ)

    return response_id


def sse_generator(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    """Convert queue entries to SSE, keep-alive every 20 s."""
    heartbeat_at = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)
            if tok is None:
                yield b"event: done\ndata: [DONE]\n\n"
                break
            yield f"data: {tok}\n\n".encode()
            heartbeat_at = time.time() + 20
        except queue.Empty:
            if time.time() > heartbeat_at:
                yield b": keep-alive\n\n"
                heartbeat_at = time.time() + 20

# ─────────────────────────────────────────────────────────────
# 5. CHAT STREAM ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    last_resp_id = session.get("prev_response_id")
    q: queue.Queue[str | None] = queue.Queue()
    overall_t0 = time.perf_counter()

    @copy_current_request_context
    def consume() -> None:
        initial = client.responses.create(
            model=MODEL,
            input=user_msg,
            previous_response_id=last_resp_id,
            tools=TOOLS,
            stream=True,
        )
        new_resp_id = _pipe_events(initial, q)
        if new_resp_id:
            session["prev_response_id"] = new_resp_id
        log.info("chat_stream %.3f s", time.perf_counter() - overall_t0)

    threading.Thread(target=consume, daemon=True).start()
    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ─────────────────────────────────────────────────────────────
# 6. CSV / CRUD  (unchanged)
# ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        t0 = time.perf_counter()
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
        log.info("CSV import %d invoices, %d new clients %.3f s",
                 len(invoices), len(new_clients), time.perf_counter() - t0)
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
    log.info("invoice %d updated", invoice_id)
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
    log.info("invoice %d deleted", invoice_id)
    flash("Invoice deleted.")
    return redirect("/")


@app.route("/export")
@app.route("/export/<int:invoice_id>")
def export_invoice(invoice_id: int | None = None):
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
    log.info("export %s", fname)
    return (
        csv_data,
        200,
        {"Content-Type": "text/csv",
         "Content-Disposition": f'attachment; filename="{fname}"'},
    )

# ─────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    log.info("Flask serving on 0.0.0.0:5005  (log-level=%s)", LOG_LEVEL)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
