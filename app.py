# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (blocking + streaming, new SDK)
# All comments in English as requested
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
from collections.abc import Sequence, Generator
from typing import Any, Optional

import httpx                     # faster HTTP client, async-friendly
from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session, Response
)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import openai
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
# 1.  ENV & OPENAI
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL = (cfg.get("DEFAULT", "model",
                 fallback=os.getenv("OPENAI_MODEL", "")).strip() or None)
ASSISTANT_ID = cfg.get("DEFAULT", "assistant_id",
                       fallback=os.getenv("ASSISTANT_ID", "")).strip()

if not OPENAI_API_KEY:
    log.critical("❌  OPENAI_API_KEY is not configured.")
    sys.exit(1)
if not ASSISTANT_ID:
    log.critical("❌  ASSISTANT_ID is not configured.")
    sys.exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)
log.info("OpenAI client ready (model=%s, assistant=%s)", MODEL or "(default)", ASSISTANT_ID)

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
# 3.  ASSISTANT TOOLS (cached + async)
# ──────────────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)


@functools.lru_cache(maxsize=2048)
def _coords_for(city: str) -> tuple[float, float] | None:
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
        return (
            f"The temperature in {location} is {cw.get('temperature')} °C "
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
# 4.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ensure_thread() -> str:
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
        log.debug("new thread %s", session["thread_id"])
    return session["thread_id"]


def _safe_add_user_message(tid: str, content: str) -> str:
    try:
        client.beta.threads.messages.create(thread_id=tid, role="user", content=content)
        return tid
    except openai.BadRequestError as exc:
        if "while a run" not in str(exc):
            raise
        new_tid = client.beta.threads.create().id
        session["thread_id"] = new_tid
        client.beta.threads.messages.create(thread_id=new_tid, role="user", content=content)
        log.warning("thread %s locked → new %s", tid, new_tid)
        return new_tid


def _run_tool(c: Any) -> str:
    try:
        args = json.loads(c.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    fn = c.function.name
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


def _arguments_ready(call: Any) -> bool:
    try:
        return bool(call.function.arguments) and json.loads(call.function.arguments) is not None
    except Exception:
        return False


def _tool_call_ready(call: Any) -> bool:
    return bool(getattr(call, "id", None)) and _arguments_ready(call)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  STREAMING CHAT  (detailed timing)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_tool_calls(ev: Any, *, run_id_hint: str | None = None):
    delta = getattr(ev.data, "delta", None)
    if delta:
        tc = getattr(delta, "tool_calls", None)
        if tc and run_id_hint and all(_tool_call_ready(c) for c in tc):
            return tc, run_id_hint
        sd = getattr(delta, "step_details", None)
        if sd and getattr(sd, "tool_calls", None) and run_id_hint and \
                all(_tool_call_ready(c) for c in sd.tool_calls):
            return sd.tool_calls, run_id_hint

    step = getattr(ev.data, "step", None)
    if step and getattr(step, "tool_calls", None):
        tc = step.tool_calls
        if all(_tool_call_ready(c) for c in tc):
            return tc, step.run_id

    ra = getattr(ev.data, "required_action", None)
    if ra and getattr(ra, "submit_tool_outputs", None):
        tc = ra.submit_tool_outputs.tool_calls
        if all(_tool_call_ready(c) for c in tc):
            return tc, ev.data.id
    return None


def _follow_stream_after_tools(run_id: str, calls, q: queue.Queue, tid: str):
    outs = [{"tool_call_id": c.id, "output": _run_tool(c)} for c in calls if _tool_call_ready(c)]
    if not outs:
        return
    follow = client.beta.threads.runs.submit_tool_outputs(
        thread_id=tid, run_id=run_id, tool_outputs=outs, stream=True
    )
    pipe_events(follow, q, tid)


def pipe_events(events, q: queue.Queue, tid: str):
    run_id: str | None = None
    t0 = time.perf_counter()
    first_tok: float | None = None
    n_frag = 0

    for ev in events:
        if ev.event == "thread.run.created":
            run_id = ev.data.id
            log.info("run %s created", run_id)
        elif ev.event.startswith("thread.run.step.") and hasattr(ev.data, "run_id"):
            run_id = ev.data.run_id

        if ev.event == "thread.message.delta":
            for part in ev.data.delta.content or []:
                if part.type == "text":
                    if first_tok is None:
                        first_tok = time.perf_counter()
                        log.info("run %s first-token %.3f s", run_id, first_tok - t0)
                    n_frag += 1
                    q.put(part.text.value)

        elif (tc_block := _extract_tool_calls(ev, run_id_hint=run_id)) is not None:
            tool_calls, run_id = tc_block
            _follow_stream_after_tools(run_id, tool_calls, q, tid)

        elif ev.event == "thread.run.completed":
            log.info("run %s done (%d frags, %.3f s total)", run_id, n_frag, time.perf_counter() - t0)
            q.put(None)
            return


def sse_generator(q: queue.Queue) -> Generator[bytes, None, None]:
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


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    tid = ensure_thread()
    tid = _safe_add_user_message(tid, user_msg)
    overall_t0 = time.perf_counter()

    q: queue.Queue[str | None] = queue.Queue()

    def consume() -> None:
        with app.app_context():
            first = client.beta.threads.runs.create(
                thread_id=tid, assistant_id=ASSISTANT_ID,
                tools=TOOLS, stream=True,
                **({"model": MODEL} if MODEL else {}),
            )
            pipe_events(first, q, tid)
        log.info("chat_stream %.3f s", time.perf_counter() - overall_t0)

    threading.Thread(target=consume, daemon=True).start()
    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6.  CSV / CRUD  (with logs)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    log.info("Flask serving on 0.0.0.0:5005  (log-level=%s)", LOG_LEVEL)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
