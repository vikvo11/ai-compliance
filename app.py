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
import logging                       #  ← NEW
from collections.abc import Sequence, Generator
from typing import Any, Optional

import httpx                         # faster HTTP client, async-friendly
from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session, Response
)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import openai
import configparser

# ──────────────────────────────────────────────────────────────────────────────
# 0.  LOGGING  (prints go to stdout → Docker logs / journald)
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("latency")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ENV & OPENAI
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY: str | None = cfg.get(
    "DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY")
)
MODEL: Optional[str] = cfg.get(
    "DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "")
).strip() or None
ASSISTANT_ID: str = cfg.get(
    "DEFAULT", "assistant_id", fallback=os.getenv("ASSISTANT_ID", "")
).strip()

if not OPENAI_API_KEY:
    sys.exit("❌  OPENAI_API_KEY is not configured.")
if not ASSISTANT_ID:
    sys.exit("❌  ASSISTANT_ID is not configured.")

client = openai.OpenAI(api_key=OPENAI_API_KEY, max_retries=3, timeout=30)

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
        "Invoice",
        backref="client",
        lazy=True,
        cascade="all, delete-orphan",
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

# ──────────────────────────────────────────────────────────────────────────────
# 3.  ASSISTANT TOOLS  (fast, cached, async) + logging
# ──────────────────────────────────────────────────────────────────────────────
_COORD_CACHE: dict[str, tuple[float, float]] = {}
HTTP_TIMEOUT = httpx.Timeout(6.0)


def _timed(label: str):
    """Small decorator that logs the runtime of the wrapped function."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                log.info("%s took %.2f s", label, time.perf_counter() - t0)
        return wrapper
    return deco


@functools.lru_cache(maxsize=2048)
@_timed("geocoding")                # ← log once per non-cached city
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) from cache or external service."""
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = resp.json().get("results")
    if not res:
        return None
    return res[0]["latitude"], res[0]["longitude"]


@_timed("weather-api")
async def _weather_for(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        resp = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    return resp.json().get("current_weather", {})


@_timed("tool:get_current_weather")
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Blocking wrapper around async tool, logs total runtime."""
    try:
        coord = _coords_for(location)
        if not coord:
            return f"Location '{location}' not found."
        cw = asyncio.run(_weather_for(*coord))
        return (
            f"The temperature in {location} is {cw.get('temperature')} °C "
            f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching weather: {exc}"


@_timed("tool:get_invoice_by_id")
def get_invoice_by_id(invoice_id: str) -> dict:
    """Look up an invoice in the database and return a JSON-serialisable dict."""
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
# 4.  SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ensure_thread() -> str:
    """Return an OpenAI thread ID stored in session, creating one if missing."""
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
    return session["thread_id"]


def _safe_add_user_message(tid: str, content: str) -> str:
    """Add a user message; if the thread is locked create a new one."""
    try:
        client.beta.threads.messages.create(thread_id=tid, role="user", content=content)
        return tid
    except openai.BadRequestError as exc:
        if "while a run" not in str(exc):
            raise
        new_tid = client.beta.threads.create().id
        session["thread_id"] = new_tid
        client.beta.threads.messages.create(thread_id=new_tid, role="user", content=content)
        return new_tid


def _run_tool(c: Any) -> str:
    """Execute one tool call and return its output string, with timing."""
    t0 = time.perf_counter()
    try:
        args_raw = getattr(c.function, "arguments", "") or ""
        args = json.loads(args_raw) if args_raw else {}
    except json.JSONDecodeError as exc:
        return f"Invalid JSON for tool '{c.function.name}': {exc}"

    fn = c.function.name
    if fn == "get_current_weather":
        result = get_current_weather(**args)
    elif fn == "get_invoice_by_id":
        result = json.dumps(get_invoice_by_id(**args))
    else:
        result = f"Unknown tool {fn}"
    log.info("tool:%s finished in %.2f s", fn, time.perf_counter() - t0)
    return result


def _arguments_ready(call: Any) -> bool:
    try:
        return bool(call.function.arguments) and json.loads(call.function.arguments) is not None
    except Exception:
        return False


def _tool_call_ready(call: Any) -> bool:
    return bool(getattr(call, "id", None)) and _arguments_ready(call)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  BLOCKING CHAT   (/chat)  — with latency logs
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat_sync():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    log.info("chat_sync: ⇢ new request (len=%d)", len(user_msg))
    tt0 = time.perf_counter()

    tid = ensure_thread()
    tid = _safe_add_user_message(tid, user_msg)

    run = client.beta.threads.runs.create(
        thread_id=tid,
        assistant_id=ASSISTANT_ID,
        tools=TOOLS,
        **({"model": MODEL} if MODEL else {}),
    )
    log.info("chat_sync: run created in %.2f s", time.perf_counter() - tt0)

    interval = 0.1
    while True:
        status = client.beta.threads.runs.retrieve(thread_id=tid, run_id=run.id)
        if status.status == "requires_action":
            calls = status.required_action.submit_tool_outputs.tool_calls
            outputs = [
                {"tool_call_id": c.id, "output": _run_tool(c)}
                for c in calls
                if _tool_call_ready(c)
            ]
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=tid, run_id=run.id, tool_outputs=outputs
            )
            interval = 0.1
        elif status.status == "completed":
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            log.warning("chat_sync: run finished with status %s", status.status)
            return jsonify({"error": f"Run {status.status}"}), 500
        time.sleep(interval)
        interval = min(interval * 1.5, 0.5)

    log.info("chat_sync: run total %.2f s", time.perf_counter() - tt0)

    msgs = client.beta.threads.messages.list(thread_id=tid, order="desc")
    for msg in msgs.data:
        if msg.role == "assistant":
            return jsonify({"response": msg.content[0].text.value})
    return jsonify({"error": "No assistant response"}), 500

# ──────────────────────────────────────────────────────────────────────────────
# 6.  STREAMING CHAT  (/chat/stream)  — extra logs for TTFB
# ──────────────────────────────────────────────────────────────────────────────
def _extract_tool_calls(ev: Any, *, run_id_hint: str | None = None
                        ) -> tuple[list[Any], str] | None:
    ...
    # (function unchanged – kept for brevity)
    ...


def _follow_stream_after_tools(run_id: str, calls: Sequence[Any],
                               q: queue.Queue, tid: str) -> None:
    ...
    # (function unchanged)
    ...


def pipe_events(events, q: queue.Queue, tid: str) -> None:
    run_id: str | None = None
    first_token_logged = False
    start = time.perf_counter()

    for ev in events:
        if ev.event == "thread.run.created":
            run_id = ev.data.id
        elif ev.event.startswith("thread.run.step.") and hasattr(ev.data, "run_id"):
            run_id = ev.data.run_id

        if ev.event == "thread.message.delta":
            for part in ev.data.delta.content or []:
                if part.type == "text":
                    if not first_token_logged:
                        log.info("chat_stream: first token in %.2f s", time.perf_counter() - start)
                        first_token_logged = True
                    q.put(part.text.value)
        elif (tc_block := _extract_tool_calls(ev, run_id_hint=run_id)) is not None:
            tool_calls, run_id = tc_block
            _follow_stream_after_tools(run_id, tool_calls, q, tid)
        elif ev.event == "thread.run.completed":
            log.info("chat_stream: run completed in %.2f s", time.perf_counter() - start)
            q.put(None)
            return


def sse_generator(q: queue.Queue) -> Generator[bytes, None, None]:
    ...
    # (function unchanged)
    ...


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    log.info("chat_stream: ⇢ new request (len=%d)", len(user_msg))
    tid = ensure_thread()
    tid = _safe_add_user_message(tid, user_msg)

    q: queue.Queue[str | None] = queue.Queue()

    def consume() -> None:
        with app.app_context():
            first = client.beta.threads.runs.create(
                thread_id=tid,
                assistant_id=ASSISTANT_ID,
                tools=TOOLS,
                stream=True,
                **({"model": MODEL} if MODEL else {}),
            )
            pipe_events(first, q, tid)

    threading.Thread(target=consume, daemon=True).start()

    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
