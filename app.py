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
# 0.  LOGGING  (prints ISO 8601 timestamps to stdout)
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("app")

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
    log.critical("❌  OPENAI_API_KEY is not configured.")
    sys.exit(1)
if not ASSISTANT_ID:
    log.critical("❌  ASSISTANT_ID is not configured.")
    sys.exit(1)

# single, warm HTTP connection pool for the lifetime of the app
client = openai.OpenAI(api_key=OPENAI_API_KEY, max_retries=3, timeout=30)
log.info("OpenAI client initialised (model=%s, assistant_id=%s)", MODEL, ASSISTANT_ID)

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
log.info("Database initialised")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  ASSISTANT TOOLS  (fast, cached, async)
# ──────────────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)


@functools.lru_cache(maxsize=2048)
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) from cache or external service."""
    t0 = time.perf_counter()
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = resp.json().get("results")
    log.debug("Geocoding %s took %.3f s", city, time.perf_counter() - t0)
    if not res:
        return None
    return res[0]["latitude"], res[0]["longitude"]


async def _weather_for(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        t0 = time.perf_counter()
        resp = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    log.debug("Weather API call took %.3f s", time.perf_counter() - t0)
    return resp.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Blocking wrapper around async tool: executed synchronously by the
    OpenAI tool-calling runtime.
    """
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
        log.info("Tool get_current_weather(%s) finished in %.3f s", location, time.perf_counter() - t0)


def get_invoice_by_id(invoice_id: str) -> dict:
    """Look up an invoice in the database and return a JSON-serialisable dict."""
    t0 = time.perf_counter()
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    log.info("DB lookup for invoice %s took %.3f s", invoice_id, time.perf_counter() - t0)
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
        log.debug("Created new OpenAI thread: %s", session['thread_id'])
    return session["thread_id"]


def _safe_add_user_message(tid: str, content: str) -> str:
    """
    Add a user message; if the thread is locked by an active run,
    create a new thread transparently.
    """
    try:
        client.beta.threads.messages.create(thread_id=tid, role="user", content=content)
        log.debug("Added user message to thread %s", tid)
        return tid
    except openai.BadRequestError as exc:  # happens if run is still running
        if "while a run" not in str(exc):
            raise
        new_tid = client.beta.threads.create().id
        session["thread_id"] = new_tid
        client.beta.threads.messages.create(thread_id=new_tid, role="user", content=content)
        log.warning("Thread %s locked; switched to new thread %s", tid, new_tid)
        return new_tid


def _run_tool(c: Any) -> str:
    """Execute one tool call and return its output string."""
    try:
        args_raw = getattr(c.function, "arguments", "") or ""
        args = json.loads(args_raw) if args_raw else {}
    except json.JSONDecodeError as exc:
        return f"Invalid JSON for tool '{c.function.name}': {exc}"

    fn = c.function.name
    t0 = time.perf_counter()
    log.info("▶ Running tool %s with args %s", fn, args)
    try:
        if fn == "get_current_weather":
            return get_current_weather(**args)
        if fn == "get_invoice_by_id":
            return json.dumps(get_invoice_by_id(**args))
        return f"Unknown tool {fn}"
    finally:
        log.info("▲ Tool %s finished in %.3f s", fn, time.perf_counter() - t0)


def _arguments_ready(call: Any) -> bool:
    """True if call.function.arguments contains valid JSON."""
    try:
        return bool(call.function.arguments) and json.loads(call.function.arguments) is not None
    except Exception:
        return False


def _tool_call_ready(call: Any) -> bool:
    """Ready when both id and arguments are present."""
    return bool(getattr(call, "id", None)) and _arguments_ready(call)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  BLOCKING CHAT   (/chat)  — with latency logs
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat_sync():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    overall_t0 = time.perf_counter()
    tid = ensure_thread()
    tid = _safe_add_user_message(tid, user_msg)

    run_t0 = time.perf_counter()
    run = client.beta.threads.runs.create(
        thread_id=tid,
        assistant_id=ASSISTANT_ID,
        tools=TOOLS,
        **({"model": MODEL} if MODEL else {}),
    )
    log.info("Run %s created in %.3f s", run.id, time.perf_counter() - run_t0)

    interval = 0.1  # start aggressive, back off gradually
    poll_count = 0
    while True:
        poll_count += 1
        status = client.beta.threads.runs.retrieve(thread_id=tid, run_id=run.id)
        if status.status == "requires_action":
            log.info("Run %s requires_action", run.id)
            calls = status.required_action.submit_tool_outputs.tool_calls
            outputs = [
                {"tool_call_id": c.id, "output": _run_tool(c)}
                for c in calls
                if _tool_call_ready(c)
            ]
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=tid, run_id=run.id, tool_outputs=outputs
            )
            interval = 0.1  # reset interval after tool execution
        elif status.status == "completed":
            log.info("Run %s completed after %d polls", run.id, poll_count)
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            log.error("Run %s ended with status %s", run.id, status.status)
            return jsonify({"error": f"Run {status.status}"}), 500
        time.sleep(interval)
        interval = min(interval * 1.5, 0.5)

    msgs = client.beta.threads.messages.list(thread_id=tid, order="desc")
    for msg in msgs.data:
        if msg.role == "assistant":
            log.info("chat_sync finished in %.3f s (total)", time.perf_counter() - overall_t0)
            return jsonify({"response": msg.content[0].text.value})
    log.error("No assistant response in thread %s", tid)
    return jsonify({"error": "No assistant response"}), 500

# ──────────────────────────────────────────────────────────────────────────────
# 6.  STREAMING CHAT  (/chat/stream)  — key events logged
# ──────────────────────────────────────────────────────────────────────────────
def _extract_tool_calls(
    ev: Any, *, run_id_hint: str | None = None
) -> tuple[list[Any], str] | None:
    """Return (tool_calls, run_id) only when each call has id + valid args."""
    delta = getattr(ev.data, "delta", None)
    if delta:
        tc = getattr(delta, "tool_calls", None)
        if tc and run_id_hint and all(_tool_call_ready(c) for c in tc):
            return tc, run_id_hint
        sd = getattr(delta, "step_details", None)
        if (
            sd
            and getattr(sd, "tool_calls", None)
            and run_id_hint
            and all(_tool_call_ready(c) for c in sd.tool_calls)
        ):
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


def _follow_stream_after_tools(
    run_id: str, calls: Sequence[Any], q: queue.Queue, tid: str
) -> None:
    """Execute finished tools, then keep streaming."""
    log.info("Stream run %s executing %d tool(s)", run_id, len(calls))
    outs = [
        {"tool_call_id": c.id, "output": _run_tool(c)}
        for c in calls
        if _tool_call_ready(c)
    ]
    if not outs:
        return
    follow = client.beta.threads.runs.submit_tool_outputs(
        thread_id=tid, run_id=run_id, tool_outputs=outs, stream=True
    )
    pipe_events(follow, q, tid)


def pipe_events(events, q: queue.Queue, tid: str) -> None:
    run_id: str | None = None
    for ev in events:
        if ev.event == "thread.run.created":
            run_id = ev.data.id
            log.info("Stream run %s created", run_id)
        elif ev.event.startswith("thread.run.step.") and hasattr(ev.data, "run_id"):
            run_id = ev.data.run_id

        if ev.event == "thread.message.delta":
            for part in ev.data.delta.content or []:
                if part.type == "text":
                    q.put(part.text.value)
        elif (
            tc_block := _extract_tool_calls(ev, run_id_hint=run_id)
        ) is not None:
            tool_calls, run_id = tc_block
            _follow_stream_after_tools(run_id, tool_calls, q, tid)
        elif ev.event == "thread.run.completed":
            log.info("Stream run %s completed", run_id)
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
        """Background worker that owns the OpenAI stream."""
        with app.app_context():  # ensure DB access in background thread
            first = client.beta.threads.runs.create(
                thread_id=tid,
                assistant_id=ASSISTANT_ID,
                tools=TOOLS,
                stream=True,
                **({"model": MODEL} if MODEL else {}),
            )
            pipe_events(first, q, tid)
        log.info("chat_stream finished in %.3f s", time.perf_counter() - overall_t0)

    threading.Thread(target=consume, daemon=True).start()

    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 7.  INDEX / CSV / EDIT / EXPORT (unchanged; added a couple of tiny logs)
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
        log.info("CSV import: %d invoices, %d new clients (%.3f s)",
                 len(invoices), len(new_clients), time.perf_counter() - t0)
        flash("Invoices uploaded successfully.")
        return redirect("/")

    return render_template("index.html", invoices=Invoice.query.all())


@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
    log.info("Invoice %d deleted", invoice_id)
    flash("Invoice deleted.")
    return redirect("/")


@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount = request.form["amount"]
    inv.date_due = request.form["date_due"]
    inv.status = request.form["status"]
    inv.client.email = request.form["client_email"]
    db.session.commit()
    log.info("Invoice %d updated", invoice_id)
    return jsonify(
        {
            "client_name": inv.client.name,
            "invoice_id": inv.invoice_id,
            "amount": inv.amount,
            "date_due": inv.date_due,
            "status": inv.status,
        }
    )


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
    log.info("Exported %s", fname)
    return (
        csv_data,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )

# ──────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # Use '0.0.0.0' inside Docker / Kubernetes;
    # disable Flask reloader to prevent duplicate background threads.
    log.info("Starting Flask on 0.0.0.0:5005")
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
