# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (blocking + streaming)
# All comments in English as requested
from __future__ import annotations

import os
import sys
import time
import json
import queue
import threading
import functools
import logging
from collections.abc import Generator
from typing import Any

import httpx
from httpx import Limits, Timeout
from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session, Response
)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import openai
import configparser

import fcc_ecfs  # FCC helper

# ────────────────────────────────────────────────────────────────────────
# 0. LOGGING
# ────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ────────────────────────────────────────────────────────────────────────
# 1. ENV & OPENAI
# ────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL = (cfg.get("DEFAULT", "model",
                 fallback=os.getenv("OPENAI_MODEL", "")).strip() or None)
ASSISTANT_ID = cfg.get("DEFAULT", "assistant_id",
                       fallback=os.getenv("ASSISTANT_ID", "")).strip()

# Propagate FCC key to environment
FCC_KEY = cfg.get("DEFAULT", "FCC_API_KEY",
                  fallback=os.getenv("FCC_API_KEY"))
if FCC_KEY:
    os.environ["FCC_API_KEY"] = FCC_KEY

if not OPENAI_API_KEY:
    log.critical("❌  OPENAI_API_KEY is not configured.")
    sys.exit(1)
if not ASSISTANT_ID:
    log.critical("❌  ASSISTANT_ID is not configured.")
    sys.exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)
log.info("OpenAI client ready (model=%s, assistant=%s)",
         MODEL or "(default)", ASSISTANT_ID)

# ────────────────────────────────────────────────────────────────────────
# 2. FLASK & DB
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# 3. HELPER: THREAD-LOCAL HTTPX CLIENT
# ────────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = Timeout(6.0)
HTTP_LIMITS = Limits(max_keepalive_connections=20, max_connections=50)
import threading
_tls = threading.local()


def _get_client() -> httpx.Client:
    if not hasattr(_tls, "client"):
        _tls.client = httpx.Client(timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS)
    return _tls.client


def _close_thread_client() -> None:
    cli = getattr(_tls, "client", None)
    if cli:
        cli.close()
        delattr(_tls, "client")


# ────────────────────────────────────────────────────────────────────────
# 4. TOOLS
# ────────────────────────────────────────────────────────────────────────
def _coords_for(city: str) -> tuple[float, float] | None:
    cli = _get_client()
    resp = cli.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    try:
        data = resp.json()
    finally:
        resp.close()
    results = data.get("results")
    if not results:
        return None
    return (results[0]["latitude"], results[0]["longitude"])


def _weather_for(lat: float, lon: float) -> dict:
    cli = _get_client()
    resp = cli.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
    )
    try:
        data = resp.json().get("current_weather", {})
    finally:
        resp.close()
    return data


def get_current_weather(location: str, unit: str = "celsius") -> str:
    coord = _coords_for(location)
    if not coord:
        return f"Location '{location}' not found."
    cw = _weather_for(*coord)
    return (
        f"The temperature in {location} is {cw.get('temperature')} °C "
        f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
    )


def get_invoice_by_id(invoice_id: str) -> dict:
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


def fcc_search_filings(company: str) -> list[dict]:
    return fcc_ecfs.search(company)


def fcc_get_filings_text(company: str, indexes: list[int]) -> dict:
    return fcc_ecfs.get_texts(company, indexes)


TOOLS = [
    {"type": "file_search"},
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
    {
        "type": "function",
        "function": {
            "name": "fcc_search_filings",
            "description": "Search FCC ECFS for all PDF attachments of a company",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fcc_get_filings_text",
            "description": "Download & parse selected FCC ECFS PDFs and return text",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "indexes": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "1-based indexes from fcc_search_filings",
                    },
                },
                "required": ["company", "indexes"],
            },
        },
    },
]


# ────────────────────────────────────────────────────────────────────────
# 5. SSE / CHAT STREAMING LOGIC
# ────────────────────────────────────────────────────────────────────────
def ensure_thread() -> str:
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
        log.debug("New thread %s", session["thread_id"])
    return session["thread_id"]


def _safe_add_user_message(thread_id: str, content: str) -> str:
    """Add user message to existing thread. If the thread is locked, create a new one."""
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )
        return thread_id
    except openai.BadRequestError as exc:
        if "while a run" not in str(exc):
            raise
        new_tid = client.beta.threads.create().id
        session["thread_id"] = new_tid
        client.beta.threads.messages.create(
            thread_id=new_tid, role="user", content=content
        )
        log.warning("Thread %s locked → new %s", thread_id, new_tid)
        return new_tid


def _run_tool(c: Any) -> str:
    """Execute the appropriate Python function given a tool call from the model."""
    try:
        args = json.loads(c.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Invalid JSON for arguments: {exc}"

    fn = c.function.name
    if fn == "get_current_weather":
        return get_current_weather(**args)
    elif fn == "get_invoice_by_id":
        return json.dumps(get_invoice_by_id(**args))
    elif fn == "fcc_search_filings":
        return json.dumps(fcc_search_filings(**args), ensure_ascii=False)
    elif fn == "fcc_get_filings_text":
        return json.dumps(fcc_get_filings_text(**args), ensure_ascii=False)
    else:
        return f"Unknown tool {fn}"


def _tool_call_ready(call: Any) -> bool:
    """True if we have a valid tool call with non-empty arguments."""
    if not getattr(call, "id", None):
        return False
    try:
        parsed = json.loads(call.function.arguments or "{}")
        return bool(parsed)
    except Exception:
        return False


def _extract_tool_calls(ev: Any, run_id_hint: str | None = None):
    """Check an event for tool calls. Return (tool_calls, run_id)."""
    delta = getattr(ev.data, "delta", None)
    # 1) If the delta has tool_calls
    if delta and getattr(delta, "tool_calls", None):
        if run_id_hint and all(_tool_call_ready(c) for c in delta.tool_calls):
            return (delta.tool_calls, run_id_hint)

    # 2) Some events have step_details.tool_calls
    step_details = getattr(delta, "step_details", None)
    if step_details and getattr(step_details, "tool_calls", None):
        if run_id_hint and all(_tool_call_ready(c) for c in step_details.tool_calls):
            return (step_details.tool_calls, run_id_hint)

    # 3) If ev.data.step has tool_calls
    step = getattr(ev.data, "step", None)
    if step and getattr(step, "tool_calls", None):
        if all(_tool_call_ready(c) for c in step.tool_calls):
            return (step.tool_calls, step.run_id)

    # 4) required_action-based
    ra = getattr(ev.data, "required_action", None)
    if ra and getattr(ra, "submit_tool_outputs", None):
        t_calls = ra.submit_tool_outputs.tool_calls
        if all(_tool_call_ready(c) for c in t_calls):
            return (t_calls, ev.data.id)

    return None


def _follow_stream_after_tools(run_id: str, calls, q: queue.Queue, tid: str) -> None:
    """Given a set of tool calls, run them and then pass the outputs back to the model."""
    outputs = []
    for c in calls:
        if _tool_call_ready(c):
            result = _run_tool(c)
            outputs.append({"tool_call_id": c.id, "output": result})

    if not outputs:
        return

    # Submit the tool outputs and then stream the new content
    follow_events = client.beta.threads.runs.submit_tool_outputs(
        thread_id=tid,
        run_id=run_id,
        tool_outputs=outputs,
        stream=True,
    )
    # re-use pipe_events to continue streaming
    pipe_events(follow_events, q, tid)


def pipe_events(events, q: queue.Queue, tid: str) -> None:
    """Process each SSE event from the openai client and put text into queue for streaming."""
    run_id: str | None = None

    for ev in events:
        if ev.event == "thread.run.created":
            run_id = ev.data.id
            log.info("New run created: %s", run_id)

        elif ev.event.startswith("thread.run.step.") and hasattr(ev.data, "run_id"):
            run_id = ev.data.run_id

        if ev.event == "thread.message.delta":
            # Gather text tokens
            for part in ev.data.delta.content or []:
                if part.type == "text":
                    # Send each partial text straight to queue
                    q.put(part.text.value)

        # Check for tool calls
        tc_block = _extract_tool_calls(ev, run_id_hint=run_id)
        if tc_block is not None:
            tool_calls, run_id = tc_block
            _follow_stream_after_tools(run_id, tool_calls, q, tid)

        if ev.event == "thread.run.completed":
            # No more tokens coming
            q.put(None)
            break


def sse_generator(q: queue.Queue) -> Generator[bytes, None, None]:
    """
    Simple SSE generator. Yields each chunk from the queue as an SSE 'data:' line.
    """
    while True:
        try:
            chunk = q.get(timeout=30)
        except queue.Empty:
            # Heartbeat
            yield b": keep-alive\n\n"
            continue

        if chunk is None:
            # End of stream
            yield b"event: done\ndata: [DONE]\n\n"
            break

        # Encode to UTF-8, preserving any newlines, etc.
        # The front-end can handle newlines as <br> or with CSS.
        yield f"data: {chunk}\n\n".encode("utf-8")


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    tid = ensure_thread()
    tid = _safe_add_user_message(tid, user_msg)

    q: queue.Queue[str | None] = queue.Queue()

    def consume() -> None:
        try:
            # Create a new 'run' to get model's response
            events = client.beta.threads.runs.create(
                thread_id=tid,
                assistant_id=ASSISTANT_ID,
                tools=TOOLS,
                stream=True,
                **({"model": MODEL} if MODEL else {}),
            )
            pipe_events(events, q, tid)
        finally:
            _close_thread_client()

    threading.Thread(target=consume, daemon=True).start()
    return Response(
        sse_generator(q),
        mimetype="text/event-stream; charset=utf-8",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache"
        },
    )


# ────────────────────────────────────────────────────────────────────────
# 6. CSV / CRUD (same as your version)
# ────────────────────────────────────────────────────────────────────────
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
def export_invoice(invoice_id: int | None = None):
    rows = (
        [Invoice.query.get_or_404(invoice_id)]
        if invoice_id
        else Invoice.query.all()
    )

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


# ────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting app on 0.0.0.0:5005")
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
