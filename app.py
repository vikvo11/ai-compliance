# app.py
# -*- coding: utf-8 -*-
"""
Flask + SQLAlchemy + OpenAI Responses API (stream + tool-calling)
Requires:
  * Python ≥ 3.10
  * openai ≥ 1.7.8  ("responses" + submit_tool_outputs)
  * Flask  ≥ 3.0
  * Flask-SQLAlchemy ≥ 3.1
  * httpx  ≥ 0.27 (for async I/O)
  * pandas ≥ 2.0   (CSV import/export)

Run:
    export OPENAI_API_KEY="sk-..."
    python app.py
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import queue
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from flask import (
    Flask,
    Response,
    copy_current_request_context,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI

# ─────────────────── 0. LOGGING ──────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ─────────────────── 1. OPENAI CONFIG ───────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

aio_client = OpenAI(  # single shared client instance
    api_key=OPENAI_API_KEY,
    timeout=30,
    max_retries=3,
)

# ─────────────────── 2. FLASK & DATABASE ────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "replace-this-secret"),
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{DATA_DIR / 'data.db'}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

db = SQLAlchemy(app)


class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128))
    invoices = db.relationship(
        "Invoice", backref="client", lazy=True, cascade="all, delete-orphan"
    )


class Invoice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_due = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)


with app.app_context():
    db.create_all()

# ─────────────────── 3. TOOL CALLABLES ───────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0, connect=4.0, read=6.0)


@functools.lru_cache(maxsize=1024)
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a given city or None if not found."""
    r = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = r.json().get("results")
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict[str, Any]:
    """Async request to current-weather endpoint (Open-Meteo)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        r = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    r.raise_for_status()
    return r.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Return a human-readable temperature string for *location*."""
    coords = _coords_for(location)
    if not coords:
        return f"Location '{location}' not found."

    cw = asyncio.run(_weather_for(*coords))
    if not cw:
        return "Weather service unavailable."

    temp_c = cw["temperature"]
    if unit == "fahrenheit":
        temp = round(temp_c * 9 / 5 + 32, 1)
        suffix = "°F"
    else:
        temp = temp_c
        suffix = "°C"
    return f"The temperature in {location} is {temp} {suffix}."


def get_invoice_by_id(invoice_id: str) -> dict[str, Any]:
    """Look up *invoice_id* and return invoice dict or error dict."""
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


# Strict JSON-schema definitions for the model
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Return current temperature for a city (°C by default).",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "celsius",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_invoice_by_id",
        "description": "Return invoice data for the given invoice number.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "Invoice #"}
            },
            "required": ["invoice_id"],
            "additionalProperties": False,
        },
    },
]

# Map tool names to the local callables.
DISPATCH: dict[str, callable] = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id": get_invoice_by_id,
}

# ─────────────────── 4. TOOL-CALL PIPELINE ───────────────────

def _run_tool(name: str, args_json: str) -> str:
    """Execute *name* using JSON-encoded *args_json* and return string."""
    try:
        params = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return f"Bad JSON args: {args_json!r}"

    fn = DISPATCH.get(name)
    if not fn:
        return f"Unknown tool: {name}"

    try:
        result = fn(**params)
    except Exception as exc:  # noqa: BLE001
        log.exception("Tool %s failed", name)
        result = f"error:{type(exc).__name__}:{exc}"

    return json.dumps(result) if isinstance(result, dict) else str(result)


def _finish_tool(
    response_id: str,
    thread_id: str | None,
    run_id: str | None,
    tool_call_id: str,
    name: str,
    args_json: str,
    q: queue.Queue[str | None],
) -> str:
    """Run tool, submit output, continue piping assistant stream."""
    log.info("Executing tool=%s id=%s", name, tool_call_id)
    output = _run_tool(name, args_json)

    follow_stream = aio_client.responses.submit_tool_outputs(
        response_id=response_id,
        tool_outputs=[{"tool_call_id": tool_call_id, "output": output}],
        stream=True,
    )
    return _pipe(follow_stream, q, thread_id, run_id)


def _pipe(  # noqa: C901
    stream: Any,
    q: queue.Queue[str | None],
    thread_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Parse streaming events from Responses API and write to *q*."""
    response_id = ""
    buffered: dict[str, dict[str, Any]] = {}

    for ev in stream:
        ev_type = getattr(ev, "type", getattr(ev, "event", "")) or ""
        delta = getattr(ev, "delta", None)

        # Persist primary IDs.
        if getattr(ev, "response", None):
            response_id = ev.response.id
            thread_id = getattr(ev.response, "thread_id", thread_id)
            run_id = getattr(ev.response, "run_id", run_id)

        # Plain text → push to SSE queue.
        if isinstance(delta, str) and "arguments" not in ev_type:
            q.put(delta)

        # Legacy schema: ev.tool_calls
        for tc in getattr(ev, "tool_calls", []) or []:
            name = getattr(tc, "function", tc).name  # nested vs flat
            iid = tc.id
            buffered[iid] = {"name": name, "parts": []}
            full = getattr(tc, "function", tc).arguments
            if full:
                response_id = _finish_tool(
                    response_id, thread_id, run_id, iid, name, full, q
                )

        # New schema: response.output_item.added
        if ev_type == "response.output_item.added":
            item = (getattr(ev, "item", None) or ev.model_dump()).get("item")
            if item and item.get("type") in ("function_call", "tool_call"):
                buffered[item["id"]] = {"name": item["name"], "parts": []}

        # Collect argument deltas.
        if "arguments.delta" in ev_type:
            iid = (
                getattr(ev, "item_id", None) or getattr(ev, "tool_call_id", None)
            )
            if iid in buffered:
                buffered[iid]["parts"].append(delta or "")

        # End-of-arguments → execute tool.
        if ev_type.endswith("arguments.done"):
            iid = (
                getattr(ev, "item_id", None) or getattr(ev, "tool_call_id", None)
            )
            if iid in buffered:
                full_args = "".join(buffered[iid]["parts"])
                response_id = _finish_tool(
                    response_id,
                    thread_id,
                    run_id,
                    iid,
                    buffered[iid]["name"],
                    full_args,
                    q,
                )
                buffered.pop(iid, None)

        # Stream finished → close SSE.
        if ev_type in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)
            return response_id

    # Edge-case: stream ended silently.
    q.put(None)
    return response_id

# ─────────────────── 5. SSE HELPER ──────────────────────────

def sse(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    """Yield Server-Sent Events for each token and keep-alive pings."""
    next_ping = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)
            if tok is None:
                yield b"event: done\ndata: [DONE]\n\n"
                break
            yield f"data: {tok}\n\n".encode()
            next_ping = time.time() + 20
        except queue.Empty:
            if time.time() >= next_ping:
                yield b": ping\n\n"
                next_ping = time.time() + 20

# ─────────────────── 6. CHAT ENDPOINT ───────────────────────

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """POST JSON {message:"…"} → SSE stream with assistant reply."""
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    last_resp_id = session.get("prev_response_id")
    q: queue.Queue[str | None] = queue.Queue()

    @copy_current_request_context
    def worker() -> None:
        messages = [{"role": "user", "content": user_msg}]
        stream = aio_client.responses.create(
            model=MODEL,
            input=messages,
            previous_response_id=last_resp_id,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )
        session["prev_response_id"] = _pipe(stream, q)

    # Run in background thread to keep request thread free.
    import threading

    threading.Thread(target=worker, daemon=True).start()
    return Response(
        sse(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ─────────────────── 7. CSV / CRUD ROUTES ───────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.lower().endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        t0 = time.perf_counter()
        df = pd.read_csv(f)

        new_clients: dict[str, Client] = {}
        invoices: list[Invoice] = []
        for _, row in df.iterrows():
            name = row["client_name"]
            client = Client.query.filter_by(name=name).first() or new_clients.get(name)
            if not client:
                client = Client(name=name, email=row.get("client_email", ""))
                new_clients[name] = client
            invoices.append(
                Invoice(
                    invoice_id=row["invoice_id"],
                    amount=row["amount"],
                    date_due=row["date_due"],
                    status=row["status"],
                    client=client,
                )
            )

        db.session.add_all(new_clients.values())
        db.session.add_all(invoices)
        db.session.commit()
        log.info(
            "CSV import: %d invoices, %d new clients (%.3f s)",
            len(invoices),
            len(new_clients),
            time.perf_counter() - t0,
        )
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


@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
    flash("Invoice deleted.")
    log.info("Invoice %d deleted", invoice_id)
    return redirect("/")


@app.route("/export")
@app.route("/export/<int:invoice_id>")
def export_invoice(invoice_id: int | None = None):
    rows = [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
    df = pd.DataFrame(
        {
            "client_name": r.client.name,
            "client_email": r.client.email,
            "invoice_id": r.invoice_id,
            "amount": r.amount,
            "date_due": r.date_due,
            "status": r.status,
        }
        for r in rows
    )
    csv_data = df.to_csv(index=False)

    fname = f"invoice_{invoice_id or 'all'}.csv"
    log.info("Exporting %s", fname)
    return (
        csv_data,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )

# ─────────────────── 8. MAIN ────────────────────────────────

if __name__ == "__main__":
    import openai as _oa

    ver_tuple = tuple(map(int, _oa.__version__.split(".")[:2]))
    if ver_tuple < (1, 7):
        sys.exit("openai-python ≥ 1.7.0 is required (pip install --upgrade openai)")

    log.info("openai-python version: %s", _oa.__version__)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
