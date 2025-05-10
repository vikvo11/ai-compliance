# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (stream + tool calling, strict mode)
# Works on ANY openai-python ≥ 1.4  (falls back if submit_tool_outputs is missing)

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
import pandas as pd
import configparser
import openai
from openai import OpenAI
from flask import (
    Flask, render_template, request, redirect, flash,
    jsonify, session, Response, copy_current_request_context
)
from flask_sqlalchemy import SQLAlchemy

# ─────────────────── 0. LOGGING ──────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ─────────────────── 1. OPENAI ───────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL = cfg.get("DEFAULT", "model",       fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY missing")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# detect whether submit_tool_outputs exists
_HAS_SUBMIT = hasattr(client.responses, "submit_tool_outputs")
log.info("submit_tool_outputs available: %s", _HAS_SUBMIT)

# ─────────────────── 2. FLASK & DB ───────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:////app/data/data.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5 MB upload limit
)
db = SQLAlchemy(app)
os.makedirs("/app/data", exist_ok=True)


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

# ─────────────────── 3. TOOL FUNCTIONS ───────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0, connect=4.0, read=6.0)


@functools.lru_cache(maxsize=1024)
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a given city; None if not found."""
    r = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    items = r.json().get("results")
    return None if not items else (items[0]["latitude"], items[0]["longitude"])


async def _weather_async(lat: float, lon: float) -> dict:
    """Async request to current-weather endpoint."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        r = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    return r.json().get("current_weather", {})


def _sync(coro: asyncio.Future) -> Any:
    """Run coroutine regardless of existing event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Return temperature string; convert units if needed."""
    coords = _coords_for(location)
    cw = _sync(_weather_async(*coords)) if coords else None
    if not cw:
        return f"Location '{location}' not found."
    temp = cw["temperature"]
    temp = round(temp * 9 / 5 + 32, 1) if unit == "fahrenheit" else temp
    return f"The temperature in {location} is {temp} {'°F' if unit == 'fahrenheit' else '°C'}."


def get_invoice_by_id(invoice_id: str) -> dict:
    """Retrieve invoice or return error dict."""
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    return (
        {"error": f"Invoice {invoice_id} not found"}
        if not inv
        else {
            "invoice_id": inv.invoice_id,
            "amount": inv.amount,
            "date_due": inv.date_due,
            "status": inv.status,
            "client_name": inv.client.name,
            "client_email": inv.client.email,
        }
    )

# ─────────────────── 3a. TOOLS SCHEMA ────────────────────────
TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather for a city.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": ["string", "null"],
                    "enum": ["celsius", "fahrenheit", None],
                    "description": "Temperature unit (default celsius).",
                    "default": "celsius",
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_invoice_by_id",
        "description": "Return invoice data for a given invoice number.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string", "description": "Invoice identifier"}
            },
            "required": ["invoice_id"],
            "additionalProperties": False,
        },
    },
]

DISPATCH = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id": get_invoice_by_id,
}

# ─────────────────── 4. STREAM HANDLER ───────────────────────
def _run_tool(name: str, args_json: str) -> str:
    """Execute a tool and return str / JSON-string."""
    try:
        params = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return f"Bad JSON: {args_json}"
    fn = DISPATCH.get(name)
    res = fn(**params) if fn else f"Unknown tool {name}"
    return json.dumps(res) if isinstance(res, dict) else str(res)


def _finish_tool(
    response_id: str,
    thread_id: str | None,
    run_id: str | None,
    tool_call_id: str,
    name: str,
    args_json: str,
    q: queue.Queue[str | None],
) -> str:
    """Run tool, send result back (new API or fallback), then continue stream."""
    log.info("tool=%s id=%s args=%s", name, tool_call_id, args_json)
    output = _run_tool(name, args_json)

    if _HAS_SUBMIT:
        # Preferred path (openai>=1.7)
        follow = client.responses.submit_tool_outputs(
            response_id=response_id,
            tool_outputs=[{"tool_call_id": tool_call_id, "output": output}],
            stream=True,
        )
    else:
        # Fallback: post a new assistant turn with function_call_output
        follow = client.responses.create(
            model=MODEL,
            thread_id=thread_id,
            run_id=run_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": output,
                }
            ],
            tools=TOOLS,
            stream=True,
        )

    return _pipe(follow, q, thread_id, run_id)


def _pipe(  # noqa: C901
    stream,
    q: queue.Queue[str | None],
    thread_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Parse streaming events and push tokens to SSE queue."""
    response_id = ""
    buf: dict[str, dict[str, Any]] = {}

    for ev in stream:
        typ = getattr(ev, "event", None) or getattr(ev, "type", "") or ""
        delta = getattr(ev, "delta", None)

        thread_id = getattr(ev, "thread_id", thread_id)
        run_id = getattr(ev, "run_id", run_id)

        if getattr(ev, "response", None):
            response_id = ev.response.id
            thread_id = getattr(ev.response, "thread_id", thread_id)
            run_id = getattr(ev.response, "run_id", run_id)

        # Plain text tokens
        if isinstance(delta, str) and "arguments" not in typ:
            q.put(delta)

        # ── old schema: response.tool_calls ───────────────────
        if getattr(ev, "tool_calls", None):
            for tc in ev.tool_calls:
                fn_name = (
                    getattr(tc, "function", None).name
                    if getattr(tc, "function", None)
                    else getattr(tc, "name", None)
                )
                full_args = (
                    getattr(tc, "function", None).arguments
                    if getattr(tc, "function", None)
                    else getattr(tc, "arguments", None)
                )
                buf[tc.id] = {"name": fn_name, "parts": []}
                if full_args:
                    response_id = _finish_tool(
                        response_id, thread_id, run_id,
                        tc.id, fn_name, full_args, q
                    )

        # ── new schema: output_item.added ─────────────────────
        if typ == "response.output_item.added":
            item = (
                getattr(ev, "item", None)
                or getattr(ev, "output_item", None)
                or ev.model_dump(exclude_none=True).get("item")
            )
            item = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
            if item and item.get("type") in ("function_call", "tool_call"):
                iid = item["id"]
                fname = (
                    item.get("function", {}).get("name")
                    or item.get("tool_call", {}).get("name")
                    or item.get("name")
                )
                buf[iid] = {"name": fname, "parts": []}

        # Accumulate argument deltas
        if "arguments.delta" in typ:
            iid = (
                getattr(ev, "output_item_id", None)
                or getattr(ev, "item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                buf[iid]["parts"].append(delta or "")

        # End-of-arguments → execute tool
        if typ.endswith("arguments.done"):
            iid = (
                getattr(ev, "output_item_id", None)
                or getattr(ev, "item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                full_json = "".join(buf[iid]["parts"])
                response_id = _finish_tool(
                    response_id, thread_id, run_id,
                    iid, buf[iid]["name"], full_json, q
                )
                buf.pop(iid, None)

        # Final markers
        if typ in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)
            return response_id

    q.put(None)
    return response_id


# ─────────────────── 5. SSE GENERATOR ────────────────────────
def sse(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    """Convert queue to Server-Sent Events with 20 s keep-alive."""
    keep_alive = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)
            if tok is None:
                yield b"event: done\ndata: [DONE]\n\n"
                break
            yield f"data: {tok}\n\n".encode()
            keep_alive = time.time() + 20
        except queue.Empty:
            if time.time() > keep_alive:
                yield b": ping\n\n"
                keep_alive = time.time() + 20


# ─────────────────── 6. CHAT ENDPOINT ───────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """POST JSON {message: ...} → SSE assistant stream."""
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    last_resp_id = session.get("prev_response_id")
    q: queue.Queue[str | None] = queue.Queue()

    @copy_current_request_context
    def work() -> None:
        stream = client.responses.create(
            model=MODEL,
            input=msg,
            previous_response_id=last_resp_id,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=True,
        )
        session["prev_response_id"] = _pipe(stream, q)

    threading.Thread(target=work, daemon=True).start()
    return Response(
        sse(q),
        mimetype="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )
    
# ─────────────────── 7. CSV / CRUD ROUTES ────────────────────
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
        log.info(
            "CSV import %d invoices, %d new clients in %.3f s",
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
    log.info("Invoice %d deleted", invoice_id)
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
    log.info("Export %s", fname)
    return (
        csv_data,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )


# ─────────────────── 8. RUN ──────────────────────────────────
if __name__ == "__main__":
    print("openai-python version:", openai.__version__)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
