# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (stream + tool calling) — SDK ≥ 1.78

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

# ───────────────────── 0. LOGGING ──────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ───────────────────── 1. OPENAI ───────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL           = cfg.get("DEFAULT", "model",          fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY missing — supply env var or cfg/openai.cfg")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# ───────────────────── 2. FLASK & DB ───────────────────────────
# SQLite file lives in ./data so the container’s writable layer stays small
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:///./data/data.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
os.makedirs("./data", exist_ok=True)
db = SQLAlchemy(app)


class Client(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(128), unique=True, nullable=False)
    email       = db.Column(db.String(128))
    invoices    = db.relationship(
        "Invoice", backref="client", lazy=True, cascade="all, delete-orphan"
    )


class Invoice(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    invoice_id  = db.Column(db.String(64), nullable=False)
    amount      = db.Column(db.Float, nullable=False)
    date_due    = db.Column(db.String(64), nullable=False)
    status      = db.Column(db.String(32), nullable=False)
    client_id   = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)


with app.app_context():
    db.create_all()

# ───────────────────── 3. TOOL FUNCTIONS ───────────────────────
# Short HTTP timeouts so the event loop can cancel quickly if the service hangs
HTTP_TIMEOUT = httpx.Timeout(6.0, connect=4.0, read=6.0)

@functools.lru_cache(maxsize=1024)
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (latitude, longitude) for a given city or None if not found."""
    r = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = r.json().get("results")
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict:
    """Async request to open-meteo current‐weather endpoint."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        r = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    return r.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Return a human‐readable temperature string for the location."""
    coords = _coords_for(location)
    if not coords:
        return f"Location '{location}' not found."
    cw = asyncio.run(_weather_for(*coords))
    if not cw:
        return "Weather service unavailable."
    temp = cw["temperature"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
    return f"The temperature in {location} is {temp} {'°F' if unit == 'fahrenheit' else '°C'}."


def get_invoice_by_id(invoice_id: str) -> dict:
    """Return invoice data as dict or {'error': ...}."""
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    return (
        {"error": f"Invoice {invoice_id} not found"}
        if not inv
        else {
            "invoice_id"  : inv.invoice_id,
            "amount"      : inv.amount,
            "date_due"    : inv.date_due,
            "status"      : inv.status,
            "client_name" : inv.client.name,
            "client_email": inv.client.email,
        }
    )

# OpenAI SDK ≥1.7 uses this flat schema with type="function" at the top level
TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "celsius"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_invoice_by_id",
        "description": "Return invoice details for an invoice number.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {
                    "type": "string",
                    "description": "Invoice reference code"
                }
            },
            "required": ["invoice_id"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# Map tool name → local Python callable
DISPATCH: dict[str, callable] = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id"  : get_invoice_by_id,
}

# ───────────────────── 4. STREAM HANDLER ───────────────────────
def _run_tool(name: str, args_json: str) -> str:
    """Execute the tool and always return a *string* (serialize dicts)."""
    try:
        params = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return f"Bad JSON: {args_json}"

    fn = DISPATCH.get(name)
    if not fn:
        return f"Unknown tool '{name}'"

    result = fn(**params)
    return json.dumps(result) if isinstance(result, (dict, list)) else str(result)


def _finish_tool(
    response_id: str,
    thread_id: str | None,
    run_id: str | None,
    tool_call_id: str,
    name: str,
    args_json: str,
    q: queue.Queue[str | None],
) -> str:
    """
    1) Execute the local Python function
    2) POST the result back with Responses.submit_tool_outputs(stream=True)
    3) Pipe the follow-up stream through _pipe
    """
    log.info("TOOL start: %s(%s)", name, args_json)
    output = _run_tool(name, args_json)

    follow = client.responses.submit_tool_outputs(
        response_id=response_id,
        tool_outputs=[{"tool_call_id": tool_call_id, "output": output}],
        stream=True,
    )
    return _pipe(follow, q, thread_id, run_id)


def _pipe(  # noqa: C901 — long but linear
    stream,
    q: queue.Queue[str | None],
    thread_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """
    Consume OpenAI streaming events (text + tool calling),
    push tokens to the SSE queue, and run tools on-the-fly.
    """
    response_id = ""
    # Buffer partial argument chunks per tool_call_id
    buf: dict[str, dict[str, Any]] = {}

    for ev in stream:
        ev_type: str = getattr(ev, "type", "") or getattr(ev, "event", "")
        delta     = getattr(ev, "delta", None)

        # Capture IDs once
        if getattr(ev, "response", None):
            response_id = ev.response.id
            thread_id   = getattr(ev.response, "thread_id", thread_id)
            run_id      = getattr(ev.response, "run_id", run_id)

        # Plain text token?
        if isinstance(delta, str) and "arguments" not in ev_type:
            q.put(delta)

        # ─── Variant 1 (old) :  ev.tool_calls (list) ──────────
        if getattr(ev, "tool_calls", None):
            for tc in ev.tool_calls:
                fn_name = (
                    getattr(tc, "function", None).name
                    if getattr(tc, "function", None)
                    else tc.name
                )
                full_args = (
                    getattr(tc, "function", None).arguments
                    if getattr(tc, "function", None)
                    else tc.arguments
                )
                buf[tc.id] = {"name": fn_name, "parts": []}
                if full_args:  # sometimes model emits full JSON at once
                    response_id = _finish_tool(
                        response_id, thread_id, run_id,
                        tc.id, fn_name, full_args, q
                    )

        # ─── Variant 2 (new) :  response.output_item.added ────
        if ev_type == "response.output_item.added":
            item = (
                getattr(ev, "item", None)
                or getattr(ev, "output_item", None)
                or ev.model_dump(exclude_none=True).get("item")
            )
            if not item:
                continue
            if item.get("type") not in ("function_call", "tool_call"):
                continue
            iid  = item["id"]
            name = (
                item.get("function", {}).get("name")
                or item.get("tool_call", {}).get("name")
                or item.get("name")
            )
            buf[iid] = {"name": name, "parts": []}

        # Accumulate streamed argument deltas
        if "arguments.delta" in ev_type:
            iid = (
                getattr(ev, "item_id", None)
                or getattr(ev, "output_item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                buf[iid]["parts"].append(delta or "")

        # Arguments finished — run the tool
        if ev_type.endswith("arguments.done"):
            iid = (
                getattr(ev, "item_id", None)
                or getattr(ev, "output_item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                args_json = "".join(buf[iid]["parts"])
                response_id = _finish_tool(
                    response_id, thread_id, run_id,
                    iid, buf[iid]["name"], args_json, q
                )
                buf.pop(iid, None)

        # End of stream markers
        if ev_type in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)
            return response_id

    # Fallback: stream ended unexpectedly
    q.put(None)
    return response_id


# ───────────────────── 5. SSE GENERATOR ───────────────────────
def sse(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    """
    Convert queue tokens to Server-Sent Events and emit heartbeat pings.
    """
    keep_alive = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)
            if tok is None:           # sentinel
                yield b"event: done\ndata: [DONE]\n\n"
                break
            yield f"data: {tok}\n\n".encode()
            keep_alive = time.time() + 20
        except queue.Empty:
            if time.time() > keep_alive:
                yield b": ping\n\n"
                keep_alive = time.time() + 20


# ───────────────────── 6. CHAT ENDPOINT ───────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """POST JSON {"message": "..."} → SSE stream with assistant reply."""
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    last_response_id = session.get("prev_response_id")
    q: queue.Queue[str | None] = queue.Queue()

    @copy_current_request_context
    def work() -> None:
        stream = client.responses.create(
            model=MODEL,
            input=msg,
            previous_response_id=last_response_id,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )
        session["prev_response_id"] = _pipe(stream, q)

    threading.Thread(target=work, daemon=True).start()
    return Response(
        sse(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ───────────────────── 7. CSV / CRUD ROUTES ───────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        df = pd.read_csv(f)
        t0 = time.perf_counter()
        new_clients: dict[str, Client] = {}
        new_invoices: list[Invoice] = []

        for _, row in df.iterrows():
            name = row["client_name"]
            client_obj = (
                Client.query.filter_by(name=name).first()
                or new_clients.get(name)
            )
            if not client_obj:
                client_obj = Client(name=name, email=row.get("client_email", ""))
                new_clients[name] = client_obj
            new_invoices.append(
                Invoice(
                    invoice_id=row["invoice_id"],
                    amount=row["amount"],
                    date_due=row["date_due"],
                    status=row["status"],
                    client=client_obj,
                )
            )

        db.session.add_all(new_clients.values())
        db.session.add_all(new_invoices)
        db.session.commit()
        log.info(
            "CSV import: %d invoices, %d new clients in %.2f s",
            len(new_invoices), len(new_clients), time.perf_counter() - t0
        )
        flash("Invoices uploaded successfully.")
        return redirect("/")

    return render_template("index.html", invoices=Invoice.query.all())


@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount        = request.form["amount"]
    inv.date_due      = request.form["date_due"]
    inv.status        = request.form["status"]
    inv.client.email  = request.form["client_email"]
    db.session.commit()
    return jsonify(
        {
            "client_name": inv.client.name,
            "invoice_id" : inv.invoice_id,
            "amount"     : inv.amount,
            "date_due"   : inv.date_due,
            "status"     : inv.status,
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
        if invoice_id else Invoice.query.all()
    )
    csv_data = pd.DataFrame(
        [
            {
                "client_name" : r.client.name,
                "client_email": r.client.email,
                "invoice_id"  : r.invoice_id,
                "amount"      : r.amount,
                "date_due"    : r.date_due,
                "status"      : r.status,
            }
            for r in rows
        ]
    ).to_csv(index=False)

    fname = f"invoice_{invoice_id or 'all'}.csv"
    return (
        csv_data,
        200,
        {
            "Content-Type"       : "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )

# ───────────────────── 8. RUN ─────────────────────────────────
if __name__ == "__main__":
    print("openai-python version:", openai.__version__)
    if tuple(map(int, openai.__version__.split(".")[:2])) < (1, 7):
        sys.exit("openai-python ≥ 1.7.0 required for Responses.submit_tool_outputs")
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
