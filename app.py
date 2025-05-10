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
MODEL = cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY missing")
    sys.exit(1)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=30,
    max_retries=3,
)

# ─────────────────── 2. FLASK & DB ───────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:////app/data/data.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
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
    """Return (lat, lon) for a given city or None if not found."""
    r = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = r.json().get("results")
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict:
    """Async request to current-weather endpoint."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        r = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    return r.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Return temperature for a location, converting units if needed."""
    coords = _coords_for(location)
    cw = asyncio.run(_weather_for(*coords)) if coords else None
    if not cw:
        return f"Location '{location}' not found."
    temp = cw["temperature"]
    temp = round(temp * 9 / 5 + 32, 1) if unit == "fahrenheit" else temp
    return f"The temperature in {location} is {temp} {'°F' if unit == 'fahrenheit' else '°C'}."


def get_invoice_by_id(invoice_id: str) -> dict:
    """Return invoice data for a given invoice number."""
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

TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_invoice_by_id",
        "description": "Return invoice data for a given invoice number",
        "parameters": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
        },
    },
]

DISPATCH: dict[str, Any] = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id": get_invoice_by_id,
}

# ─────────────────── 4. STREAM HANDLER ───────────────────────
def _run_tool(name: str, args: str) -> str:
    """Execute a tool locally and always return str (JSON-encoded when needed)."""
    try:
        params = json.loads(args or "{}")
    except json.JSONDecodeError:
        return f"Bad JSON: {args}"
    fn = DISPATCH.get(name)
    res = fn(**params) if fn else f"Unknown tool {name}"
    return json.dumps(res) if isinstance(res, dict) else str(res)


def _finish_tool(
    response_id: str,
    call_id: str,
    name: str,
    args_json: str,
    q: queue.Queue[str | None],
) -> str:
    """Run the tool and send its output back with a fresh responses.create."""
    log.info("RUN tool=%s id=%s args=%s", name, call_id, args_json)

    # 1) Execute the local Python function.
    output = _run_tool(name, args_json)

    # 2) Send the result to OpenAI as a new input item.
    follow = client.responses.create(
        model=MODEL,
        previous_response_id=response_id,
        input=[
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        ],
        stream=True,
    )

    # 3) Continue piping the assistant’s follow-up.
    return _pipe(follow, q)


def _pipe(  # noqa: C901
    stream,
    q: queue.Queue[str | None],
) -> str:
    """
    Pump a Responses streaming generator → SSE queue.
    Handles incremental function-call arguments and plaintext tokens.
    """
    response_id = ""
    buf: dict[str, dict[str, Any]] = {}  # call_id -> {"name": str, "parts": list[str]}

    for ev in stream:
        et = ev.type

        # Save response_id for subsequent requests.
        if hasattr(ev, "response") and ev.response:
            response_id = ev.response.id

        # ───── Plain text tokens ─────────────────────────────
        if et == "output_text.delta":
            q.put(ev.delta)
            continue

        # ───── Function-call handling ────────────────────────
        if et == "function_call":
            # First chunk contains metadata.
            buf[ev.id] = {"name": ev.name, "parts": []}
            if ev.arguments is not None:  # non-streaming arguments
                response_id = _finish_tool(
                    response_id,
                    ev.id,
                    ev.name,
                    ev.arguments,
                    q,
                )
                buf.pop(ev.id, None)
            continue

        if et == "function_call.arguments_delta":
            # Collect argument chunks.
            if ev.id in buf:
                buf[ev.id]["parts"].append(ev.delta or "")
            continue

        if et == "function_call.arguments_done":
            # Arguments are complete → run the tool.
            if ev.id in buf:
                full_json = "".join(buf[ev.id]["parts"])
                response_id = _finish_tool(
                    response_id,
                    ev.id,
                    buf[ev.id]["name"],
                    full_json,
                    q,
                )
                buf.pop(ev.id, None)
            continue

        # ───── Final marker ─────────────────────────────────
        if et == "end":
            q.put(None)
            return response_id

    # Stream ended without explicit "end".
    q.put(None)
    return response_id


# ─────────────────── 5. SSE GENERATOR ────────────────────────
def sse(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    """Server-Sent Events wrapper, forwards tokens & keeps connection alive."""
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
    """
    POST JSON {"message": "..."} →
    SSE stream with the assistant reply (tool calling is automatic).
    """
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    last_resp_id = session.get("prev_response_id")
    q: queue.Queue[str | None] = queue.Queue()

    @copy_current_request_context
    def work() -> None:
        stream = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": msg}],
            previous_response_id=last_resp_id,
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

# ─────────────────── 7. CSV / CRUD (unchanged) ───────────────
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
            "CSV import %d invoices, %d new clients %.3f s",
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
    log.info("export %s", fname)
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
    major, minor, *_ = map(int, openai.__version__.split("."))
    if (major, minor) < (1, 7):
        sys.exit("openai-python ≥ 1.7.0 required")
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
