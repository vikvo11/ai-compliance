# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (stream + tool calling, strict mode) — SDK ≥ 1.78

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
import fcc_ecfs
from flask import (
    Flask, render_template, request, redirect, flash,
    jsonify, session, Response, copy_current_request_context,
    stream_with_context,
)
from flask_sqlalchemy import SQLAlchemy

# ───────────────────────── 0. LOGGING ─────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ──────────────────────── 1. OPENAI CLIENT ───────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL          = cfg.get("DEFAULT", "model",            fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY missing")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# ----- ➊ NEW: propagate FCC key to environment ------------------------
FCC_KEY = cfg.get("DEFAULT", "FCC_API_KEY", fallback=os.getenv("FCC_API_KEY"))
if FCC_KEY:
    os.environ["FCC_API_KEY"] = FCC_KEY
# ----------------------------------------------------------------------

# ──────────────────────── 2. FLASK & DATABASE ────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:////app/data/data.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    # SESSION_COOKIE_HTTPONLY=True,
    # SESSION_COOKIE_SECURE=True, 
    SESSION_COOKIE_SAMESITE="None",
)
db = SQLAlchemy(app)
os.makedirs("/app/data", exist_ok=True)


class Client(db.Model):
    id    = db.Column(db.Integer, primary_key=True)
    name  = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128))
    invoices = db.relationship(
        "Invoice", backref="client", lazy=True, cascade="all, delete-orphan"
    )


class Invoice(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount     = db.Column(db.Float,      nullable=False)
    date_due   = db.Column(db.String(64), nullable=False)
    status     = db.Column(db.String(32), nullable=False)
    client_id  = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)


with app.app_context():
    db.create_all()

# ─────────────────────── 3. LOCAL TOOL FUNCTIONS ─────────────
HTTP_TIMEOUT = httpx.Timeout(6.0, connect=4.0, read=6.0)


@functools.lru_cache(maxsize=1_024)
def _coords_for(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a given city or None if not found."""
    try:
        r = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        res = r.json().get("results")
    except httpx.HTTPError as exc:
        log.error("Weather geocoding failed: %s", exc)
        return None
    
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for_async(lat: float, lon: float) -> dict:
    """Call current-weather endpoint asynchronously."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
            r = await ac.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": True},
            )
            r.raise_for_status()
        return r.json().get("current_weather", {})
    except httpx.HTTPError as exc:
        log.error("Weather fetch failed: %s", exc)
        return {}


def _run_sync(coro: asyncio.Future) -> Any:
    """Run a coroutine even if an event loop already exists."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Return a temperature string for the given location."""
    coords = _coords_for(location)
    cw = _run_sync(_weather_for_async(*coords)) if coords else None
    if not cw:
        return f"Location '{location}' not found."
    temp = cw["temperature"]
    temp = round(temp * 9 / 5 + 32, 1) if unit == "fahrenheit" else temp
    return f"The temperature in {location} is {temp} {'°F' if unit == 'fahrenheit' else '°C'}."


def get_invoice_by_id(invoice_id: str) -> dict:
    """Return invoice data or an error dict."""
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    return (
        {"error": f"Invoice {invoice_id} not found"}
        if not inv
        else {
            "invoice_id":  inv.invoice_id,
            "amount":      inv.amount,
            "date_due":    inv.date_due,
            "status":      inv.status,
            "client_name": inv.client.name,
            "client_email": inv.client.email,
        }
    )

# ──────────── FCC wrappers (use fcc_ecfs module) ────────────
def fcc_search_filings(company: str) -> list[dict]:
    """Return numbered list of PDFs for company."""
    return fcc_ecfs.search(company)


def fcc_get_filings_text(company: str, indexes: list[int]) -> dict:
    """Download & parse selected FCC PDFs (1-based indexes)."""
    return fcc_ecfs.get_texts(company, indexes)

# ───────────────────────── 3a. TOOL SCHEMA ───────────────────
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
                "invoice_id": {"type": "string", "description": "Invoice identifier"},
            },
            "required": ["invoice_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fcc_search_filings",
        "description": "Search FCC ECFS for all PDF attachments of a company",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "Company name to search for",
                },
            },
            "required": ["company"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
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
            "additionalProperties": False
        },
        "strict": True,
    },
]

DISPATCH = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id":   get_invoice_by_id,
    "fcc_search_filings":  fcc_search_filings,
    "fcc_get_filings_text": fcc_get_filings_text,
}

# ─────────────────────── 4. STREAM HANDLER ───────────────────
def _run_tool(name: str, args: str) -> str:
    """Invoke a tool and always return a string (JSON if dict)."""
    try:
        params = json.loads(args or "{}")
    except json.JSONDecodeError:
        return f"Bad JSON: {args}"
    fn = DISPATCH.get(name)
    res = fn(**params) if fn else f"Unknown tool {name}"
    return json.dumps(res) if isinstance(res, dict) else str(res)


def _finish_tool(
    response_id: str,
    run_id: str | None,
    tool_call_id: str,
    name: str,
    args_json: str,
    q: queue.Queue[Any],
) -> str:
    """Send function_call_output message after executing local tool."""
    output = _run_tool(name, args_json)

    follow = client.responses.create(
        model=MODEL,
        input=[{
            "type":   "function_call_output",
            "call_id": tool_call_id,
            "output":  output,
        }],
        previous_response_id=response_id,
        tools=TOOLS,
        stream=True,
    )
    return _pipe(follow, q, run_id)


def _pipe(
    stream,
    q: queue.Queue[Any],
    run_id: str | None = None,
) -> str:
    """
    Parse streaming events, forward assistant text to queue (as SSE `data:`),
    execute tool-calls, and, ОNE TIME, push a special *meta* event that carries
    the freshly issued `previous_response_id`.  Cookie-free clients can store
    that ID (e.g. localStorage) and send it in the next /chat/stream request.
    """
    response_id  = ""
    meta_sent    = False                # guard for single meta push
    early_text: list[str] = []          # buffer tokens until meta is sent
    buf: dict[str, dict[str, Any]] = {} # temp storage for tool-call args

    for ev in stream:
        typ   = getattr(ev, "event", None) or getattr(ev, "type", "") or ""
        delta = getattr(ev, "delta", None)
        run_id = getattr(ev, "run_id", run_id)

        # ───────────────── first fragment with whole Response ─────────────
        if getattr(ev, "response", None):
            response_id = ev.response.id

            # send meta only once and before any user-visible text
            if not meta_sent:
                q.put({"meta": {"prev_id": response_id}})
                for t in early_text:         # flush buffered tokens
                    q.put(t)
                early_text.clear()
                meta_sent = True

        # ───────────────── plain-text token (assistant stream) ────────────
        if isinstance(delta, str) and "arguments" not in typ:
            (q.put if meta_sent else early_text.append)(delta)

        # ───────────────── legacy tool_calls array ────────────────────────
        if getattr(ev, "tool_calls", None):
            for tc in ev.tool_calls:
                fn_name   = tc.function.name if getattr(tc, "function", None) else tc.name
                full_args = tc.function.arguments if getattr(tc, "function", None) else tc.arguments
                call_id   = getattr(tc, "call_id", None) or tc.id
                buf[tc.id] = {"name": fn_name, "parts": [], "call_id": call_id}
                if full_args:
                    response_id = _finish_tool(
                        response_id, run_id,
                        call_id, fn_name, full_args, q
                    )

        # ───────────────── new output_item schema ─────────────────────────
        if typ == "response.output_item.added":
            item = (
                getattr(ev, "item", None)
                or getattr(ev, "output_item", None)
                or ev.model_dump(exclude_none=True).get("item")
            )
            item = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
            if item and item.get("type") in ("function_call", "tool_call"):
                iid   = item["id"]
                fname = item.get("function", {}).get("name") \
                        or item.get("tool_call", {}).get("name") \
                        or item.get("name")
                call_id = item.get("call_id") or iid
                buf[iid] = {"name": fname, "parts": [], "call_id": call_id}

        # ───────────────── accumulate argument chunks ─────────────────────
        if "arguments.delta" in typ:
            iid = (
                getattr(ev, "output_item_id", None)
                or getattr(ev, "item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                buf[iid]["parts"].append(delta or "")

        # ───────────────── end-of-args → run tool ─────────────────────────
        if typ.endswith("arguments.done"):
            iid = (
                getattr(ev, "output_item_id", None)
                or getattr(ev, "item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                full_json = "".join(buf[iid]["parts"])
                fn_name   = buf[iid]["name"]
                call_id   = buf[iid]["call_id"]

                response_id = _finish_tool(
                    response_id, run_id,
                    call_id, fn_name, full_json, q
                )
                buf.pop(iid, None)

        # ───────────────── turn completed ─────────────────────────────────
        if typ in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)             # sentinel for SSE generator
            return response_id

    q.put(None)
    return response_id

# ─────────────────── 5. SSE GENERATOR ────────────────────────
def sse(q: queue.Queue[Any]) -> Generator[bytes, None, None]:
    """Convert queue events to SSE, record prev_response_id before streaming."""
    keep_alive = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)

            if isinstance(tok, dict) and "meta" in tok:
                yield b"event: meta\ndata: " + json.dumps(tok["meta"]).encode() + b"\n\n"
                continue

            if isinstance(tok, dict) and "resp_id" in tok:
                session["prev_response_id"] = tok["resp_id"]
                session.modified = True
                continue

            if tok is None:
                yield b"event: done\ndata: [DONE]\n\n"
                break

            yield f"data: {tok}\n\n".encode()
            keep_alive = time.time() + 20

        except queue.Empty:
            if time.time() > keep_alive:
                yield b": ping\n\n"
                keep_alive = time.time() + 20

# ─────────────────────── 6. CHAT ENDPOINT ────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    POST JSON  {
        "message":               "<user text>",
        "previous_response_id":  "<id from last meta event, optional>"
    }
    ──────────────────────────────────────────────────────────────
    Returns an SSE stream with:
      • assistant tokens         →  event: <default>   data: …
      • meta containing new ID   →  event: meta        data: {"prev_id": "..."}
      • completion sentinel      →  event: done        data: [DONE]
    The browser should save the prev_id from the *meta* event (e.g. localStorage)
    and include it in the next POST.  Куки больше не требуются.
    """
    data = request.get_json(force=True, silent=True) or {}
    msg  = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    last_resp_id = data.get("previous_response_id")
    log.info("prev_response_id from client: %s", last_resp_id)

    q: queue.Queue[Any] = queue.Queue()

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
        _pipe(stream, q)

    threading.Thread(target=work, daemon=True).start()
    return Response(
        stream_with_context(sse(q)),
        mimetype="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )

# ─────────────────── 7. CSV IMPORT / CRUD ROUTES ─────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV.")
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
        log.info("CSV: %d invoices, %d new clients (%.3fs)",
                 len(invoices), len(new_clients), time.perf_counter() - t0)
        flash("Invoices uploaded successfully.")
        return redirect("/")

    return render_template("index.html", invoices=Invoice.query.all())


@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount       = request.form["amount"]
    inv.date_due     = request.form["date_due"]
    inv.status       = request.form["status"]
    inv.client.email = request.form["client_email"]
    db.session.commit()
    return jsonify(
        {"client_name": inv.client.name,
         "invoice_id":  inv.invoice_id,
         "amount":      inv.amount,
         "date_due":    inv.date_due,
         "status":      inv.status}
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
    rows = [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
    csv_data = pd.DataFrame(
        [{
            "client_name":  r.client.name,
            "client_email": r.client.email,
            "invoice_id":   r.invoice_id,
            "amount":       r.amount,
            "date_due":     r.date_due,
            "status":       r.status,
        } for r in rows]
    ).to_csv(index=False)

    fname = f"invoice_{invoice_id or 'all'}.csv"
    return (
        csv_data,
        200,
        {"Content-Type": "text/csv",
         "Content-Disposition": f'attachment; filename="{fname}"'},
    )

# ───────────────────────── 8. RUN APP ────────────────────────
if __name__ == "__main__":
    print("openai-python version:", openai.__version__)
    if tuple(map(int, openai.__version__.split(".")[:2])) < (1, 7):
        sys.exit("openai-python ≥ 1.7.0 required for function calling.")
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
