# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (stream + tool-calling, strict mode)
# Added: OpenAI Agents (MCP) integration
# All comments are in English (per user preference).

from __future__ import annotations

import os
import sys
import re
import time
import json
import queue
import threading
import asyncio
import functools
import logging
from collections.abc import Generator
from typing import Any

# ────────────────── AGENTS: additional imports ───────────────
try:
    from agents import Agent, Runner
    from agents.mcp import MCPServerStreamableHttp
    from agents.model_settings import ModelSettings
except ImportError:
    raise RuntimeError(
        "openai-agents not installed. Install it with:\n"
        "   pip install openai-agents>=0.3.0"
    )
# ─────────────────────────────────────────────────────────────

from cors import register_cors

import httpx
import pandas as pd
import configparser
import openai                          # ← single top-level import is enough
from openai import OpenAI

# ---------- universal alias for “invalid request” exception ----------
try:                                   # openai-python 0.* branch
    from openai.error import InvalidRequestError as _InvalidRequest
except ImportError:                    # openai-python ≥1.0
    from openai import BadRequestError as _InvalidRequest
# ---------------------------------------------------------------------

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
MODEL          = cfg.get("DEFAULT", "model",         fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY missing")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# ──────────── AGENTS: configuration variables ───────────────
AGENT_MODEL     = cfg.get("DEFAULT", "agent_model", fallback=os.getenv("OPENAI_AGENT_MODEL", MODEL))
AGENT_MAX_TURNS = int(cfg.get("DEFAULT", "agent_max_turns", fallback=os.getenv("AGENT_MAX_TURNS", "6")))
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
# ─────────────────────────────────────────────────────────────

MCP_SERVER_URL = cfg.get("DEFAULT", "MCP_SERVER_URL", fallback=os.getenv("MCP_SERVER_URL"))

# ----- propagate FCC key to environment ----------------------
FCC_KEY = cfg.get("DEFAULT", "FCC_API_KEY", fallback=os.getenv("FCC_API_KEY"))
if FCC_KEY:
    os.environ["FCC_API_KEY"] = FCC_KEY
# -------------------------------------------------------------
import fcc_ecfs  # noqa: E402  (depends on env var)

# ──────────────────── instructions for Responses API ─────────
try:
    with open("responcess_api_instructions.cfg", "r", encoding="utf-8") as f:
        INSTRUCTIONS = f.read().strip()
except OSError as exc:
    log.warning("Failed to load instructions.cfg: %s", exc)
    INSTRUCTIONS = ""

# ──────────────────────── 2. FLASK & DATABASE ────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:////app/data/data.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    SESSION_COOKIE_SAMESITE="None",
)
# db = SQLAlchemy(app)  # enable if models are defined
# os.makedirs("/app/data", exist_ok=True)

register_cors(app)

# class Client(db.Model): ...
# class Invoice(db.Model): ...

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
            "invoice_id":   inv.invoice_id,
            "amount":       inv.amount,
            "date_due":     inv.date_due,
            "status":       inv.status,
            "client_name":  inv.client.name,
            "client_email": inv.client.email,
        }
    )

# ───────────── FCC wrappers (use fcc_ecfs module) ────────────
def fcc_search_filings(company: str) -> list[dict]:
    """Return numbered list of PDFs for company."""
    return fcc_ecfs.search(company)

def fcc_get_filings_text(company: str, indexes: list[int]) -> dict:
    """Download and parse the selected FCC PDFs without saving them."""
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
                    "description": "Temperature unit (default: celsius).",
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
        "name": "ask_mcp_agent",
        "description": "Delegate the user's request to an external MCP-powered agent.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Full user question"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

DISPATCH = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id":   get_invoice_by_id,
    "ask_mcp_agent":       lambda query: _run_agent(query),
}

@app.before_request
def _log():
    print("MCP <--", request.method, request.path)

# ──────────── AGENTS: helper wrappers ────────────────────────
async def _run_agent_async(user_text: str, previous_response_id: str | None = None) -> str:
    """Run the MCP agent asynchronously and return its final text output."""
    async with MCPServerStreamableHttp(
        name="External MCP Server",
        params={"url": MCP_SERVER_URL.rstrip('/') + '/'},
        client_session_timeout_seconds=25,
    ) as mcp_server:
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant. Use the MCP tools when appropriate.",
            mcp_servers=[mcp_server],
            model=AGENT_MODEL, #"gpt-4o-mini",
            model_settings=ModelSettings(tool_choice="required"),
        )
        result = await Runner.run(
            starting_agent=agent,
            input=user_text,
            previous_response_id=previous_response_id,
            max_turns=AGENT_MAX_TURNS,
        )
        return result.final_output

def _run_agent(user_text: str, previous_response_id: str | None = None) -> str:
    """Synchronous wrapper around the async agent runner."""
    return _run_sync(_run_agent_async(user_text, previous_response_id))

# ─────────────────── 4. STREAM & TOOL HANDLING ───────────────
def _run_tool(name: str, args: str) -> str:
    """Execute a local tool and return its result as a string."""
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
    """Send function_call_output after executing a tool; continue piping."""
    try:
        output = _run_tool(name, args_json)
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Tool '%s' failed: %s", name, exc)
        output = f"ERROR: {exc}"

    try:
        follow = client.responses.create(
            model=MODEL,
            input=[{
                "type":   "function_call_output",
                "call_id": tool_call_id,
                "output":  output,
            }],
            previous_response_id=response_id,
            instructions=INSTRUCTIONS,
            stream=True,
        )
        return _pipe(follow, q, run_id)
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Failed to send tool output: %s", exc)
        q.put(str(exc))
        return response_id

def _pipe(
    stream,
    q: queue.Queue[Any],
    run_id: str | None = None,
) -> str:
    """Forward an OpenAI streaming response to the SSE queue, handling tools."""
    response_id  = ""
    meta_sent    = False
    early_text: list[str] = []
    buf: dict[str, dict[str, Any]] = {}

    for ev in stream:
        typ   = getattr(ev, "event", None) or getattr(ev, "type", "") or ""
        delta = getattr(ev, "delta", None)
        run_id = getattr(ev, "run_id", run_id)

        if getattr(ev, "response", None):  # first fragment
            response_id = ev.response.id
            if not meta_sent:
                q.put({"meta": {"prev_id": response_id}})
                for t in early_text:
                    q.put(t)
                early_text.clear()
                meta_sent = True

        if isinstance(delta, str) and "arguments" not in typ:  # plain text token
            (q.put if meta_sent else early_text.append)(delta)

        # legacy tool_calls
        if getattr(ev, "tool_calls", None):
            for tc in ev.tool_calls:
                fn_name   = tc.function.name if getattr(tc, "function", None) else tc.name
                full_args = tc.function.arguments if getattr(tc, "function", None) else tc.arguments
                call_id   = getattr(tc, "call_id", None) or tc.id
                buf[tc.id] = {"name": fn_name, "parts": [], "call_id": call_id}
                if full_args:
                    response_id = _finish_tool(
                        response_id, run_id, call_id, fn_name, full_args, q
                    )

        # new schema
        if typ == "response.output_item.added":
            item = (
                getattr(ev, "item", None)
                or getattr(ev, "output_item", None)
                or ev.model_dump(exclude_none=True).get("item")
            )
            item = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
            if item and item.get("type") in ("function_call", "tool_call"):
                iid     = item["id"]
                fname   = (item.get("function", {}) or item.get("tool_call", {})).get("name") or item.get("name")
                call_id = item.get("call_id") or iid
                full_args = (item.get("function", {}) or item.get("tool_call", {})).get("arguments")
                if full_args:
                    response_id = _finish_tool(
                        response_id, run_id, call_id, fname, full_args, q
                    )
                else:
                    buf[iid] = {"name": fname, "parts": [], "call_id": call_id}

        # accumulating chunks
        if "arguments.delta" in typ:
            iid = getattr(ev, "output_item_id", None) or getattr(ev, "item_id", None) or getattr(ev, "tool_call_id", None)
            if iid and iid in buf:
                buf[iid]["parts"].append(delta or "")

        # end of args
        if typ.endswith("arguments.done"):
            iid = getattr(ev, "output_item_id", None) or getattr(ev, "item_id", None) or getattr(ev, "tool_call_id", None)
            if iid and iid in buf:
                full_json = "".join(buf[iid]["parts"])
                response_id = _finish_tool(
                    response_id, run_id,
                    buf[iid]["call_id"], buf[iid]["name"], full_json, q
                )
                buf.pop(iid, None)

        if typ in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)
            return response_id

    q.put(None)
    return response_id

# ─────────────────── 5. SSE GENERATOR ────────────────────────
def sse(q: queue.Queue[Any]) -> Generator[bytes, None, None]:
    """Convert queue events to SSE bytes."""
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

# ─────── 6. HELPER THAT RESCUES BROKEN TOOL-CALL CHAINS ──────
def _create_stream_with_rescue(msg: str, last_resp_id: str | None):
    """
    Call OpenAI Responses.create; if API complains about a missing
    function_call_output, automatically send a stub output and retry.
    """
    try:  # first attempt
        return client.responses.create(
            model=MODEL,
            input=msg,
            previous_response_id=last_resp_id,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=True,
        )

    except _InvalidRequest as exc:
        if "No tool output found" not in str(exc) or not last_resp_id:
            raise  # not our case

        # extract missing call_id
        m = re.search(r"function call ([\w-]+)\.", str(exc))
        missing_call_id = m.group(1) if m else None
        if not missing_call_id:
            raise

        log.warning("Rescuing broken branch: sending stub output for call_id=%s", missing_call_id)

        # send stub output
        stub_resp = client.responses.create(
            model=MODEL,
            input=[{
                "type":   "function_call_output",
                "call_id": missing_call_id,
                "output":  "ERROR: tool interrupted (auto-generated stub)",
            }],
            previous_response_id=last_resp_id,
            instructions=INSTRUCTIONS,
        )

        # retry original request
        return client.responses.create(
            model=MODEL,
            input=msg,
            previous_response_id=stub_resp.id,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=True,
        )

# ─────────────────────── 7. CHAT ENDPOINT ────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """SSE endpoint that streams assistant tokens + meta + done."""
    data = request.get_json(force=True, silent=True) or {}
    msg  = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    last_resp_id = data.get("previous_response_id")
    log.info("prev_response_id from client: %s", last_resp_id)

    q: queue.Queue[Any] = queue.Queue()

    @copy_current_request_context
    def work() -> None:
        try:
            stream = _create_stream_with_rescue(msg, last_resp_id)
            _pipe(stream, q)
        except Exception as exc:  # pylint: disable=broad-except
            log.error("OpenAI stream failed: %s", exc)
            q.put(str(exc))
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    return Response(
        stream_with_context(sse(q)),
        mimetype="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )

# ───────────────── 8. CSV IMPORT / CRUD ROUTES ───────────────
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
        log.info(
            "CSV: %d invoices, %d new clients (%.3fs)",
            len(invoices), len(new_clients), time.perf_counter() - t0,
        )
        flash("Invoices uploaded successfully.")
        return redirect("/")

    # return render_template("index.html", invoices=Invoice.query.all())
    return render_template("index.html", invoices='')

@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount       = request.form["amount"]
    inv.date_due     = request.form["date_due"]
    inv.status       = request.form["status"]
    inv.client.email = request.form["client_email"]
    db.session.commit()
    return jsonify(
        {
            "client_name": inv.client.name,
            "invoice_id":  inv.invoice_id,
            "amount":      inv.amount,
            "date_due":    inv.date_due,
            "status":      inv.status,
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
        {
            "Content-Type":        "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )

# ───────────────────────── 9. RUN APP ────────────────────────
if __name__ == "__main__":
    print("openai-python version:", openai.__version__)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
