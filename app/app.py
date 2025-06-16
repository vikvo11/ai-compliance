# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (stream + tool-calling, strict mode)
# Multi-agent via HTTP: tools are discovered automatically from each FastMCP
# server’s /schema.  No nested “LLM-inside-LLM”.
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
from threading import Lock
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

import httpx
import pandas as pd
import configparser
import openai                              # single import is enough
from openai import OpenAI

from flask import (
    Flask, render_template, request, redirect, flash,
    jsonify, session, Response, copy_current_request_context,
    stream_with_context,
)
from flask_sqlalchemy import SQLAlchemy

from cors import register_cors

# ───────────────────────── 0. LOGGING ─────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ---------- universal alias for “invalid request” exception ----------
try:                                   # openai-python 0.* branch
    from openai.error import InvalidRequestError as _InvalidRequest
except ImportError:                    # openai-python ≥1.0
    from openai import BadRequestError as _InvalidRequest
# ---------------------------------------------------------------------

# ──────────────────────── 1. OPENAI CLIENT ───────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL  = cfg.get("DEFAULT", "model",         fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY missing")
    sys.exit(1)

# Make the key visible for openai-agents exporter *before* we import tracing
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

# ─────────────── tracing helpers (import *после* установки env) ─────────
try:
    # pip install openai-agents>=0.3.0
    from agents.tracing import trace, generation_span, function_span       # type: ignore
except ImportError:                                                        # pragma: no cover
    log.warning("openai-agents not available – Traces are disabled")

    @contextmanager
    def _noop_cm(*_a, **_kw):
        yield

    trace = generation_span = function_span = _noop_cm                    # type: ignore

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# Helper: wrap any client.responses.create() in generation_span ───────────
def _responses_create_traced(**kwargs):
    """
    Execute client.responses.create() inside a generation_span so that each
    LLM call shows up as a child span in the Trace timeline.
    """
    orig_input = kwargs.get("input")

    # If user passed a raw string, wrap it to keep the schema valid
    if isinstance(orig_input, str):
        orig_input = [{"type": "text", "text": orig_input[:200]}]

    with generation_span(model=kwargs.get("model"), input=orig_input):
        return client.responses.create(**kwargs)

# The model may change at runtime via /config
MODEL: str = DEFAULT_MODEL                               # guarded by CONFIG_LOCK
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

# ─────────────── 1a. GLOBAL CONFIG (MODEL & INSTRUCTIONS) ───────────────
try:
    with open("responcess_api_instructions.cfg", "r", encoding="utf-8") as f:
        INSTRUCTIONS: str = f.read().strip()
except OSError as exc:
    log.warning("Failed to load instructions.cfg: %s", exc)
    INSTRUCTIONS = ""

CONFIG_LOCK = Lock()   # protects MODEL and INSTRUCTIONS

# ─────────────────── 1b. AGENT / MCP REGISTRY ───────────────────
AGENTS: dict[str, str] = {
    "telco": cfg.get("DEFAULT", "MCP_SERVER_URL", fallback=os.getenv("MCP_SERVER_URL", "")).rstrip("/"),
}
if not AGENTS["telco"]:
    log.error("MCP_SERVER_URL missing — Telco tools will be unavailable")

HTTP_TIMEOUT = httpx.Timeout(6.0, connect=4.0, read=6.0)

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

# ───────────────────────── 2a. /config ENDPOINT ─────────────────────────
@app.route("/config", methods=["GET", "POST", "OPTIONS"])
def config_endpoint():
    """
    GET  → return current MODEL and INSTRUCTIONS.
    POST → JSON body { "model": "<model>", "instructions": "<text>" }
    """
    global MODEL, INSTRUCTIONS

    if request.method == "GET":
        with CONFIG_LOCK:
            return jsonify(
                {
                    "model": MODEL,
                    "instructions": INSTRUCTIONS,
                    "available_models": AVAILABLE_MODELS,
                }
            )

    data = request.get_json(force=True, silent=True) or {}
    new_model        = data.get("model")
    new_instructions = data.get("instructions")

    with CONFIG_LOCK:
        if new_model:
            if new_model not in AVAILABLE_MODELS:
                return jsonify({"error": f"Model '{new_model}' not allowed"}), 400
            MODEL = new_model
            log.info("MODEL changed at runtime to %s", MODEL)

        if isinstance(new_instructions, str):
            INSTRUCTIONS = new_instructions
            log.info("INSTRUCTIONS updated (length=%d chars)", len(INSTRUCTIONS))

        return jsonify({"ok": True, "model": MODEL, "instructions": INSTRUCTIONS})

# ─────────────────────── 3. LOCAL TOOL FUNCTIONS ─────────────
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

# ───────────────── 3a. GENERIC MCP FUNCTION ──────────────────
def _run_mcp_generic(agent: str, base_url: str, tool_name: str, **payload) -> dict | str:
    """
    Call FastMCP tool <tool_name> via JSON-RPC and return result or text.
    Wrapped in function_span so latency as well as inputs/outputs are visible.
    """
    root = base_url.rstrip("/")
    url  = (root if root.endswith("/mcp") else root + "/mcp").rstrip("/") + "/"

    rpc_payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 1_000_000,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": payload or {}},
    }

    headers = {
        "Accept":       "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    payload_json = json.dumps(payload, default=str)[:1_000]

    with function_span(name=f"{agent}.{tool_name}", input=payload_json) as _span:   # noqa: F841
        try:
            r = httpx.post(
                url,
                json=rpc_payload,
                headers=headers,
                timeout=httpx.Timeout(20.0, connect=4.0, read=20.0),
                follow_redirects=True,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            out = f"MCP error ({agent}.{tool_name}): {exc}"
            _emit_result_span(f"{agent}.{tool_name}.result", out[:1_000])          # ── FIX
            log.error("MCP %s.%s failed: %s", agent, tool_name, exc)
            return out

    ctype = r.headers.get("content-type", "")
    if "json" in ctype:
        try:
            obj = r.json()
        except ValueError:
            result = r.text
            _emit_result_span(f"{agent}.{tool_name}.result", result[:1_000])       # ── FIX
            return result
        if isinstance(obj, dict):
            if "result" in obj:
                result = obj["result"]
                _emit_result_span(                                              # ── FIX
                    f"{agent}.{tool_name}.result",
                    json.dumps(result, default=str)[:1_000],
                )
                return result
            if "error" in obj:
                result = f"MCP error ({agent}.{tool_name}): {obj['error']}"
                _emit_result_span(f"{agent}.{tool_name}.result", result[:1_000])   # ── FIX
                return result
        _emit_result_span(                                                      # ── FIX
            f"{agent}.{tool_name}.result",
            json.dumps(obj, default=str)[:1_000],
        )
        return obj

    # Non-JSON
    _emit_result_span(f"{agent}.{tool_name}.result", r.text[:1_000])              # ── FIX
    return r.text

# ───────────────── instructions for Responses API ────────────
# (INSTRUCTIONS loaded earlier and mutable via /config)

# ───────────────────────── 4. TOOL SCHEMA ────────────────────
TOOLS: list[dict] = [
    {"type": "web_search"},
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
]

# 4b — DISPATCH MAP
DISPATCH: dict[str, Callable[..., Any]] = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id":   get_invoice_by_id,
}

# 4c — dynamic discovery of agent tools
def _register_agent(agent_name: str, base_url: str) -> None:
    """
    Discover tools from a FastMCP server via JSON-RPC and register them.
    """
    url = f"{base_url.rstrip('/')}/mcp/"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

    try:
        r = httpx.post(
            url,
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
        rpc = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.error("Cannot fetch tools list from %s: %s", url, exc)
        return

    result = rpc.get("result", {})
    tools = result["tools"] if isinstance(result, dict) and "tools" in result else result
    if not isinstance(tools, list):
        log.error("Cannot parse tools list from %s: %s", url, rpc)
        return

    def _strictify(node: dict) -> dict:
        if not isinstance(node, dict):
            return node
        node = {k: _strictify(v) if isinstance(v, dict) else v for k, v in node.items()}

        for key in ("anyOf", "oneOf", "allOf"):
            if key in node and isinstance(node[key], list):
                node[key] = [_strictify(s) for s in node[key]]
        if "items" in node and isinstance(node["items"], dict):
            node["items"] = _strictify(node["items"])

        if node.get("type") == "object":
            node.setdefault("properties", {})
            node["additionalProperties"] = False
            node["properties"] = {p: _strictify(s) for p, s in node["properties"].items()}
            node["required"] = sorted(node["properties"].keys())
        return node

    for raw in tools:
        tname = raw.get("name")
        if not tname or tname in DISPATCH:
            continue

        origin_schema = raw.get("parameters") or raw.get("inputSchema") or {}
        schema = _strictify(origin_schema)
        if schema.get("type") != "object":
            schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "required": [],
            }

        spec = {
            "type":        "function",
            "name":        tname,
            "description": raw.get("description", ""),
            "strict":      True,
            "parameters":  schema,
        }
        TOOLS.append(spec)
        DISPATCH[tname] = functools.partial(
            _run_mcp_generic,
            agent=agent_name,
            base_url=base_url,
            tool_name=tname,
        )
        log.info("Registered tool %s from agent %s", tname, agent_name)

# ───────────────────────── 5. UTILITIES ──────────────────────
@app.before_request
def _log():
    log.debug("HTTP %s %s", request.method, request.path)

# ───── helper: tiny span just for output ────────────
def _emit_result_span(span_name: str, output: str) -> None:              # ── FIX
    """Fire-and-forget span that only carries output."""
    with function_span(name=span_name, output=output):
        pass

def _run_tool(name: str, args: str) -> str:
    """
    Execute a tool and return its result as str.
    Wrapped in function_span with both input and output captured.
    """
    try:
        params = json.loads(args or "{}")
    except json.JSONDecodeError:
        return f"Bad JSON: {args}"

    fn = DISPATCH.get(name)
    if not fn:
        return f"Unknown tool {name}"

    params_json = json.dumps(params, default=str)[:1_000]

    with function_span(name=name, input=params_json) as _span:           # noqa: F841
        res = fn(**params)

    # Log output in a tiny follow-up span (so that timing of main call remains accurate)
    out_json = (
        json.dumps(res, default=str)[:1_000]
        if isinstance(res, (dict, list))
        else str(res)[:1_000]
    )
    _emit_result_span(f"{name}.result", out_json)                        # ── FIX

    return json.dumps(res) if isinstance(res, (dict, list)) else str(res)

# ─────────── 5a. TOOL-FINISH WITH TRACING ────────────
def _finish_tool(
    response_id: str,
    run_id: str | None,
    tool_call_id: str,
    name: str,
    args_json: str,
    q: queue.Queue[Any],
) -> str:
    """
    Execute tool → send function_call_output → continue piping (traced).
    """
    try:
        output = _run_tool(name, args_json)
    except Exception as exc:                                            # pylint: disable=broad-except
        log.error("Tool '%s' failed: %s", name, exc)
        output = f"ERROR: {exc}"

    follow = _responses_create_traced(
        model=MODEL,
        input=[
            {
                "type":   "function_call_output",
                "call_id": tool_call_id,
                "output":  output,
            }
        ],
        previous_response_id=response_id,
        instructions=INSTRUCTIONS,
        stream=True,
    )
    return _pipe(follow, q, run_id)

# ─────────────────── 6. STREAM PIPELINE ──────────────────────
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

        if isinstance(delta, str) and "arguments" not in typ:
            (q.put if meta_sent else early_text.append)(delta)

        # legacy tool_calls (0.* schema)
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

        # new schema (1.*)
        if typ == "response.output_item.added":
            item = (
                getattr(ev, "item", None)
                or getattr(ev, "output_item", None)
                or ev.model_dump(exclude_none=True).get("item")
            )
            item = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
            if item and item.get("type") in ("function_call", "tool_call"):
                iid     = item["id"]
                fname   = (
                    (item.get("function", {}) or item.get("tool_call", {})).get("name")
                    or item.get("name")
                )
                call_id = item.get("call_id") or iid
                full_args = (
                    (item.get("function", {}) or item.get("tool_call", {})).get("arguments")
                )
                if full_args:
                    response_id = _finish_tool(
                        response_id, run_id, call_id, fname, full_args, q
                    )
                else:
                    buf[iid] = {"name": fname, "parts": [], "call_id": call_id}

        # accumulating argument chunks
        if "arguments.delta" in typ:
            iid = (
                getattr(ev, "output_item_id", None)
                or getattr(ev, "item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                buf[iid]["parts"].append(delta or "")

        # end-of-arguments
        if typ.endswith("arguments.done"):
            iid = (
                getattr(ev, "output_item_id", None)
                or getattr(ev, "item_id", None)
                or getattr(ev, "tool_call_id", None)
            )
            if iid and iid in buf:
                full_json = "".join(buf[iid]["parts"])
                response_id = _finish_tool(
                    response_id,
                    run_id,
                    buf[iid]["call_id"],
                    buf[iid]["name"],
                    full_json,
                    q,
                )
                buf.pop(iid, None)

        if typ in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)
            return response_id

    q.put(None)
    return response_id

# ───────────── 7. SSE & RESCUE ─────────────
def sse(q: queue.Queue[Any]) -> Generator[bytes, None, None]:
    """Convert queue events to SSE bytes."""
    keep_alive = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)

            if isinstance(tok, dict) and "meta" in tok:
                yield (
                    b"event: meta\ndata: "
                    + json.dumps(tok["meta"]).encode()
                    + b"\n\n"
                )
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

def _create_stream_with_rescue(msg: str, last_resp_id: str | None):
    """
    Prepare first streaming Responses.create with stub-rescue logic.
    (Без внешнего trace — теперь trace стартует в work().)
    """
    try:
        return _responses_create_traced(
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
            raise
        m = re.search(r"function call ([\w-]+)\.", str(exc))
        missing_call_id = m.group(1) if m else None
        if not missing_call_id:
            raise
        log.warning("Rescuing: sending stub output for call_id=%s", missing_call_id)
        stub = _responses_create_traced(
            model=MODEL,
            input=[
                {
                    "type":   "function_call_output",
                    "call_id": missing_call_id,
                    "output":  "ERROR: tool interrupted (auto-generated stub)",
                }
            ],
            previous_response_id=last_resp_id,
            instructions=INSTRUCTIONS,
        )
        return _responses_create_traced(
            model=MODEL,
            input=msg,
            previous_response_id=stub.id,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=True,
        )

# ─────────────── 8. HTTP ENDPOINTS ───────────────
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
        # -------- Trace covers the whole backend workflow ----------
        with trace("flask_chat_workflow"):
            try:
                stream = _create_stream_with_rescue(msg, last_resp_id)
                _pipe(stream, q)
            except Exception as exc:                                        # pylint: disable=broad-except
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

# ---------- CSV / CRUD ROUTES & EXPORT ----------
@app.route("/", methods=["GET", "POST"])
def index():
    """Upload CSV with invoices or list current invoices."""
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
            len(invoices),
            len(new_clients),
            time.perf_counter() - t0,
        )
        flash("Invoices uploaded successfully.")
        return redirect("/")

    return render_template("index.html", invoices="")

@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    """Edit a single invoice inline (AJAX)."""
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
    """Delete an invoice and redirect to index."""
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
    flash("Invoice deleted.")
    return redirect("/")

@app.route("/export")
@app.route("/export/<int:invoice_id>")
def export_invoice(invoice_id: int | None = None):
    """
    Export a CSV with either one invoice (/export/<id>)
    or all invoices (/export).
    """
    rows = [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
    csv_data = pd.DataFrame(
        [
            {
                "client_name":  r.client.name,
                "client_email": r.client.email,
                "invoice_id":   r.invoice_id,
                "amount":       r.amount,
                "date_due":     r.date_due,
                "status":       r.status,
            }
            for r in rows
        ]
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

# ───────────────────────── 10. RUN APP ───────────────────────
if __name__ == "__main__":
    print("openai-python version:", openai.__version__)
    # Discover tools from all configured FastMCP servers
    for agent_name, base_url in AGENTS.items():
        if not base_url:
            log.warning("Agent %s has no base_url → skipped", agent_name)
            continue
        _register_agent(agent_name, base_url)

    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
