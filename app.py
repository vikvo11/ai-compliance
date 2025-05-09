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

# NEW: FCC helper module (fcc_ecfs.py must be in PYTHONPATH)


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

# ----- ➊ NEW: propagate FCC key to environment ------------------------
FCC_KEY = cfg.get("DEFAULT", "FCC_API_KEY",
                  fallback=os.getenv("FCC_API_KEY"))
if FCC_KEY:
    os.environ["FCC_API_KEY"] = FCC_KEY
# ----------------------------------------------------------------------
import fcc_ecfs

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
# 3. ASSISTANT TOOLS  (thread-local httpx.Client)
# ────────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = Timeout(6.0)
HTTP_LIMITS = Limits(max_keepalive_connections=20, max_connections=50)
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


@functools.lru_cache(maxsize=2048)
def _coords_for(city: str) -> tuple[float, float] | None:
    cli = _get_client()
    t0 = time.perf_counter()
    resp = cli.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    try:
        res = resp.json().get("results")
    finally:
        resp.close()
    log.debug("geocode '%s' %.3f s", city, time.perf_counter() - t0)
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


def _weather_for(lat: float, lon: float) -> dict:
    cli = _get_client()
    t0 = time.perf_counter()
    resp = cli.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
    )
    try:
        data = resp.json().get("current_weather", {})
    finally:
        resp.close()
    log.debug("weather API %.3f s", time.perf_counter() - t0)
    return data


def get_current_weather(location: str, unit: str = "celsius") -> str:
    t0 = time.perf_counter()
    try:
        coord = _coords_for(location)
        if not coord:
            return f"Location '{location}' not found."
        cw = _weather_for(*coord)
        return (
            f"The temperature in {location} is {cw.get('temperature')} °C "
            f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
        )
    finally:
        log.info("tool:get_current_weather('%s') %.3f s",
                 location, time.perf_counter() - t0)


def get_invoice_by_id(invoice_id: str) -> dict:
    t0 = time.perf_counter()
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    log.info("tool:get_invoice_by_id '%s' DB %.3f s",
             invoice_id, time.perf_counter() - t0)
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

# ──────────── FCC wrappers (use fcc_ecfs module) ────────────
def fcc_search_filings(company: str) -> list[dict]:
    """Return numbered list of PDFs for company."""
    return fcc_ecfs.search(company)


def fcc_get_filings_text(company: str, indexes: list[int]) -> dict:
    """Download & parse selected FCC PDFs (1-based indexes)."""
    return fcc_ecfs.get_texts(company, indexes)

# ────────────────────────────────────────────────────────────
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
    {
        "type": "function",
        "function": {
            "name": "fcc_search_filings",
            "description": "Search FCC ECFS for all PDF attachments of a company",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string",
                                "description": "Company name to search for"},
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
# 4. HELPERS (thread / tool plumbing)
# ────────────────────────────────────────────────────────────────────────
def ensure_thread() -> str:
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
        log.debug("new thread %s", session["thread_id"])
    return session["thread_id"]


def _safe_add_user_message(tid: str, content: str) -> str:
    try:
        client.beta.threads.messages.create(
            thread_id=tid, role="system", content=content)
        return tid
    except openai.BadRequestError as exc:
        if "while a run" not in str(exc):
            raise
        new_tid = client.beta.threads.create().id
        session["thread_id"] = new_tid
        client.beta.threads.messages.create(
            thread_id=new_tid, role="system", content=content)
        log.warning("thread %s locked → new %s", tid, new_tid)
        return new_tid


def _run_tool(c: Any) -> str:
    try:
        args = json.loads(c.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    fn = c.function.name
    t0 = time.perf_counter()
    log.info("▶ tool %s %s", fn, args)
    try:
        if fn == "get_current_weather":
            return get_current_weather(**args)
        if fn == "get_invoice_by_id":
            return json.dumps(get_invoice_by_id(**args))
        if fn == "fcc_search_filings":
            return json.dumps(fcc_search_filings(**args), ensure_ascii=False)
        if fn == "fcc_get_filings_text":
            return json.dumps(fcc_get_filings_text(**args), ensure_ascii=False)
        return f"Unknown tool {fn}"
    finally:
        log.info("▲ tool %s done %.3f s", fn, time.perf_counter() - t0)

# --- helpers needed by streaming code (MUST be above _extract_tool_calls) ---
def _arguments_ready(call: Any) -> bool:
    try:
        return bool(call.function.arguments) and json.loads(call.function.arguments) is not None
    except Exception:
        return False


def _tool_call_ready(call: Any) -> bool:
    return bool(getattr(call, "id", None)) and _arguments_ready(call)

# ────────────────────────────────────────────────────────────────────────
# 5. STREAMING CHAT (batch-flush + tool handling)
# ────────────────────────────────────────────────────────────────────────
def _extract_tool_calls(ev: Any, *, run_id_hint: str | None = None):
    delta = getattr(ev.data, "delta", None)
    if delta:
        tc = getattr(delta, "tool_calls", None)
        if tc and run_id_hint and all(_tool_call_ready(c) for c in tc):
            return tc, run_id_hint
        sd = getattr(delta, "step_details", None)
        if sd and getattr(sd, "tool_calls", None) and run_id_hint and \
                all(_tool_call_ready(c) for c in sd.tool_calls):
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


def _follow_stream_after_tools(run_id: str, calls, q: queue.Queue, tid: str):
    outs = [
        {"tool_call_id": c.id, "output": _run_tool(c)}
        for c in calls if _tool_call_ready(c)
    ]
    if not outs:
        return
    follow = client.beta.threads.runs.submit_tool_outputs(
        thread_id=tid, run_id=run_id,
        tool_outputs=outs, stream=True
    )
    pipe_events(follow, q, tid)


def _flush_buf(buf: list[str], q: queue.Queue):
    if buf:
        q.put("".join(buf))
        buf.clear()


def pipe_events(events, q: queue.Queue, tid: str):
    run_id: str | None = None
    t0 = time.perf_counter()
    run_created_at: float | None = None
    first_tok: float | None = None
    done_at: float | None = None
    n_frag = 0

    tok_buf: list[str] = []
    buf_chars = 0
    next_flush = time.perf_counter() + 0.10  # flush every 100 ms

    for ev in events:
        if ev.event == "thread.run.created":
            run_id = ev.data.id
            run_created_at = time.perf_counter()
            log.info("run %s created", run_id)
        elif ev.event.startswith("thread.run.step.") and hasattr(ev.data, "run_id"):
            run_id = ev.data.run_id

        if ev.event == "thread.message.delta":
            for part in ev.data.delta.content or []:
                if part.type == "text":
                    if first_tok is None:
                        first_tok = time.perf_counter()
                        log.info("run %s first-token %.3f s",
                                 run_id, first_tok - (run_created_at or t0))
                    n_frag += 1
                    tok_buf.append(part.text.value)
                    buf_chars += len(part.text.value)
                    if buf_chars >= 256 or time.perf_counter() >= next_flush:
                        _flush_buf(tok_buf, q)
                        buf_chars = 0
                        next_flush = time.perf_counter() + 0.10

        elif (tc_block := _extract_tool_calls(ev, run_id_hint=run_id)) is not None:
            tool_calls, run_id = tc_block
            _follow_stream_after_tools(run_id, tool_calls, q, tid)

        elif ev.event == "thread.run.completed":
            done_at = time.perf_counter()
            _flush_buf(tok_buf, q)
            log.info("run %s done (%d frags, %.3f s total)",
                     run_id, n_frag, done_at - t0)
            q.put(None)
            break

    if run_created_at and done_at:
        metrics = {
            "queue_to_run_created": f"{run_created_at - t0:.3f}s",
            "run_created_to_first_token": (
                f"{first_tok - run_created_at:.3f}s" if first_tok else "n/a"
            ),
            "first_token_to_done": (
                f"{done_at - first_tok:.3f}s" if first_tok else "n/a"
            ),
            "overall": f"{done_at - t0:.3f}s",
            "fragments": n_frag,
        }
        log.info("timing-summary %s %s",
                 run_id, json.dumps(metrics, ensure_ascii=False))


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
        try:
            with app.app_context():
                first = client.beta.threads.runs.create(
                    thread_id=tid,
                    assistant_id=ASSISTANT_ID,
                    tools=TOOLS,
                    stream=True,
                    **({"model": MODEL} if MODEL else {}),
                )
                pipe_events(first, q, tid)
        finally:
            _close_thread_client()
            log.info("chat_stream %.3f s", time.perf_counter() - overall_t0)

    threading.Thread(target=consume, daemon=True).start()
    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache"
        },
    )

# ────────────────────────────────────────────────────────────────────────
# 6. CSV / CRUD (unchanged from your version)
# ────────────────────────────────────────────────────────────────────────
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
        log.info("CSV import %d invoices, %d new clients %.3f s",
                 len(invoices), len(new_clients), time.perf_counter() - t0)
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
        if invoice_id else Invoice.query.all()
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

# ────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    log.info("Flask serving on 0.0.0.0:5005  (log-level=%s)", LOG_LEVEL)
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
