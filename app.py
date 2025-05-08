# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (blocking + streaming, SDK ≥ 1.23)
# NOTE: All comments are in English!

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections.abc import Generator, Sequence
from typing import Any, Optional

import configparser
import requests
import openai
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
)
from flask_sqlalchemy import SQLAlchemy

# ──────────────────────────────────────────────────────────────────────────────
# 0.  ENV & OPENAI
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL: Optional[str] = (
    cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "")).strip() or None
)
ASSISTANT_ID: str = cfg.get("DEFAULT", "assistant_id",
                            fallback=os.getenv("ASSISTANT_ID", "")).strip()

if not OPENAI_API_KEY:
    sys.exit("❌  OPENAI_API_KEY is not configured.")
if not ASSISTANT_ID:
    sys.exit("❌  ASSISTANT_ID is not configured.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FLASK & DB
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

# ──────────────────────────────────────────────────────────────────────────────
# 2.  ASSISTANT TOOLS
# ──────────────────────────────────────────────────────────────────────────────
def get_current_weather(location: str, unit: str = "celsius") -> str:  # noqa: D401
    """Return a short string with the current weather for the location."""
    try:
        print(f"[DEBUG] get_current_weather({location=}, {unit=})")
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=5,
        ).json().get("results")
        if not geo:
            return f"Location '{location}' not found."
        lat, lon = geo[0]["latitude"], geo[0]["longitude"]
        cw = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=5,
        ).json().get("current_weather", {})
        return (
            f"The temperature in {location} is {cw.get('temperature')} °C "
            f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching weather: {exc}"


def get_invoice_by_id(invoice_id: str) -> dict:
    """Look up an invoice in the database and return a JSON-serialisable dict."""
    print(f"[DEBUG] get_invoice_by_id({invoice_id=})")
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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
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
                "properties": {
                    "invoice_id": {"type": "string"},
                },
                "required": ["invoice_id"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ensure_thread() -> str:
    """Ensure there is an OpenAI thread ID stored in the Flask session."""
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
        print(f"[DEBUG] Created new thread ID: {session['thread_id']}")
    else:
        print(f"[DEBUG] Using existing thread ID: {session['thread_id']}")
    return session["thread_id"]


def _run_tool(c: Any) -> str:
    """Execute one tool call and return its output string."""
    args = json.loads(c.function.arguments)
    fn = c.function.name
    if fn == "get_current_weather":
        return get_current_weather(**args)
    if fn == "get_invoice_by_id":
        return json.dumps(get_invoice_by_id(**args))
    return f"Unknown tool {fn}"


def _wait_for_active_run(tid: str, timeout: float = 30.0) -> None:
    """Block until the last run in the thread is finished or timeout is reached."""
    done = {"completed", "failed", "cancelled", "expired"}
    runs = client.beta.threads.runs.list(thread_id=tid, limit=1)
    if not runs.data:
        return

    run = runs.data[0]
    end_at = time.time() + timeout
    while run.status not in done:
        if time.time() > end_at:
            raise RuntimeError(f"Previous run '{run.id}' is still active.")
        time.sleep(0.35)
        run = client.beta.threads.runs.retrieve(thread_id=tid, run_id=run.id)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  WEB ROUTES  (index / delete / edit / export)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        df = pd.read_csv(f)
        for _, row in df.iterrows():
            cl = Client.query.filter_by(name=row["client_name"]).first()
            if not cl:
                cl = Client(name=row["client_name"], email=row.get("client_email") or "")
                db.session.add(cl)
                db.session.flush()
            db.session.add(
                Invoice(
                    invoice_id=row["invoice_id"],
                    amount=row["amount"],
                    date_due=row["date_due"],
                    status=row["status"],
                    client_id=cl.id,
                )
            )
        db.session.commit()
        flash("Invoices uploaded successfully.")
        return redirect("/")

    return render_template("index.html", invoices=Invoice.query.all())


@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
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
    return csv_data, 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": f'attachment; filename="{fname}"',
    }

# ──────────────────────────────────────────────────────────────────────────────
# 5.  BLOCKING CHAT   (/chat)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat_sync():
    """Blocking (synchronous) chat endpoint."""
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    tid = ensure_thread()
    try:
        _wait_for_active_run(tid)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 429

    client.beta.threads.messages.create(thread_id=tid, role="user", content=user_msg)

    run = client.beta.threads.runs.create(
        thread_id=tid,
        assistant_id=ASSISTANT_ID,
        tools=TOOLS,
        **({"model": MODEL} if MODEL else {}),
    )

    while True:
        status = client.beta.threads.runs.retrieve(thread_id=tid, run_id=run.id)
        if status.status == "requires_action":
            calls = status.required_action.submit_tool_outputs.tool_calls
            outputs = [{"tool_call_id": c.id, "output": _run_tool(c)} for c in calls]
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=tid, run_id=run.id, tool_outputs=outputs
            )
        elif status.status == "completed":
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            return jsonify({"error": f"Run {status.status}"}), 500
        time.sleep(0.35)

    msgs = client.beta.threads.messages.list(thread_id=tid, order="desc")
    for msg in msgs.data:
        if msg.role == "assistant":
            return jsonify({"response": msg.content[0].text.value})
    return jsonify({"error": "No assistant response"}), 500

# ──────────────────────────────────────────────────────────────────────────────
# 6.  STREAMING CHAT  (/chat/stream)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_tool_calls(ev: Any) -> list[Any] | None:
    """Return tool_calls if present; otherwise None."""
    delta = getattr(ev.data, "delta", None)
    if delta and getattr(delta, "tool_calls", None):
        return delta.tool_calls

    sd = getattr(delta, "step_details", None)
    if sd and getattr(sd, "tool_calls", None):
        return sd.tool_calls

    step = getattr(ev.data, "step", None)
    if step and getattr(step, "tool_calls", None):
        return step.tool_calls

    ra = getattr(ev.data, "required_action", None)
    if ra and getattr(ra, "submit_tool_outputs", None):
        return ra.submit_tool_outputs.tool_calls
    return None


def _follow_stream_after_tools(
    run_id: str, calls: Sequence[Any], q: queue.Queue, tid: str
) -> None:
    """Execute tool calls and continue streaming the run."""
    outs = [{"tool_call_id": c.id, "output": _run_tool(c)} for c in calls]
    follow = client.beta.threads.runs.submit_tool_outputs(
        thread_id=tid,
        run_id=run_id,
        tool_outputs=outs,
        stream=True,
    )
    pipe_events(follow, q, tid, run_id)  # recurse


def pipe_events(events, q: queue.Queue, tid: str, run_id: str) -> None:
    """Iterate over events, pushing text chunks into the queue."""
    for ev in events:
        print("[EVENT]", ev.event, flush=True)

        if ev.event == "thread.message.delta":
            for part in (ev.data.delta.content or []):
                if part.type == "text":
                    q.put(part.text.value)

        elif (tcs := _extract_tool_calls(ev)) is not None:
            _follow_stream_after_tools(run_id, tcs, q, tid)

        elif ev.event == "thread.run.completed":
            q.put(None)
            return


def sse_generator(q: queue.Queue) -> Generator[bytes, None, None]:
    """Yield Server-Sent Events; send heartbeat every 20 s."""
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
    """Streaming chat endpoint (SSE)."""
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    tid = ensure_thread()
    try:
        _wait_for_active_run(tid)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 429

    client.beta.threads.messages.create(thread_id=tid, role="user", content=user_msg)
    q: queue.Queue[str | None] = queue.Queue()

    def consume():
        print("[DEBUG] consume() started", flush=True)
        first = client.beta.threads.runs.create(
            thread_id=tid,
            assistant_id=ASSISTANT_ID,
            tools=TOOLS,
            stream=True,
            **({"model": MODEL} if MODEL else {}),
        )
        pipe_events(first, q, tid, first.id)
        print("[DEBUG] consume() finished", flush=True)

    threading.Thread(target=consume, daemon=True).start()

    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("[DEBUG] Starting Flask app on 0.0.0.0:5005")
    app.run(host="0.0.0.0", port=5005, debug=True)
