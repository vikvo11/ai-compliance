# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (function calling & streaming)
# ================================================================

import os
import time
import json
import queue
import requests
import configparser
from collections.abc import Generator, Sequence
from typing import Any

from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session, Response
)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import openai
from openai import AssistantEventHandler

# -------------------------------------------------------------------
# Basic setup
# -------------------------------------------------------------------

os.makedirs("/app/data", exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -------------------------------------------------------------------
# OpenAI configuration
# -------------------------------------------------------------------

cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")            # optional; env vars override

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL         = cfg.get("DEFAULT", "model", fallback="gpt-4o-mini").strip()
ASSISTANT_ID  = cfg.get("DEFAULT", "assistant_id", fallback="").strip()
SYSTEM_PROMPT = cfg.get("DEFAULT", "system_prompt",
                        fallback="You are a helpful assistant.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# DB models
# -------------------------------------------------------------------

class Client(db.Model):
    __tablename__ = "client"
    id    = db.Column(db.Integer, primary_key=True)
    name  = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128), nullable=True)
    invoices = db.relationship(
        "Invoice", backref="client",
        lazy=True, cascade="all, delete-orphan"
    )

class Invoice(db.Model):
    __tablename__ = "invoice"
    id         = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount     = db.Column(db.Float , nullable=False)
    date_due   = db.Column(db.String(64), nullable=False)
    status     = db.Column(db.String(32), nullable=False)
    client_id  = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)

with app.app_context():
    db.create_all()

# -------------------------------------------------------------------
# Assistant–side helper functions
# -------------------------------------------------------------------

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Very small wrapper around open-meteo.com."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1}, timeout=5
        ).json().get("results")
        if not geo:
            return f"Location '{location}' not found"
        lat, lon = geo[0]["latitude"], geo[0]["longitude"]

        cw = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current_weather": True
            }, timeout=5
        ).json().get("current_weather", {})
        return (
            f"The temperature in {location} is {cw.get('temperature')} °C "
            f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}."
        )
    except Exception as exc:          # noqa: BLE001
        return f"Error fetching weather: {exc}"

def get_invoice_by_id(invoice_id: str) -> dict:
    """Return one invoice as dict or error message."""
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    if not inv:
        return {"error": f"Invoice {invoice_id} not found"}
    return {
        "invoice_id" : inv.invoice_id,
        "amount"     : inv.amount,
        "date_due"   : inv.date_due,
        "status"     : inv.status,
        "client_name": inv.client.name,
        "client_email": inv.client.email,
    }

# JSON schemas for the two tools
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

# -------------------------------------------------------------------
# Routes – UI helpers
# -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """Simple index with upload & list of invoices."""
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        df = pd.read_csv(f)
        for _, row in df.iterrows():
            client = Client.query.filter_by(name=row["client_name"]).first()
            if not client:
                client = Client(name=row["client_name"],
                                email=row.get("client_email"))
                db.session.add(client)
                db.session.flush()
            db.session.add(Invoice(
                invoice_id=row["invoice_id"],
                amount=row["amount"],
                date_due=row["date_due"],
                status=row["status"],
                client_id=client.id,
            ))
        db.session.commit()
        flash("Invoices uploaded successfully.")
        return redirect("/")

    return render_template("index.html",
                           invoices=Invoice.query.all())

@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
    flash("Invoice deleted.")
    return redirect("/")

@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    """Ajax helper to update invoice fields & client email."""
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount   = request.form["amount"]
    inv.date_due = request.form["date_due"]
    inv.status   = request.form["status"]
    inv.client.email = request.form["client_email"]
    db.session.commit()
    return jsonify({
        "client_name": inv.client.name,
        "invoice_id" : inv.invoice_id,
        "amount"     : inv.amount,
        "date_due"   : inv.date_due,
        "status"     : inv.status,
    })

@app.route("/export")
@app.route("/export/<int:invoice_id>")
def export_invoice(invoice_id: int | None = None):
    invs   = [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
    fname  = f"invoice_{invoice_id or 'all'}.csv"
    csv    = pd.DataFrame([{
        "client_name" : i.client.name,
        "client_email": i.client.email,
        "invoice_id"  : i.invoice_id,
        "amount"      : i.amount,
        "date_due"    : i.date_due,
        "status"      : i.status,
    } for i in invs]).to_csv(index=False)
    return (csv, 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": f'attachment; filename="{fname}"'
    })

# -------------------------------------------------------------------
# Assistants – synchronous JSON endpoint
# -------------------------------------------------------------------

def ensure_thread() -> str:
    """Create a thread once per Flask session."""
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
    return session["thread_id"]

def handle_tool_calls(
    thread_id: str,
    run_id: str,
    tool_calls: Sequence[Any],   # compatible with any openai-python version
) -> None:
    """Answer each requested tool, then post outputs."""
    outputs = []
    for call in tool_calls:
        fn_name = call.function.name
        args = json.loads(call.function.arguments)

        if fn_name == "get_current_weather":
            result = get_current_weather(**args)
        elif fn_name == "get_invoice_by_id":
            result = json.dumps(get_invoice_by_id(**args))
        else:
            result = f"Unknown tool {fn_name}"

        outputs.append({"tool_call_id": call.id, "output": result})

    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=outputs,
    )

@app.route("/chat", methods=["POST"])
def chat_sync():
    """POST → single assistant reply (blocking)."""
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    thread_id = ensure_thread()
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_msg
    )
    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=ASSISTANT_ID,
        model=MODEL, tools=TOOLS
    )

    # Poll until finished
    while True:
        status = client.beta.threads.runs.retrieve(thread_id=thread_id,
                                                   run_id=run.id)
        if status.status == "requires_action":
            handle_tool_calls(thread_id, run.id,
                              status.required_action.submit_tool_outputs.tool_calls)
        elif status.status == "completed":
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            return jsonify({"error": f"Run {status.status}"}), 500
        time.sleep(0.4)

    reply = client.beta.threads.messages.list(
        thread_id=thread_id, limit=1
    ).data[0].content[0].text.value
    return jsonify({"response": reply})

# -------------------------------------------------------------------
# Assistants – streaming SSE endpoint
# -------------------------------------------------------------------

class SSEHandler(AssistantEventHandler):
    """Collect deltas & handle tool calls during streaming."""
    def __init__(self, q: queue.Queue, thread_id: str):
        super().__init__()
        self._q = q
        self._thread_id = thread_id

    def on_text_delta(self, delta, snapshot):     # noqa: D401
        self._q.put(delta.value)

    def on_tool_call(self, tcall):               # noqa: D401
        handle_tool_calls(self._thread_id, tcall.run_id, [tcall])

    def on_end(self, _run):                      # noqa: D401
        self._q.put(None)                        # sentinel → stream finished

def sse_generator(q: queue.Queue) -> Generator[bytes, None, None]:
    """Yield tokens as SSE lines."""
    while True:
        tok = q.get()
        if tok is None:            # finished
            yield b"event: done\ndata: [DONE]\n\n"
            break
        yield f"data: {tok}\n\n".encode("utf-8")

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    thread_id = ensure_thread()
    client.beta.threads.messages.create(thread_id=thread_id,
                                        role="user", content=user_msg)

    q: queue.Queue[str | None] = queue.Queue()
    handler = SSEHandler(q, thread_id)

    # запуск + стрим
    _ = client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        model=MODEL,
        tools=TOOLS,
        event_handler=handler,
    )

    # headers: text/event-stream + no-cache
    return Response(
        sse_generator(q),
        mimetype="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=5005, debug=True)
