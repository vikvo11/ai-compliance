# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (blocking + streaming, all SDK versions)
# All inline comments are in English, as requested.

import os
import time
import json
import queue
import threading
import requests
import sys
import configparser
from collections.abc import Sequence, Generator
from typing import Any, Optional

from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session, Response
)
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import openai
from openai import AssistantEventHandler

# ──────────────────────────────────────────────────────────────────────────────
# 0.  ENV & OPENAI
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
# Do NOT hard-code a model unless you must override what is configured in the assistant
MODEL: Optional[str] = cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "")).strip() or None
ASSISTANT_ID: str = cfg.get(
    "DEFAULT", "assistant_id", fallback=os.getenv("ASSISTANT_ID", "")
).strip()

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
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Return a short string with the current weather for the location.
    """
    try:
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
    """
    Look up an invoice in the database and return a JSON-serialisable dict.
    """
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
    """
    Ensure there is an OpenAI thread ID stored in the Flask session.
    """
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
    return session["thread_id"]


def handle_tool_calls(thread_id: str, run_id: str, calls: Sequence[Any]) -> None:
    """
    Execute tool calls requested by the assistant and submit results.
    """
    outputs = []
    for c in calls:
        fn = c.function.name
        args = json.loads(c.function.arguments)
        if fn == "get_current_weather":
            result = get_current_weather(**args)
        elif fn == "get_invoice_by_id":
            result = json.dumps(get_invoice_by_id(**args))
        else:
            result = f"Unknown tool {fn}"
        outputs.append({"tool_call_id": c.id, "output": result})

    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=outputs,
    )


def last_assistant_text(thread_id: str) -> str:
    """
    Return the last text block sent by the assistant, or a warning if none found.
    """
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="desc")
    for msg in msgs.data:
        if msg.role != "assistant":
            continue
        for blk in msg.content:
            if blk.type == "text":
                return blk.text.value
    return "⚠️ Assistant returned no text."


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
    rows = (
        [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
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
    return (
        csv_data,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )

# ──────────────────────────────────────────────────────────────────────────────
# 5.  BLOCKING CHAT   (/chat)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat_sync():
    """
    Blocking (synchronous) chat endpoint.
    """
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    tid = ensure_thread()
    client.beta.threads.messages.create(thread_id=tid, role="user", content=user_msg)

    run = client.beta.threads.runs.create(
        thread_id=tid,
        assistant_id=ASSISTANT_ID,
        **({"model": MODEL} if MODEL else {}),
        tools=TOOLS,
    )

    # Poll the run status until completion or error.
    while True:
        status = client.beta.threads.runs.retrieve(thread_id=tid, run_id=run.id)
        if status.status == "requires_action":
            handle_tool_calls(
                tid, run.id, status.required_action.submit_tool_outputs.tool_calls
            )
        elif status.status == "completed":
            break
        elif status.status in {"failed", "cancelled", "expired"}:
            return jsonify({"error": f"Run {status.status}"}), 500
        time.sleep(0.4)

    reply = last_assistant_text(tid)
    return jsonify({"response": reply})

# ──────────────────────────────────────────────────────────────────────────────
# 6.  STREAMING CHAT  (/chat/stream)
# ──────────────────────────────────────────────────────────────────────────────
class SSEHandler(AssistantEventHandler):
    """
    Custom event handler that places tokens in a queue for SSE.
    """

    def __init__(self, q: queue.Queue, tid: str):
        super().__init__()
        self.q = q
        self.tid = tid

    # ≥ 1.13 SDK  – called for each delta
    def on_text_delta(self, delta, _):
        self.q.put(getattr(delta, "delta", getattr(delta, "value", "")))

    # ≤ 1.12 SDK – called for each full chunk
    def on_text(self, txt):
        self.q.put(txt)

    def on_tool_call(self, tcall):
        handle_tool_calls(self.tid, tcall.run_id, [tcall])

    def on_end(self, _run):
        self.q.put(None)


def sse_generator(q: queue.Queue) -_
