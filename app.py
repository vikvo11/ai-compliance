# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (blocking + streaming, SDK ≥ 1.19)

from __future__ import annotations

import json, os, sys, time, queue, threading
from collections.abc import Generator, Sequence
from typing import Any, Optional

import configparser, requests, openai, pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session
from flask_sqlalchemy import SQLAlchemy

# ──────────────────────────────────────────────────────────────────────────────
# 0.  ENV & OPENAI
# ──────────────────────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL: Optional[str] = cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "")).strip() or None
ASSISTANT_ID: str = cfg.get("DEFAULT", "assistant_id", fallback=os.getenv("ASSISTANT_ID", "")).strip()

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
    invoices = db.relationship("Invoice", backref="client", lazy=True, cascade="all, delete-orphan")


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
    """Short weather string via Open-Meteo."""
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
        ).json()["current_weather"]
        return (
            f"Temp in {location}: {cw['temperature']} °C, "
            f"wind {cw['windspeed']} m/s ({cw['time']})."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Weather error: {exc}"


def get_invoice_by_id(invoice_id: str) -> dict:
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
                "properties": { "invoice_id": {"type": "string"} },
                "required": ["invoice_id"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _run_tool(c: Any) -> str:
    args = json.loads(c.function.arguments)
    if c.function.name == "get_current_weather":
        return get_current_weather(**args)
    if c.function.name == "get_invoice_by_id":
        return json.dumps(get_invoice_by_id(**args))
    return f"Unknown tool {c.function.name}"


def wait_for_last_run(tid: str, poll: float = 0.35) -> None:
    """Block until the most recent run in a thread is finished."""
    while True:
        lst = client.beta.threads.runs.list(thread_id=tid, order="desc", limit=1)
        if not lst.data:
            return
        st = lst.data[0].status
        if st in {"completed", "failed", "cancelled", "expired"}:
            return
        time.sleep(poll)


def ensure_thread() -> str:
    """Return a thread id, creating one if needed."""
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
    return session["thread_id"]

# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROUTES (index / CRUD / export)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Upload a valid CSV.")
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
        flash("Invoices uploaded.")
        return redirect("/")

    return render_template("index.html", invoices=Invoice.query.all())


@app.route("/delete/<int:iid>", methods=["POST"])
def delete_invoice(iid: int):
    db.session.delete(Invoice.query.get_or_404(iid))
    db.session.commit()
    flash("Invoice deleted.")
    return redirect("/")


@app.route("/edit/<int:iid>", methods=["POST"])
def edit_invoice(iid: int):
    inv = Invoice.query.get_or_404(iid)
    inv.amount = request.form["amount"]
    inv.date_due = request.form["date_due"]
    inv.status = request.form["status"]
    inv.client.email = request.form["client_email"]
    db.session.commit()
    return jsonify(get_invoice_by_id(inv.invoice_id))


@app.route("/export")
@app.route("/export/<int:iid>")
def export_invoice(iid: int | None = None):
    rows = [Invoice.query.get_or_404(iid)] if iid else Invoice.query.all()
    df = pd.DataFrame(
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
    )
    return df.to_csv(index=False), 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": f'attachment; filename="invoice_{iid or "all"}.csv"',
    }

# ──────────────────────────────────────────────────────────────────────────────
# 5.  BLOCKING CHAT   (/chat)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat_sync():
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty"}), 400

    tid = ensure_thread()
    wait_for_last_run(tid)                               # ← NEW
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)

    run = client.beta.threads.runs.create(
        thread_id=tid, assistant_id=ASSISTANT_ID, tools=TOOLS, **({"model": MODEL} if MODEL else {})
    )

    while True:
        st = client.beta.threads.runs.retrieve(thread_id=tid, run_id=run.id)
        if st.status == "requires_action":
            outs = [{"tool_call_id": c.id, "output": _run_tool(c)}
                    for c in st.required_action.submit_tool_outputs.tool_calls]
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=tid, run_id=run.id, tool_outputs=outs
            )
        elif st.status == "completed":
            break
        elif st.status in {"failed", "cancelled", "expired"}:
            return jsonify({"error": f"Run {st.status}"}), 500
        time.sleep(0.3)

    last = next(m for m in client.beta.threads.messages.list(thread_id=tid, order="desc").data
                if m.role == "assistant")
    return jsonify({"response": last.content[0].text.value})

# ──────────────────────────────────────────────────────────────────────────────
# 6.  STREAMING CHAT  (/chat/stream) – SDK 1.19
# ──────────────────────────────────────────────────────────────────────────────
def _extract_tool_calls(ev: Any) -> tuple[list[Any], str] | None:
    delta = getattr(ev.data, "delta", None)
    if delta and (tc := getattr(delta, "tool_calls", None)):
        return tc, ev.data.run_id
    step = getattr(ev.data, "step", None)
    if step and getattr(step, "tool_calls", None):
        return step.tool_calls, step.run_id
    ra = getattr(ev.data, "required_action", None)
    if ra and getattr(ra, "submit_tool_outputs", None):
        return ra.submit_tool_outputs.tool_calls, ev.data.id
    return None


def _continue_after_tools(tid: str, run_id: str, calls: Sequence[Any], q: queue.Queue) -> None:
    outs = [{"tool_call_id": c.id, "output": _run_tool(c)} for c in calls]
    cont = client.beta.threads.runs.submit_tool_outputs(
        thread_id=tid, run_id=run_id, tool_outputs=outs, stream=True
    )
    _pipe_events(cont, q, tid)  # recursion


def _pipe_events(events, q: queue.Queue, tid: str) -> None:
    for ev in events:
        if ev.event == "thread.message.delta":
            for part in (ev.data.delta.content or []):
                if part.type == "text":
                    q.put(part.text.value)
        elif (tc := _extract_tool_calls(ev)) is not None:
            calls, rid = tc
            _continue_after_tools(tid, rid, calls, q)
        elif ev.event == "thread.run.completed":
            q.put(None)
            return


def _sse(q: queue.Queue) -> Generator[bytes, None, None]:
    keep = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)
            if tok is None:
                yield b"event: done\ndata: [DONE]\n\n"; break
            yield f"data: {tok}\n\n".encode(); keep = time.time() + 20
        except queue.Empty:
            if time.time() > keep:
                yield b": keep-alive\n\n"; keep = time.time() + 20


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty"}), 400

    tid = ensure_thread()
    wait_for_last_run(tid)                               # ← NEW
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)

    q: queue.Queue[str | None] = queue.Queue()

    def _consume():
        first = client.beta.threads.runs.create(
            thread_id=tid, assistant_id=ASSISTANT_ID, tools=TOOLS,
            stream=True, **({"model": MODEL} if MODEL else {})
        )
        _pipe_events(first, q, tid)

    threading.Thread(target=_consume, daemon=True).start()

    return Response(
        _sse(q),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("[DEBUG] Flask on 0.0.0.0:5005")
    app.run(host="0.0.0.0", port=5005, debug=True)
