# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Assistants (function calling & streaming)
# ================================================================

import os, time, json, queue, threading, requests, configparser
from collections.abc import Sequence, Generator
from typing import Any
from flask import Flask, render_template, request, redirect, flash, jsonify, session, Response
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
cfg = configparser.ConfigParser(); cfg.read("cfg/openai.cfg")
OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL          = cfg.get("DEFAULT", "model",          fallback="gpt-4o-mini").strip()
ASSISTANT_ID   = cfg.get("DEFAULT", "assistant_id",   fallback="").strip()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# DB models
# -------------------------------------------------------------------
class Client(db.Model):
    __tablename__ = "client"
    id = db.Column(db.Integer, primary_key=True)
    name  = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128))
    invoices = db.relationship("Invoice", backref="client",
                               lazy=True, cascade="all, delete-orphan")

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
# Assistant helper functions
# -------------------------------------------------------------------
def get_current_weather(location: str, unit: str = "celsius") -> str:
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": location, "count": 1}, timeout=5).json().get("results")
        if not geo:
            return f"Location '{location}' not found"
        lat, lon = geo[0]["latitude"], geo[0]["longitude"]
        cw = requests.get("https://api.open-meteo.com/v1/forecast",
                          params={"latitude": lat, "longitude": lon, "current_weather": True},
                          timeout=5).json().get("current_weather", {})
        return (f"The temperature in {location} is {cw.get('temperature')} °C "
                f"with wind {cw.get('windspeed')} m/s at {cw.get('time')}.")
    except Exception as exc:
        return f"Error fetching weather: {exc}"

def get_invoice_by_id(invoice_id: str) -> dict:
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    return {"error": f"Invoice {invoice_id} not found"} if not inv else {
        "invoice_id": inv.invoice_id, "amount": inv.amount,
        "date_due": inv.date_due, "status": inv.status,
        "client_name": inv.client.name, "client_email": inv.client.email,
    }

TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_weather",
        "description": "Get weather",
        "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_invoice_by_id",
        "description": "Lookup invoice",
        "parameters": {"type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"]}}},
]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_thread() -> str:
    if "thread_id" not in session:
        session["thread_id"] = client.beta.threads.create().id
    return session["thread_id"]

def handle_tool_calls(thread_id: str, run_id: str, calls: Sequence[Any]) -> None:
    outs = []
    for c in calls:
        fn = c.function.name
        args = json.loads(c.function.arguments)
        res = get_current_weather(**args) if fn == "get_current_weather" \
              else json.dumps(get_invoice_by_id(**args)) if fn == "get_invoice_by_id" \
              else f"Unknown tool {fn}"
        outs.append({"tool_call_id": c.id, "output": res})
    client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id,
                                                 run_id=run_id,
                                                 tool_outputs=outs)

# -------------------------------------------------------------------
# Routes — UI helpers (index / delete / edit)
# -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file."); return redirect(request.url)
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            cl = Client.query.filter_by(name=r["client_name"]).first()
            if not cl:
                cl = Client(name=r["client_name"], email=r.get("client_email"))
                db.session.add(cl); db.session.flush()
            db.session.add(Invoice(invoice_id=r["invoice_id"], amount=r["amount"],
                                   date_due=r["date_due"], status=r["status"],
                                   client_id=cl.id))
        db.session.commit(); flash("Invoices uploaded."); return redirect("/")
    return render_template("index.html", invoices=Invoice.query.all())

@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id:int):
    inv = Invoice.query.get_or_404(invoice_id); db.session.delete(inv); db.session.commit()
    flash("Invoice deleted."); return redirect("/")

@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id:int):
    inv = Invoice.query.get_or_404(invoice_id)
    inv.amount = request.form["amount"]; inv.date_due = request.form["date_due"]
    inv.status = request.form["status"]; inv.client.email = request.form["client_email"]
    db.session.commit()
    return jsonify({"client_name": inv.client.name, "invoice_id": inv.invoice_id,
                    "amount": inv.amount, "date_due": inv.date_due, "status": inv.status})

# ---- fixed decorator block -------------------------------------------------
@app.route("/export")
@app.route("/export/<int:invoice_id>")
def export_invoice(invoice_id:int | None = None):
    invs = [Invoice.query.get_or_404(invoice_id)] if invoice_id else Invoice.query.all()
    csv = pd.DataFrame([{"client_name": i.client.name, "client_email": i.client.email,
                         "invoice_id": i.invoice_id, "amount": i.amount,
                         "date_due": i.date_due, "status": i.status} for i in invs]).to_csv(index=False)
    fname = f"invoice_{invoice_id or 'all'}.csv"
    return (csv, 200, {"Content-Type": "text/csv",
                       "Content-Disposition": f'attachment; filename="{fname}"'})

# -------------------------------------------------------------------
# Blocking /chat endpoint
# -------------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat_sync():
    user_msg = (request.json or {}).get("message","").strip()
    if not user_msg: return jsonify({"error":"Empty message"}),400
    thread_id = ensure_thread()
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_msg)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID,
                                          model=MODEL, tools=TOOLS)
    while True:
        st = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if st.status == "requires_action":
            handle_tool_calls(thread_id, run.id, st.required_action.submit_tool_outputs.tool_calls)
        elif st.status == "completed":
            break
        elif st.status in {"failed","cancelled","expired"}:
            return jsonify({"error": f"Run {st.status}"}), 500
        time.sleep(0.4)
    reply = client.beta.threads.messages.list(thread_id=thread_id, limit=1).data[0].content[0].text.value
    return jsonify({"response": reply})

# -------------------------------------------------------------------
# Streaming SSE endpoint
# -------------------------------------------------------------------
class SSEHandler(AssistantEventHandler):
    def __init__(self, q:queue.Queue, tid:str):
        super().__init__(); self.q=q; self.tid=tid
    def on_text_delta(self, delta, _snapshot):
        self.q.put(getattr(delta, "delta", getattr(delta, "value", "")))
    def on_tool_call(self, t): handle_tool_calls(self.tid, t.run_id, [t])
    def on_end(self, _): self.q.put(None)

def sse_gen(q:queue.Queue) -> Generator[bytes,None,None]:
    while True:
        tok = q.get()
        if tok is None:
            yield b"event: done\ndata: [DONE]\n\n"; break
        yield f"data: {tok}\n\n".encode()

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message","").strip()
    if not user_msg: return jsonify({"error":"Empty message"}),400
    thread_id = ensure_thread()
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_msg)

    q:queue.Queue[str|None] = queue.Queue()
    handler = SSEHandler(q, thread_id)

    def consume():
        for _ in client.beta.threads.runs.stream(thread_id=thread_id,
                                                 assistant_id=ASSISTANT_ID,
                                                 model=MODEL,
                                                 tools=TOOLS,
                                                 event_handler=handler):
            pass
    threading.Thread(target=consume, daemon=True).start()

    return Response(sse_gen(q), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering":"no","Cache-Control":"no-cache"})

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context(): db.create_all()
    app.run(host="0.0.0.0", port=5005, debug=True)
