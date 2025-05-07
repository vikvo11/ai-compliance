# app.py
# All comments are in English, as requested.

import os
import time
import json
import requests
import configparser
from flask import (
    Flask, render_template, request, redirect,
    flash, jsonify, session
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
import pandas as pd
import openai

# -------------------------------------------------------------------
# Basic setup
# -------------------------------------------------------------------

# Ensure the data directory exists (for SQLite file)
os.makedirs("/app/data", exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")

# SQLAlchemy configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -------------------------------------------------------------------
# OpenAI configuration
# -------------------------------------------------------------------

cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")  # falls back to env vars if file is absent

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL           = cfg.get("DEFAULT", "model"         , fallback="gpt-3.5-turbo").strip()
SYSTEM_PROMPT   = cfg.get("DEFAULT", "system_prompt" , fallback="You are a helpful assistant.")
ASSISTANT_ID    = cfg.get("DEFAULT", "assistant_id"  , fallback="").strip()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# Database models
# -------------------------------------------------------------------

class Client(db.Model):
    __tablename__ = "client"

    id    = db.Column(db.Integer, primary_key=True)
    name  = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128), nullable=True)

    invoices = db.relationship(
        "Invoice", backref="client", lazy=True, cascade="all, delete-orphan"
    )

class Invoice(db.Model):
    __tablename__ = "invoice"

    id         = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount     = db.Column(db.Float, nullable=False)
    date_due   = db.Column(db.String(64), nullable=False)
    status     = db.Column(db.String(32), nullable=False)

    client_id  = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)

# Create tables if they do not exist
with app.app_context():
    db.create_all()

# -------------------------------------------------------------------
# Helper functions (tooling for the assistant)
# -------------------------------------------------------------------

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Very simple weather lookup using open-meteo.com APIs.
    """
    try:
        geo_r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=5,
        )
        geo_r.raise_for_status()
        results = geo_r.json().get("results")
        if not results:
            return f"Location '{location}' not found."

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        weather_r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
            },
            timeout=5,
        )
        weather_r.raise_for_status()
        cw = weather_r.json().get("current_weather", {})

        temp   = cw.get("temperature")
        wind   = cw.get("windspeed")
        time_s = cw.get("time")

        return (
            f"The temperature in {location} is {temp} °C with wind speed "
            f"{wind} m/s at {time_s}."
        )
    except Exception as exc:
        return f"Error fetching weather: {exc}"

def get_invoice_by_id(invoice_id: str) -> dict:
    """
    Small helper so the assistant can fetch an invoice by external ID.
    """
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

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  – show all invoices
    POST – (hidden in UI) upload CSV with invoices
    """
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or not f.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(request.url)

        df = pd.read_csv(f)
        for _, row in df.iterrows():
            client = Client.query.filter_by(name=row["client_name"]).first()
            if not client:
                client = Client(
                    name=row["client_name"],
                    email=row.get("client_email"),
                )
                db.session.add(client)
                db.session.flush()  # ensures client.id is available

            invoice = Invoice(
                invoice_id=row["invoice_id"],
                amount=row["amount"],
                date_due=row["date_due"],
                status=row["status"],
                client_id=client.id,
            )
            db.session.add(invoice)

        db.session.commit()
        flash("Invoices uploaded successfully.")
        return redirect("/")

    invoices = Invoice.query.all()
    # Minimal stub template usage – you can style it however you like.
    return render_template("index.html", invoices=invoices)

@app.route("/delete/<int:invoice_id>", methods=["POST"])
def delete_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)
    db.session.delete(inv)
    db.session.commit()
    flash("Invoice deleted successfully.")
    return redirect("/")

@app.route("/edit/<int:invoice_id>", methods=["POST"])
def edit_invoice(invoice_id: int):
    inv = Invoice.query.get_or_404(invoice_id)

    # Update invoice fields
    inv.amount   = request.form["amount"]
    inv.date_due = request.form["date_due"]
    inv.status   = request.form["status"]

    # Update related client
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
    """
    Download one or all invoices as CSV.
    """
    if invoice_id:
        invs = [Invoice.query.get_or_404(invoice_id)]
        filename = f"invoice_{invoice_id}.csv"
    else:
        invs = Invoice.query.all()
        filename = "invoices.csv"

    data = [{
        "client_name" : i.client.name,
        "client_email": i.client.email,
        "invoice_id"  : i.invoice_id,
        "amount"      : i.amount,
        "date_due"    : i.date_due,
        "status"      : i.status,
    } for i in invs]

    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)

    return (
        csv_data,
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )

# -------------------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    """
    POST JSON: {"message": "<user text>"}
    Returns:    {"response": "<assistant text>"}
    """
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        # Reuse the same thread for one browser session
        if "thread_id" not in session:
            session["thread_id"] = client.beta.threads.create().id
        thread_id = session["thread_id"]

        # Append user message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_msg,
        )

        # Kick off a run
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            model=MODEL,
            tools=[
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
            ],
        )

        # Poll until completion or tool call
        while True:
            status = client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )

            if status.status == "requires_action":
                # The assistant wants to call a tool
                for call in status.required_action.submit_tool_outputs.tool_calls:
                    fn_name = call.function.name
                    args = json.loads(call.function.arguments)

                    if fn_name == "get_current_weather":
                        output = get_current_weather(**args)
                    elif fn_name == "get_invoice_by_id":
                        output = json.dumps(get_invoice_by_id(**args))
                    else:
                        output = f"Unknown tool: {fn_name}"

                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=[{
                            "tool_call_id": call.id,
                            "output": output,
                        }],
                    )
                # Continue polling after answering tool calls
            elif status.status == "completed":
                break
            elif status.status in {"failed", "cancelled", "expired"}:
                return jsonify({"error": f"Run {status.status}"}), 500

            time.sleep(0.5)

        # Read the most recent assistant message
        msgs = client.beta.threads.messages.list(thread_id=thread_id, limit=1)
        if not msgs.data:
            return jsonify({"error": "No assistant response"}), 500

        assistant_reply = msgs.data[0].content[0].text.value
        return jsonify({"response": assistant_reply})

    except Exception as exc:
        # A top-level safeguard so the user sees an error
        return jsonify({"error": str(exc)}), 500

# -------------------------------------------------------------------
# Application entry-point
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Make sure tables exist when running via `python app.py`
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=5005, debug=True)
