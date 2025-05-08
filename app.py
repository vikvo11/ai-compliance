# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (stream + tool calling)
from __future__ import annotations

import os, sys, time, json, queue, threading, asyncio, functools, logging
from collections.abc import Generator
from typing import Any

import httpx, pandas as pd, openai, configparser
from flask import (
    Flask, render_template, request, redirect, flash,
    jsonify, session, Response, copy_current_request_context
)
from flask_sqlalchemy import SQLAlchemy

# ─────────────────── 0. LOGGING ──────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ─────────────────── 1. ENV & OPENAI ─────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")
OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL = cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
if not OPENAI_API_KEY:
    log.critical("OPENAI_API_KEY is missing"); sys.exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# ─────────────────── 2. FLASK & DB ───────────────────────────
os.makedirs("/app/data", exist_ok=True)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(SQLALCHEMY_DATABASE_URI="sqlite:////app/data/data.db",
                  SQLALCHEMY_TRACK_MODIFICATIONS=False)
db = SQLAlchemy(app)


class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128))
    invoices = db.relationship("Invoice", backref="client", lazy=True, cascade="all, delete-orphan")


class Invoice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_due = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey("client.id"), nullable=False)


with app.app_context():
    db.create_all()

# ─────────────────── 3. TOOL FUNCTIONS ───────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)


@functools.lru_cache(maxsize=1024)
def _coords_for(city: str) -> tuple[float, float] | None:
    resp = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": city, "count": 1}, timeout=HTTP_TIMEOUT)
    res = resp.json().get("results")
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        resp = await ac.get("https://api.open-meteo.com/v1/forecast",
                            params={"latitude": lat, "longitude": lon, "current_weather": True})
    return resp.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    coord = _coords_for(location)
    if not coord:
        return f"Location '{location}' not found."
    cw = asyncio.run(_weather_for(*coord))
    t = cw.get("temperature")
    if unit == "fahrenheit":
        t = round(t * 9 / 5 + 32, 1); sig = "°F"
    else:
        sig = "°C"
    return f"The temperature in {location} is {t} {sig}."


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
        "name": "get_current_weather",
        "type": "function",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_invoice_by_id",
        "type": "function",
        "description": "Return invoice data by invoice_id",
        "parameters": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
        },
    },
]

_TOOL_MAP = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id": get_invoice_by_id,
}

# ─────────────────── 4. STREAM PARSER ────────────────────────
def _run_tool(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return f"Invalid JSON for {name}: {args_json}"
    fn = _TOOL_MAP.get(name)
    res = fn(**args) if fn else f"Unknown tool {name}"
    return json.dumps(res) if isinstance(res, dict) else str(res)


def _pipe(stream, q: queue.Queue[str | None]) -> str:
    """
    Collect deltas, handle tool calls, write into queue.
    Returns last response_id.
    """
    buf: dict[str, dict] = {}           # id -> {"name": str, "parts": list[str]}
    resp_id: str | None = None

    for ev in stream:
        typ = getattr(ev, "event", None) or getattr(ev, "type", None)

        # response starts
        if typ == "response.created":
            resp_id = ev.response.id

        # normal text token
        if hasattr(ev, "delta") and isinstance(ev.delta, str):
            q.put(ev.delta)

        # header with tool info
        if getattr(ev, "tool_calls", None):
            for tc in ev.tool_calls:
                buf[tc.id] = {"name": tc.function.name, "parts": []}

        # arguments chunk
        if typ and "arguments.delta" in typ:
            tc_id = getattr(ev, "tool_call_id", getattr(ev, "item_id", None))
            if tc_id and tc_id in buf:
                buf[tc_id]["parts"].append(ev.delta or "")

        # arguments done
        if typ and typ.endswith("arguments.done"):
            tc_id = getattr(ev, "tool_call_id", getattr(ev, "item_id", None))
            if tc_id and tc_id in buf:
                meta = buf.pop(tc_id)
                full_json = "".join(meta["parts"])
                output = _run_tool(meta["name"], full_json)
                follow = client.responses.submit_tool_outputs(
                    response_id=resp_id,
                    tool_outputs=[{"tool_call_id": tc_id, "output": output}],
                    stream=True,
                )
                resp_id = _pipe(follow, q)  # recurse, continue piping

        # end markers
        if typ in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None); return resp_id or ""

    q.put(None); return resp_id or ""

# ─────────────────── 5. SSE ──────────────────────────────────
def sse_gen(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    hb = time.time() + 20
    while True:
        try:
            chunk = q.get(timeout=1)
            if chunk is None:
                yield b"event: done\ndata: [DONE]\n\n"; break
            yield f"data: {chunk}\n\n".encode(); hb = time.time() + 20
        except queue.Empty:
            if time.time() > hb:
                yield b": keep-alive\n\n"; hb = time.time() + 20

# ─────────────────── 6. CHAT ENDPOINT ───────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    prev_id = session.get("prev_response_id")
    q: queue.Queue[str | None] = queue.Queue()

    @copy_current_request_context
    def worker():
        stream = client.responses.create(
            model=MODEL,
            input=user_msg,
            previous_response_id=prev_id,
            tools=TOOLS,
            stream=True,
        )
        new_id = _pipe(stream, q)
        session["prev_response_id"] = new_id

    threading.Thread(target=worker, daemon=True).start()
    return Response(sse_gen(q), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

# ─────────────────── 7. CSV / CRUD  (unchanged) ──────────────
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
    log.info("export %s", fname)
    return (
        csv_data,
        200,
        {"Content-Type": "text/csv",
         "Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ─────────────────── 8. ENTRY POINT ─────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
