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
logging.basicConfig(level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z", force=True)
log = logging.getLogger("app")

# ─────────────────── 1. OPENAI ───────────────────────────────
cfg = configparser.ConfigParser(); cfg.read("cfg/openai.cfg")
OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY", fallback=os.getenv("OPENAI_API_KEY"))
MODEL = cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
if not OPENAI_API_KEY: log.critical("OPENAI_API_KEY missing"); sys.exit(1)
client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)

# ─────────────────── 2. FLASK & DB ───────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config.update(SQLALCHEMY_DATABASE_URI="sqlite:////app/data/data.db",
                  SQLALCHEMY_TRACK_MODIFICATIONS=False)
db = SQLAlchemy(app); os.makedirs("/app/data", exist_ok=True)

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

with app.app_context(): db.create_all()

# ─────────────────── 3. TOOL FUNCTIONS ───────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)

@functools.lru_cache(maxsize=1024)
def _coords_for(city: str) -> tuple[float, float] | None:
    r = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                  params={"name": city, "count": 1}, timeout=HTTP_TIMEOUT)
    res = r.json().get("results"); return None if not res else (res[0]["latitude"], res[0]["longitude"])

async def _weather_for(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        r = await ac.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon, "current_weather": True})
    return r.json().get("current_weather", {})

def get_current_weather(location: str, unit: str = "celsius") -> str:
    c = _coords_for(location);  cw = asyncio.run(_weather_for(*c)) if c else None
    if not cw: return f"Location '{location}' not found."
    t = cw["temperature"]; t = round(t*9/5+32,1) if unit=="fahrenheit" else t
    return f"The temperature in {location} is {t} {'°F' if unit=='fahrenheit' else '°C'}."

def get_invoice_by_id(invoice_id: str) -> dict:
    inv = Invoice.query.filter_by(invoice_id=invoice_id).first()
    return {"error": f"Invoice {invoice_id} not found"} if not inv else {
        "invoice_id": inv.invoice_id, "amount": inv.amount, "date_due": inv.date_due,
        "status": inv.status, "client_name": inv.client.name, "client_email": inv.client.email}

TOOLS = [
    {"name": "get_current_weather","type": "function","description": "Get current weather", "parameters": {
        "type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"default":"celsius"}},"required":["location"]}},
    {"name": "get_invoice_by_id","type": "function","description": "Return invoice data","parameters":{
        "type":"object","properties":{"invoice_id":{"type":"string"}},"required":["invoice_id"]}},
]
DISPATCH = {"get_current_weather": get_current_weather, "get_invoice_by_id": get_invoice_by_id}

# ─────────────────── 4. STREAM HANDLER ───────────────────────
def _run_tool(name: str, args: str) -> str:
    try: params = json.loads(args or "{}")
    except json.JSONDecodeError: return f"Bad JSON: {args}"
    fn = DISPATCH.get(name); res = fn(**params) if fn else f"Unknown tool {name}"
    return json.dumps(res) if isinstance(res, dict) else str(res)

# ─── helper: выполнить функцию и продолжить поток ────────────
def _finish_tool(resp_id: str, tc_id: str, name: str, args_json: str,
                 q: queue.Queue[str | None]) -> str:
    log.info("RUN  tool=%s  id=%s  args=%s", name, tc_id, args_json)
    out = _run_tool(name, args_json)
    follow = client.responses.submit_tool_outputs(
        response_id=resp_id,
        tool_outputs=[{"tool_call_id": tc_id, "output": out}],
        stream=True,
    )
    return _pipe(follow, q)      # рекурсивно продолжаем стрим


# ─── основной парсер потока ──────────────────────────────────
def _pipe(stream, q: queue.Queue[str | None]) -> str:
    resp_id = ""
    buf: dict[str, dict] = {}    # id → {"name": str, "parts": list[str]}

    for ev in stream:
        typ   = getattr(ev, "event", None) or getattr(ev, "type", None) or ""
        delta = getattr(ev, "delta", None)
        log.info("▶ %s   delta=%s", typ, bool(delta))

        # запоминаем последний response.id
        if getattr(ev, "response", None):
            resp_id = ev.response.id

        # текстовые токены (игнорируем куски JSON-аргументов)
        if isinstance(delta, str) and "arguments" not in typ:
            q.put(delta)

        # ── 1. старая ветка: response.tool_calls ───────────────
        if getattr(ev, "tool_calls", None):
            for tc in ev.tool_calls:
                buf[tc.id] = {"name": tc.function.name, "parts": []}
                full = getattr(tc.function, "arguments", None)
                if full:
                    resp_id = _finish_tool(resp_id, tc.id, tc.function.name, full, q)

        # ── 2. новая ветка: output_item.added → arguments.* ────
        if typ == "response.output_item.added":
            # достаём pydantic-item и приводим к dict
            item_obj = getattr(ev, "item", None) or getattr(ev, "output_item", None) \
                       or ev.model_dump(exclude_none=True).get("item")
            item = item_obj if isinstance(item_obj, dict) else \
                   item_obj.model_dump(exclude_none=True)
            if item and item.get("type") in ("function_call", "tool_call"):
                iid   = item["id"]
                fname = item["function"]["name"] if item["type"] == "function_call" \
                        else item["tool_call"]["name"]
                buf[iid] = {"name": fname, "parts": []}
                log.info("  declare id=%s func=%s", iid, fname)

        # накапливаем кусочки JSON-аргументов
        if "arguments.delta" in typ:
            iid = getattr(ev, "output_item_id", None) \
                  or getattr(ev, "item_id", None) \
                  or getattr(ev, "tool_call_id", None)
            if iid and iid in buf:
                buf[iid]["parts"].append(delta or "")

        # закрытие аргументов → исполняем
        if typ.endswith("arguments.done"):
            iid = getattr(ev, "output_item_id", None) \
                  or getattr(ev, "item_id", None) \
                  or getattr(ev, "tool_call_id", None)
            if iid and iid in buf:
                full_json = "".join(buf[iid]["parts"])
                resp_id = _finish_tool(resp_id, iid, buf[iid]["name"], full_json, q)
                buf.pop(iid, None)

        # ── 3. very old: required_action.submit_tool_outputs ───
        act = getattr(ev, "required_action", None)
        if act and getattr(act, "submit_tool_outputs", None):
            outs = [{"tool_call_id": tc.id, "output": _run_tool(tc.name, tc.args)}
                    for tc in act.submit_tool_outputs.tool_calls]
            follow = client.responses.submit_tool_outputs(
                response_id=ev.id, tool_outputs=outs, stream=True)
            resp_id = _pipe(follow, q)

        # завершение потока
        if typ in ("response.done", "response.completed", "response.output_text.done"):
            q.put(None)
            return resp_id

    q.put(None)
    return resp_id
    
# ─────────────────── 5. SSE ──────────────────────────────────
def sse(q: queue.Queue[str | None]) -> Generator[bytes, None, None]:
    keep = time.time() + 20
    while True:
        try:
            tok = q.get(timeout=1)
            if tok is None: yield b"event: done\ndata: [DONE]\n\n"; break
            yield f"data: {tok}\n\n".encode(); keep = time.time() + 20
        except queue.Empty:
            if time.time() > keep:
                yield b": ping\n\n"; keep = time.time() + 20

# ─────────────────── 6. CHAT ENDPOINT ───────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    msg = (request.json or {}).get("message","").strip()
    if not msg: return jsonify({"error":"Empty message"}),400
    last = session.get("prev_response_id"); q: queue.Queue[str|None]=queue.Queue()

    @copy_current_request_context
    def work():
        stream = client.responses.create(model=MODEL,input=msg,previous_response_id=last,
                                         tools=TOOLS,stream=True)
        session["prev_response_id"] = _pipe(stream, q)

    threading.Thread(target=work, daemon=True).start()
    return Response(sse(q), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering":"no","Cache-Control":"no-cache"})

# ─────────────────── 7. CSV / CRUD (оставьте как было) ───────
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


# ─────────────────── 8. RUN ──────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
