# app.py
# -*- coding: utf-8 -*-
# Flask + SQLAlchemy + OpenAI Responses API (blocking, псевдо-stream)
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

# ─────────────────────────────────────────────────────────────
# 0. LOGGING
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    force=True,
)
log = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────
# 1. ENV & OPENAI
# ─────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/openai.cfg")

OPENAI_API_KEY = cfg.get("DEFAULT", "OPENAI_API_KEY",
                         fallback=os.getenv("OPENAI_API_KEY"))
MODEL = cfg.get("DEFAULT", "model", fallback=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

if not OPENAI_API_KEY:
    log.critical("❌  OPENAI_API_KEY is not configured.")
    sys.exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=3)
log.info("OpenAI client ready (model=%s)", MODEL)

# ─────────────────────────────────────────────────────────────
# 2. FLASK & DB
# ─────────────────────────────────────────────────────────────
os.makedirs("/app/data", exist_ok=True)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
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

# ─────────────────────────────────────────────────────────────
# 3. TOOLS
# ─────────────────────────────────────────────────────────────
HTTP_TIMEOUT = httpx.Timeout(6.0)


@functools.lru_cache(maxsize=2048)
def _coords_for(city: str) -> tuple[float, float] | None:
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=HTTP_TIMEOUT,
    )
    res = resp.json().get("results")
    return None if not res else (res[0]["latitude"], res[0]["longitude"])


async def _weather_for(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as ac:
        resp = await ac.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
    return resp.json().get("current_weather", {})


def get_current_weather(location: str, unit: str = "celsius") -> str:
    coord = _coords_for(location)
    if not coord:
        return f"Location '{location}' not found."
    cw = asyncio.run(_weather_for(*coord))
    t = cw.get("temperature")
    if unit == "fahrenheit":
        t = round(t * 9 / 5 + 32, 1)
        sig = "°F"
    else:
        sig = "°C"
    return f"The temperature in {location} is {t} {sig} with wind {cw.get('windspeed')} m/s."


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
        "description": "Get current weather",
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
        "description": "Return invoice data",
        "parameters": {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
            "required": ["invoice_id"],
        },
    },
]

# map tool name → real function
_TOOL_DISPATCH = {
    "get_current_weather": get_current_weather,
    "get_invoice_by_id": get_invoice_by_id,
}

# ─────────────────────────────────────────────────────────────
# 4. CORE CHAT LOGIC (blocking, с циклом tool-calls)
# ─────────────────────────────────────────────────────────────
def chat_once(user_msg: str, prev_id: str | None) -> tuple[str, str]:
    """
    One turn: send user_msg, выполнять tool-calls пока модель не пришлёт
    окончательный текст. Возвращает (new_response_id, final_text).
    """
    resp = client.responses.create(
        model=MODEL,
        input=user_msg,
        previous_response_id=prev_id,
        tools=TOOLS,
    )

    # цикл: если нужны функции ➜ выполняем ➜ submit ➜ получаем новый resp
    while getattr(resp, "required_action", None):
        calls = resp.required_action.submit_tool_outputs.tool_calls
        outs = []
        for c in calls:
            args = json.loads(c.args)
            fn = _TOOL_DISPATCH[c.name]
            result = fn(**args)
            outs.append({"tool_call_id": c.id, "output": json.dumps(result) if isinstance(result, dict) else result})
        resp = client.responses.submit_tool_outputs(response_id=resp.id, tool_outputs=outs)

    return resp.id, resp.output_text


# ─────────────────────────────────────────────────────────────
# 5. SSE HELPERS
# ─────────────────────────────────────────────────────────────
def sse_from_text(text: str) -> Generator[bytes, None, None]:
    """Imitate streaming: отправляем слово каждые ~40 мс."""
    for word in text.split():
        yield f"data: {word} ".encode() + b"\n\n"
        time.sleep(0.04)
    yield b"event: done\ndata: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────
# 6. CHAT ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    prev_id = session.get("prev_response_id")

    @copy_current_request_context
    def compute(q: queue.Queue):
        try:
            new_id, answer = chat_once(user_msg, prev_id)
            session["prev_response_id"] = new_id
            q.put(answer)
        except Exception as exc:
            q.put(f"[error] {exc}")
        finally:
            q.put(None)

    q: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=compute, args=(q,), daemon=True).start()

    def event_stream():
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield from sse_from_text(chunk)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

# ─────────────────────────────────────────────────────────────
# 7. CSV / CRUD  — unchanged (точно ваш прежний код)
# ─────────────────────────────────────────────────────────────
# ...  (оставьте свой CRUD-блок без изменений) ...

# ─────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)
