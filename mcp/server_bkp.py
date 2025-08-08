# -*- coding: utf-8 -*-
"""
Echo Server + Telco API helpers + **Final Contact-E-mail sender (STRICT hand-off)**

comments in the code always in English!
"""
import os
import re
import json
import time
import random
import requests
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# ───────────────────────── 0. STARTUP ─────────────────────────
start_time = time.time()

# ─────────────────────── 1. CONSTANTS / ENV ───────────────────
TELCO_BASE_URL = os.getenv("TELCO_BASE_URL", "*")

# Single bearer token for BOTH Telco endpoints **and** e-mail webhook
TELCO_QUERY_KEY = os.getenv("TELCO_QUERY_KEY", "*")

EMAIL_WEBHOOK_URL = os.getenv(
    "EMAIL_WEBHOOK_URL",
    "https://ringinx.app.n8n.cloud/webhook/583b15fc-04ae-4491-9876-12141188a029",
)

COMMON_HEADERS = {
    "Authorization": f"Bearer {TELCO_QUERY_KEY}",
    "Content-Type":  "application/json",
    "Accept":        "application/json",
}

# The only allowed recipient
# RECIPIENT = "viktor.voronov@ringinx.com"
RECIPIENT = os.getenv("RECIPIENT", "viktor.voronov@ringinx.com")

# ─────────────────────── 2. FAST-MCP SERVER ───────────────────
mcp = FastMCP(
    "Echo Server",
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
    json_response=True
    # instructions=(
    #     # Telco instructions
    #     "Two-step Telco flow:\n"
    #     "1) search_telco_dataset — exactly ONE namespace.\n"
    #     "2) get_telco_document   — for EVERY hit before answering.\n"
    #     "Repeat the pair for 'stirshaken' then 'cpni' if both are required.\n\n"
    #     # Handoff instructions
    #     "Contact handoff flow:\n"
    #     "• Collect (a) client_name, (b) company_name, (c) client_email, "
    #     "and (d) AI-generated summary of the request.\n"
    #     "• Confirm the company name if already known from FCC data.\n"
    #     "• Ask explicit permission: “Would you like me to forward a summary to Brita?”\n"
    #     "• Only AFTER consent and all four data points are ready, call send_contact_email.\n"
    #     "• Users NEVER write the message body; the AI composes it.\n"
    #     "• Every line in the e-mail body must end with ':)'.\n"
    #     "• The e-mail is ALWAYS sent to viktor.voronov@ringinx.com.\n"
    #     "• Validate every input before sending.\n"
    # ),
)

tool_names: list[str] = []

# ───────────────────────── 3. DEMO TOOLS ──────────────────────
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print(f"[debug-server] add({a}, {b})")
    return a + b
tool_names.append("add")


@mcp.tool()
def get_secret_word() -> str:
    """Return a random secret word."""
    print("[debug-server] get_secret_word()")
    return random.choice(["apple", "banana", "cherry"])
tool_names.append("get_secret_word")


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Fetch plain-text weather report from wttr.in."""
    print(f"[debug-server] get_current_weather({city})")
    resp = requests.get(f"https://wttr.in/{city}")
    return resp.text
tool_names.append("get_current_weather")


@mcp.tool()
def get_supper(word: str) -> str:
    """Generate a playful supper message."""
    print(f"[debug-server] get_supper({word})")
    return f"Supper! {word} Yeah!"
tool_names.append("get_supper")

# ───────────────────── 4. TELCO SEARCH TOOLS ──────────────────
@mcp.tool()
def search_telco_dataset(namespace: str,
                         user_input: str,
                         top_k: int = 2,
                         filter: dict | None = None) -> dict:
    """Search one Telco dataset and return the full documents."""
    print(f"[debug-server] search_telco_dataset({namespace=}, {user_input=})")

    payload = {
        "namespace": namespace,
        "user_input": user_input,
        "top_k":     max(1, top_k),
    }
    if filter:
        payload["filter"] = filter

    try:
        resp = requests.post(
            f"{TELCO_BASE_URL}/query",
            headers=COMMON_HEADERS,
            data=json.dumps(payload),
            timeout=20,
        )
        print(">>> payload", json.dumps(payload, indent=2))
        print(">>> status ", resp.status_code)
        print(">>> body   ", resp.text[:600])
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise Exception(f"Telco /query failed: {exc}") from exc

    return resp.json()

tool_names.append("search_telco_dataset")


@mcp.tool()
def get_telco_document(version: str, source_file: str) -> dict:
    """Retrieve a single document from the Telco knowledge base."""
    print(f"[debug-server] get_telco_document({version=}, {source_file=})")
    try:
        resp = requests.get(
            f"{TELCO_BASE_URL}/document",
            headers=COMMON_HEADERS,
            params={"version": version, "source_file": source_file},
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise Exception(f"Telco /document failed: {exc}") from exc

    return resp.json()

tool_names.append("get_telco_document")

# ───────────────────── 5. E-MAIL HANDOFF TOOL ─────────────────
_EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)

def _validate_email(addr: str) -> None:
    if not _EMAIL_REGEX.fullmatch(addr or ""):
        raise ValueError(f"Invalid e-mail address: {addr!r}")

#NAME_REGEX = re.compile(r"[A-Za-z][A-Za-z .,'&()-]{1,98}[A-Za-z]")
NAME_REGEX = re.compile(r"[A-Za-z][A-Za-z .,'&()-]{1,98}[A-Za-z.]")

def _validate_name(name: str, field: str) -> None:
    if not NAME_REGEX.fullmatch(name.strip()):
        raise ValueError(f"Invalid {field}: {name!r}")

@mcp.tool(
    name="send_contact_email",
    description=(
        "Send ONE final intake summary e-mail "
        "to viktor.voronov@ringinx.com (fixed recipient).\n"
        "Arguments required: client_name, company_name, client_email, summary.\n"
        "If user asks to send anywhere else, politely refuse — "
        "this function always delivers to the Brita intake team.\n"
        "Every line in the message body must end with ':)'."
        "Always show a draft of the email (without TO sections) and ask for confirmation before sending it."
    ),
)
def send_contact_email(client_name: str,
                       company_name: str,
                       client_email: str,
                       subject:str,
                    #    cc:str,
                       summary: str) -> dict:
    """
    Forward a consultation summary to BritaAI’s team.

    Parameters
    ----------
    client_name : str
        Full name of the requester.
    company_name : str
        Confirmed legal or trade name of the company.
    client_email : str
        E-mail of the requester (used for CC and context).
    summary : str
        AI-generated summary of the discussion (10-2000 chars).

    Returns
    -------
    dict with HTTP status and detail text.
    """
    print("[debug-server] send_contact_email("
          f"{client_name=}, {company_name=}, {client_email=}, "
          f"len(summary)={len(summary)})")

    # ─────────── validation ───────────
    _validate_name(client_name, "client_name")
    _validate_name(company_name, "company_name")
    _validate_email(client_email)

    summary = summary.strip()
    if not (10 <= len(summary) <= 2000):
        raise ValueError("`summary` must be 10-2000 characters long")

    # Ensure every line ends with :)
    body_lines = [
        (line.rstrip() + (" :)" if not line.rstrip().endswith(":)") else ""))
        for line in summary.splitlines()
    ]
    # Prepend identity block
    identity_block = [
        f"Name: {client_name} ",  
        f"Company: {company_name} ",
        f"Client e-mail: {client_email} ",
        "----",
        # f"+CC: {cc} ",
    ]
    body_html = "<br>".join(identity_block + body_lines)

    subject_ = f"Intake summary for {company_name} + {subject}"

    payload = {
        "email":   RECIPIENT,   # fixed recipient
        "subject": subject_,
        "message": body_html,
    }

    # ───────── HTTP POST ─────────
    try:
        resp = requests.post(
            EMAIL_WEBHOOK_URL,
            headers=COMMON_HEADERS,
            data=json.dumps(payload),
            timeout=20,
        )
        print(">>> EMAIL status", resp.status_code)
        print(">>> EMAIL resp  ", resp.text[:400])
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise Exception(f"E-mail webhook failed: {exc}") from exc

    return {"status_code": resp.status_code, "detail": resp.text}

tool_names.append("send_contact_email")

# ───────────────────── 6. HEALTH CHECK ───────────────────────
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Basic health-check endpoint."""
    return JSONResponse({
        "status":           "ok",
        "uptime_seconds":   int(time.time() - start_time),
        "registered_tools": tool_names,
        "telco_base_url":   TELCO_BASE_URL,
        "email_webhook":    EMAIL_WEBHOOK_URL,
        "recipient":        RECIPIENT,
    })

# ──────────────────────── 7. MAIN ────────────────────────────
if __name__ == "__main__":
    # Streamable-HTTP works great in local Docker; SSE is identical.
    mcp.run(transport="streamable-http")
    # mcp.run(transport="streamable-http")
    # mcp.run(transport="sse")
