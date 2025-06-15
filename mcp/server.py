import os
import json
import time
import random
import requests
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# Track when the server started
start_time = time.time()

# Create server
mcp = FastMCP("Echo Server",
              host="0.0.0.0",
              port=8000,
              stateless_http=True,
              json_response=True,
              instructions=(
        "Two-step Telco flow:\n"
        "1) search_telco_dataset — exactly ONE namespace.\n"
        "2) get_telco_document   — for EVERY hit before answering.\n"
        "If user needs both datasets, repeat the pair for "
        '"stirshaken" then "cpni".'
    ),)

# Keep your own registry of tool names
tool_names: list[str] = []

# ---------------------------------------------------------------------------
# Existing demo tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"[debug-server] add({a}, {b})")
    return a + b
tool_names.append("add")


@mcp.tool()
def get_secret_word() -> str:
    """Return a random secret word"""
    print("[debug-server] get_secret_word()")
    return random.choice(["apple", "banana", "cherry"])
tool_names.append("get_secret_word")


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Fetch weather from wttr.in for a given city"""
    print(f"[debug-server] get_current_weather({city})")
    endpoint = "https://wttr.in"
    response = requests.get(f"{endpoint}/{city}")
    return response.text
tool_names.append("get_current_weather")


@mcp.tool()
def get_supper(word: str) -> str:
    """Generate a playful supper message"""
    print(f"[debug-server] get_supper({word})")
    return f"Supper! {word} Yeah!"
tool_names.append("get_supper")

# ---------------------------------------------------------------------------
# NEW: Telco Industry Query API helpers
# ---------------------------------------------------------------------------

TELCO_BASE_URL = os.getenv(
    "TELCO_BASE_URL",
    "*"
)

# Bearer token: env → fallback default
TELCO_QUERY_KEY = os.getenv(
    "TELCO_QUERY_KEY",
    "*"
)

COMMON_HEADERS = {
    "Authorization": f"Bearer {TELCO_QUERY_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# ---------- STEP 1: /query ----------

@mcp.tool()
def search_telco_dataset(namespace: str,
                         user_input: str,
                         top_k: int = 2,
                         filter: dict | None = None) -> dict:
    """
    Search one telco dataset *and* immediately retrieve the full documents.

    Parameters
    ----------
    namespace : str
        "stirshaken", "cpni", or "" (empty string means **assistant decides**).
    user_input : str
        Free‑form search string.
    top_k : int, optional
        How many top matches to request (default 2).
    filter_ : dict, optional
        Optional Mongo‑style filter per OpenAPI spec.

    Returns
    -------
    dict
        {
            "total_found": int,
            "hits": [
                {
                    "version": "...",
                    "source_file": "...",
                    "document": {... full parsed doc ...}
                },
                ...
            ]
        }
    """
    print(f"[debug-server] search_telco_dataset({namespace=}, {user_input=})")

    # Build request payload
    payload = {
        "namespace": namespace,
        "user_input": user_input,
        "top_k": max(1, top_k),
    }
    if filter is not None:
        payload["filter"] = filter

    url = f"{TELCO_BASE_URL}/query"
    try:
        resp = requests.post(url, headers=COMMON_HEADERS, data=json.dumps(payload), timeout=20)
        print(">>> payload", json.dumps(payload, indent=2))
        print(">>> status ", resp.status_code)
        print(">>> body   ", resp.text[:600])

        resp.raise_for_status()
    except requests.RequestException as exc:
        # Re-raise as plain Exception so FastMCP serialises it
        raise Exception(f"Telco /query failed: {exc}") from exc

    return resp.json()

tool_names.append("search_telco_dataset")

# ---------- STEP 2: /document ----------

@mcp.tool()
def get_telco_document(version: str, source_file: str) -> dict:
    """
    Call GET /document for a single {version, source_file} pair
    obtained from a previous search_telco_dataset() call.
    """
    print(f"[debug-server] get_telco_document({version=}, {source_file=})")

    url = f"{TELCO_BASE_URL}/document"
    params = {"version": version, "source_file": source_file}

    try:
        resp = requests.get(url, headers=COMMON_HEADERS, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise Exception(f"Telco /document failed: {exc}") from exc

    return resp.json()

tool_names.append("get_telco_document")

# ---------------------------------------------------------------------------
# Health-check endpoint — unchanged
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Simple health-check endpoint."""
    uptime_seconds = int(time.time() - start_time)
    health = {
        "status": "ok",
        "registered_tools": tool_names,
        "uptime_seconds": uptime_seconds,
        # Expose which API base URL we are using (useful in staging/prod)
        "telco_base_url": TELCO_BASE_URL,
    }
    return JSONResponse(health)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # mcp.run(transport="sse")
