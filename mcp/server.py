import time
import random
import requests
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# Track when the server started
start_time = time.time()

# Create server
mcp = FastMCP("Echo Server",host="0.0.0.0", port=8000)

# Keep your own registry of tool names
tool_names = []


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"[debug-server] add({a}, {b})")
    return a + b
# register name
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


# Health-check endpoint using FastMCP's custom_route
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """
    Simple health-check endpoint.
    Returns JSON with overall status and basic dependency checks.
    """
    uptime_seconds = int(time.time() - start_time)
    health = {
        "status": "ok",
        "registered_tools": tool_names,
        "uptime_seconds": uptime_seconds,
    }
    return JSONResponse(health)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
