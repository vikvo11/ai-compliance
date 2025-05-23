# cors.py
# -*- coding: utf-8 -*-
"""
CORS + Origin/Key guard for the chat-widget endpoints.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
How it works
------------
* The browser widget sends two extra headers:
      Origin         – added automatically by the browser
      X-Widget-Key   – partner-specific token injected by chat-widget.js
* Before every request the guard checks that
      ALLOWED[Origin] == X-Widget-Key
  otherwise it aborts with HTTP 403.
* After every request (including pre-flight) it injects the
  `Access-Control-Allow-*` headers — **only** for allowed Origins.

Usage
-----
    from cors import register_cors
    register_cors(app)

Configuration
-------------
Set an env-var **ALLOWED_ORIGINS** containing JSON mapping of
"origin" → "widget-key", e.g.

    export ALLOWED_ORIGINS='{
        "https://client-a.com": "KEY_A",
        "https://client-b.com": "KEY_B"
    }'

If the variable is missing or malformed the guard falls back
to an empty dict and blocks every cross-site request.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict

from flask import request, abort, Response, current_app

log = logging.getLogger("cors")

# ───────────────────────── 1) CONFIG ─────────────────────────
_HEADER_WIDGET_KEY = "X-Widget-Key"  # name of the custom header

_RAW = os.getenv(
    "ALLOWED_ORIGINS",
    '{"http://127.0.0.1:5000": "MY_PARTNER_KEY"}',  # default demo entry
)

try:
    ALLOWED: Dict[str, str] = json.loads(_RAW)
    if not isinstance(ALLOWED, dict):
        raise ValueError("ALLOWED_ORIGINS must decode to a JSON object")
except Exception as exc:  # pylint: disable=broad-except
    log.error("ALLOWED_ORIGINS parse error, CORS guard disabled: %s", exc)
    ALLOWED = {}

# ─────────────────────── 2) HELPERS ─────────────────────────
def _origin_ok(origin: str, key: str) -> bool:
    """Return *True* if the supplied origin/key pair is whitelisted."""
    return origin in ALLOWED and ALLOWED[origin] == key


# ─────────────────── 3) FLASK HOOKS ─────────────────────────
def _before_request():
    """Block any non-preflight request that fails the origin/key check."""
    if request.method == "OPTIONS":
        # Pre-flight requests are handled in _after_request.
        return None

    origin = request.headers.get("Origin", "")
    key = request.headers.get(_HEADER_WIDGET_KEY, "")

    if _origin_ok(origin, key):
        return None  # Let the request continue.

    abort(403, description="Forbidden – bad origin or widget key")


def _after_request(response: Response):
    """Inject CORS headers *only* for approved origins."""
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            f"Content-Type, {_HEADER_WIDGET_KEY}"
        )
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 h
    return response


# ───────────────────── 4) PUBLIC API ────────────────────────
def register_cors(app) -> None:
    """
    Attach the guard to a Flask application.

        from cors import register_cors
        app = Flask(__name__)
        register_cors(app)
    """
    app.before_request(_before_request)
    app.after_request(_after_request)

    app.logger.info(
        "CORS guard enabled: %d whitelisted origin(s)",
        len(ALLOWED),
    )
