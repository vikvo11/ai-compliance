# cors.py
# -*- coding: utf-8 -*-
"""
CORS + Origin/Key guard for the chat-widget endpoints.

Configuration source
--------------------
1. **INI file** `cfg/partners.cfg` (section `[DEFAULT]`, key `ALLOWED_ORIGINS`)
2. Environment variable `ALLOWED_ORIGINS` (if not present in INI file)
3. Hardcoded demo JSON (fallback)

`ALLOWED_ORIGINS` — a JSON object like:
    {
        "https://client-a.com": "KEY_A",
        "http://127.0.0.1:5500": "LOCAL_DEV_KEY"
    }
"""

from __future__ import annotations

import os
import json
import logging
import configparser
from typing import Dict

from flask import request, abort, Response

log = logging.getLogger("cors")

_HEADER_WIDGET_KEY = "X-Widget-Key"       # custom header from the widget

# ───────────────────── 1) LOAD CONFIG ──────────────────────────
cfg = configparser.ConfigParser()
cfg.read("cfg/partners.cfg")               # <-- your path to the INI file

_raw = cfg.get(
    "DEFAULT",
    "ALLOWED_ORIGINS",
    fallback=os.getenv(
        "ALLOWED_ORIGINS",
        '{"http://127.0.0.1:5000": "MY_PARTNER_KEY"}',  # last resort fallback
    ),
)

_PROTECTED_PREFIXES = ("/chat", "/chat/stream")

try:
    ALLOWED: Dict[str, str] = json.loads(_raw)
    if not isinstance(ALLOWED, dict):
        raise ValueError("ALLOWED_ORIGINS must decode to a JSON object")
except Exception as exc:  # pylint: disable=broad-except
    log.error("ALLOWED_ORIGINS parse error, guard disabled: %s", exc)
    ALLOWED = {}

# ───────────────────── 2) HELPERS ──────────────────────────────
def _origin_ok(origin: str, key: str) -> bool:
    return origin in ALLOWED and ALLOWED[origin] == key


# ───────────────────── 3) FLASK HOOKS ──────────────────────────
def _before_request():
    if request.method == "OPTIONS" or not request.path.startswith(_PROTECTED_PREFIXES):
        return None

    origin = request.headers.get("Origin", "")
    key    = request.headers.get(_HEADER_WIDGET_KEY, "")
    print(f'origin={origin}, key = {key}')
    if _origin_ok(origin, key):
        return None

    abort(403, description="Forbidden – bad origin or widget key")


def _after_request(response: Response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED:
        response.headers["Access-Control-Allow-Origin"]  = origin
        response.headers["Vary"]                         = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            f"Content-Type, {_HEADER_WIDGET_KEY}"
        )
        response.headers["Access-Control-Max-Age"]       = "86400"
    return response


# ───────────────────── 4) PUBLIC API ───────────────────────────
def register_cors(app) -> None:
    """Attach the guard to a Flask app."""
    app.before_request(_before_request)
    app.after_request(_after_request)
    app.logger.info("CORS guard enabled: %d whitelisted origin(s)", len(ALLOWED))