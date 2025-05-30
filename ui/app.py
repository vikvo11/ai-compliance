"""
Simple Flask application acting as a Remote MCP server.
Comments in English as requested.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import logging
from urllib.parse import urlparse

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# In-memory list to store <script> tags for the front-end demo (dev only).
scripts = []

def is_valid_url(u: str) -> bool:
    """Basic URL validation."""
    try:
        r = urlparse(u)
        return all([r.scheme, r.netloc])
    except ValueError:
        return False

@app.route("/", methods=["GET", "POST"])
def index():
    """Simple HTML form to add script entries (unchanged except validation)."""
    if request.method == "POST":
        url = request.form.get("script_url", "").strip()
        if not is_valid_url(url):
            return "Invalid URL", 400
        scripts.append({
            "src": url,
            "backend": request.form.get("backend"),
            "stream": request.form.get("stream") or "true",
            "key": request.form.get("key") or "MY_PARTNER_KEY",
            "allowed_origin": request.form.get("allowed_origin") or "https://client.com",
            "defer": bool(request.form.get("defer")),
        })
        app.logger.info("Added script %s", url)
        return redirect(url_for("index"))
    return render_template("index.html", scripts=scripts)

@app.route("/delete/<int:script_id>", methods=["POST"])
def delete_script(script_id):
    """Remove a script entry by index (unchanged)."""
    if 0 <= script_id < len(scripts):
        scripts.pop(script_id)
    return redirect(url_for("index"))

@app.get("/health")
def health():
    """Lightweight liveness probe."""
    return jsonify(status="ok")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # NOTE: debug=True is for development only. Use a WSGI server (e.g. Gunicorn)
    # with debug=False in production.
    app.run(host="0.0.0.0", port=5006, debug=True)
