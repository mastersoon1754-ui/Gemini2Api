"""Configuration management."""
import json
import os

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260830.05_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.6-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
    "max_agent_turns": 8,
    "agent_tool_timeout_sec": 30,
    "tool_max_tools": 96,
    "tool_max_prompt_chars": 120000,
    "tool_validate_retry": 1,
    # ── Keepalive (Render free tier) ──────────────────────────────
    # Si défini, le serveur se ping lui-même toutes les N secondes
    # pour éviter la mise en veille après 15 min d'inactivité.
    # Peut être une URL complète ou vide (auto-détection via
    # RENDER_EXTERNAL_URL / KEEPALIVE_URL). 0 = désactivé.
    "keepalive_url": None,
    "keepalive_interval_sec": 600,
}

CONFIG = dict(DEFAULT_CONFIG)


def load_config(path: str = None):
    """Load config from JSON file."""
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
    return CONFIG


def find_config():
    """Search for config file in standard locations."""
    for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None
