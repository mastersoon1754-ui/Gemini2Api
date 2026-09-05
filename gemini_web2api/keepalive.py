"""Self-ping keepalive for Render free tier.

Render met en veille les services gratuits après ~15 min sans trafic HTTP entrant.
Ce module lance un thread daemon qui se requête lui-même toutes les N secondes
(via RENDER_EXTERNAL_URL / KEEPALIVE_URL) pour générer du trafic entrant et
empêcher la mise en veille.

Activé automatiquement si l'une de ces variables/env est définie:
  - RENDER_EXTERNAL_URL (injectée automatiquement par Render)
  - KEEPALIVE_URL (override manuel, ex: https://mon-app.onrender.com)
  - config.json -> keepalive_url

Désactivé si:
  - keepalive_url == false / ""  ET aucune env var
  - keepalive_interval_sec <= 0
  - KEEPALIVE_INTERVAL_SEC=0

Le ping vise /health (léger, <1KB, pas d'auth) avec timeout 10s.
"""
import os
import time
import threading
import urllib.request
import ssl


def _resolve_keepalive_target():
    from .config import CONFIG

    # 1) Config file / CONFIG dict
    url = CONFIG.get("keepalive_url")

    # 2) Env overrides (manuels)
    env_url = os.environ.get("KEEPALIVE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    # Render expose parfois RENDER_EXTERNAL_HOSTNAME sans schéma
    if not env_url:
        hostname = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
        if hostname:
            env_url = f"https://{hostname}"

    # KEEPALIVE_URL / RENDER_EXTERNAL_URL prime sur config si présent
    if env_url:
        url = env_url

    # Normalisation: désactivé si vide / false / None
    if not url or (isinstance(url, str) and url.strip().lower() in ("0", "false", "disabled", "off")):
        return None, 0

    url = str(url).strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    # Interval: env prime sur config
    interval = CONFIG.get("keepalive_interval_sec", 600)
    env_interval = os.environ.get("KEEPALIVE_INTERVAL_SEC") or os.environ.get("KEEPALIVE_INTERVAL")
    if env_interval is not None:
        try:
            interval = int(str(env_interval).strip())
        except ValueError:
            pass
    try:
        interval = int(interval)
    except (ValueError, TypeError):
        interval = 600

    if interval <= 0:
        return None, 0

    # Cible = /health (endpoint léger sans auth)
    target = url.rstrip("/") + "/health"
    return target, interval


def start_keepalive():
    """Lance le thread keepalive si configuré. Retourne (target, interval) ou (None, 0)."""
    from .config import CONFIG
    try:
        from .gemini import log
    except ImportError:
        def log(msg):
            import sys
            if CONFIG.get("log_requests"):
                sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                sys.stderr.flush()

    target, interval = _resolve_keepalive_target()
    if not target:
        log("Keepalive: désactivé (pas de KEEPALIVE_URL / RENDER_EXTERNAL_URL)")
        return None, 0

    # Clamp: Render dort à 15 min => ping à 10 min max. Alerte si > 14 min.
    if interval > 840:
        log(f"Keepalive: interval {interval}s trop grand pour Render (veille à 900s), clamp à 600s")
        interval = 600

    def _loop():
        # Attendre que le serveur soit prêt avant le premier ping
        time.sleep(10)
        ctx = ssl.create_default_context()
        log(f"Keepalive: actif -> {target} toutes les {interval}s")
        while True:
            time.sleep(interval)
            try:
                req = urllib.request.Request(target, headers={"User-Agent": "gemini-web2api-keepalive/1.0"})
                # Pas de vérif proxy nécessaire: cible externe via Render
                with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                    resp.read(1024)
                log(f"Keepalive: ping OK {target} [{resp.status}]")
            except Exception as e:
                log(f"Keepalive: ping échoué {target}: {e}")

    t = threading.Thread(target=_loop, daemon=True, name="render-keepalive")
    t.start()
    return target, interval
