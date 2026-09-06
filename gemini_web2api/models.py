"""Model definitions and mapping from Gemini frontend JS source."""

# MODE_CATEGORY enum from 028-6eb337387583.js:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

MODELS = {
    "gemini-3.7-flash": {
        "mode": 1, "think": 4,
        "desc": "Latest all-around model (Gemini 3.7 Flash)",
    },
    "gemini-3.6-flash": {
        "mode": 1, "think": 4,
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.5-flash": {
        "mode": 1, "think": 4,
        "desc": "Alias for gemini-3.6-flash (backend upgraded)",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 2, "think": 0,
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    # Modèles Pro : pas de backend distinct sans abonnement payant.
    # En anonyme (sans cookie Gemini Advanced), le backend ignore MODE 3 et renvoie Flash.
    # On les garde uniquement si un cookie est configuré, sinon ils ne sont pas exposés
    # pour ne pas faire croire à un "faux Pro" (consigne : use only what is possible).
    "gemini-3.1-pro": {
        "mode": 3, "think": 4,
        "desc": "Pro — requires Gemini Advanced cookie, otherwise same as Flash",
        "requires_cookie": True,
    },
    "gemini-3.1-pro-enhanced": {
        "mode": 3, "think": 4, "extra": {31: 2, 80: 3},
        "desc": "Pro enhanced — requires cookie (experimental)",
        "requires_cookie": True,
    },
    "gemini-auto": {
        "mode": 4, "think": 4,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 0,
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4,
        "desc": "Lightweight fast model",
    },
}


def _has_cookie() -> bool:
    """True si un cookie est configuré (fichier existant). Utilisé pour filtrer les modèles Pro."""
    from .config import CONFIG
    import os
    cf = CONFIG.get("cookie_file")
    return bool(cf and os.path.exists(cf))


def resolve_model(model_name: str, default: str = "gemini-3.6-flash"):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields).

    Unknown model names fall back to default rather than erroring,
    since upstream clients may request arbitrary model identifiers.

    Si le modèle requiert un cookie (Pro) et qu'aucun cookie n'est configuré,
    on fallback vers le modèle par défaut avec un log explicite — pas de faux Pro.
    """
    think_override = None
    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None
    cfg = MODELS.get(model_name)
    if not cfg:
        from .gemini import log
        log(f"Unknown model '{model_name}', falling back to '{default}'")
        model_name = default
        cfg = MODELS[default]
    # Pas de faux Pro : si cookie requis et absent, on retombe sur Flash
    if cfg.get("requires_cookie") and not _has_cookie():
        from .gemini import log
        log(f"Model '{model_name}' requires Gemini Advanced cookie — falling back to '{default}' (no fake Pro)")
        model_name = default
        cfg = MODELS[default]
    mode_id = cfg["mode"]
    think_mode = think_override if think_override is not None else cfg["think"]
    extra = cfg.get("extra")
    return model_name, mode_id, think_mode, None, extra


def available_models() -> dict:
    """Retourne les modèles réellement utilisables avec la config actuelle (filtre Pro sans cookie)."""
    if _has_cookie():
        return dict(MODELS)
    return {k: v for k, v in MODELS.items() if not v.get("requires_cookie")}
