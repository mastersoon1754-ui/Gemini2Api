"""Orchestrator — sépare API layer et Gemini Web backend.

Architecture cible :
    API (server.py)  →  Orchestrator (ce fichier)  →  GeminiBackend (gemini.py)
                                      ↓
                                  ToolSystem (tools.py)

Rôle :
- Ne connaît pas les détails SSE / OpenAI / Google / Anthropic (formatage reste dans server.py)
- Connaît : conversations (prompt + history), tool calls (validation, réinjection), agent loop, retries, streaming events.

Le backend Gemini Web (gemini.py) ne connaît pas OpenAI.
Le Tool System (tools.py) ne connaît pas Gemini.

Ce fichier est intentionnellement fin : il compose les trois couches sans les mélanger.
"""
import uuid

from .config import CONFIG
from .gemini import generate, generate_stream, log
from .tools import (
    messages_to_prompt,
    google_contents_to_prompt,
    extract_openai_tool_defs,
    extract_google_tool_defs,
    generate_validated,
    StreamToolCallParser,
    parse_google_calls_as_tool_calls,
)
from .multimodal import detect_image_mime, fetch_image_bytes, upload_image


# ─── Backend helpers (image upload reste côté backend) ────────────────────────

def upload_images(images: list) -> list | None:
    """Upload images via Scotty et retourne list[file_ref]. None si vide."""
    if not images:
        return None
    file_refs = []
    for item in images:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        data, mime = item
        if isinstance(data, str):  # URL distante
            data = fetch_image_bytes(data)
            mime = mime or "image/png"
        if not data:
            raise RuntimeError("image fetch failed")
        mime = detect_image_mime(data, mime or "image/png")
        try:
            ref = upload_image(data, "image.png", mime or "image/png")
            file_refs.append(ref)
        except Exception as e:
            raise RuntimeError(f"image upload failed: {e}") from e
    return file_refs if file_refs else None


# ─── Prompt builders (délègue à tools.py, mais centralise le choix) ──────────

def prepare_openai_prompt(messages: list, tools: list | None, tool_choice) -> tuple[str, list, list]:
    """Retourne (prompt, images, tool_defs) pour surface OpenAI."""
    tool_defs = extract_openai_tool_defs(tools) if tools and tool_choice != "none" else []
    prompt, images = messages_to_prompt(messages, tools, tool_choice)
    return prompt, images, tool_defs


def prepare_google_prompt(req: dict) -> tuple[str, list, list]:
    """Retourne (prompt, images, tool_defs) pour surface Google native."""
    tool_defs = extract_google_tool_defs(req)
    prompt, images = google_contents_to_prompt(req)
    return prompt, images, tool_defs


# ─── Single-turn (non-stream) ─────────────────────────────────────────────────

def complete(
    prompt: str,
    tool_defs: list,
    tool_choice,
    model_id: int,
    think_mode: int,
    file_refs: list | None = None,
    extra_fields: dict | None = None,
    parse_fn=None,
    block_format: str = "tool_call",
) -> tuple[str, list]:
    """Un tour modèle → (text, tool_calls) validés."""
    return generate_validated(
        prompt, tool_defs, tool_choice, generate,
        model_id, think_mode, file_refs, extra_fields,
        parse_fn=parse_fn, block_format=block_format,
    )


def complete_google(
    prompt: str,
    tool_defs: list,
    model_id: int,
    think_mode: int,
    file_refs: list | None = None,
    extra_fields: dict | None = None,
) -> tuple[str, list]:
    """Variante Google (tool_choice=auto, block_format=function_call)."""
    return complete(
        prompt, tool_defs, "auto", model_id, think_mode, file_refs, extra_fields,
        parse_fn=parse_google_calls_as_tool_calls, block_format="function_call",
    )


# ─── Streaming ────────────────────────────────────────────────────────────────

def stream_complete(
    prompt: str,
    model_id: int,
    think_mode: int,
    file_refs: list | None = None,
    extra_fields: dict | None = None,
):
    """Yield deltas texte bruts (sans tools)."""
    yield from generate_stream(prompt, model_id, think_mode, file_refs, extra_fields)


def stream_with_tools(
    prompt: str,
    model_id: int,
    think_mode: int,
    file_refs: list | None = None,
    extra_fields: dict | None = None,
):
    """Yield events StreamToolCallParser : (kind, ...).

    kind ∈ { "text", "tool_start", "tool_args", "tool_end" }
    Permet à l'API layer de formatter sans connaître le parsing.
    """
    parser = StreamToolCallParser()
    for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
        for event in parser.feed(delta):
            yield event
    for event in parser.finish():
        yield event


# ─── Agent loop (délègue à agent.py, future parallélisation ici) ────────────

def run_agent(
    messages: list,
    tools: list,
    tool_choice,
    executor_url: str,
    model_id: int,
    think_mode: int,
    extra_fields: dict | None = None,
    file_refs: list | None = None,
) -> tuple[str, list]:
    """Boucle agentique côté serveur. Retourne (final_text, steps)."""
    from .agent import run_agent_loop  # import tardif pour éviter cycle
    return run_agent_loop(messages, tools, tool_choice, executor_url, model_id, think_mode, extra_fields, file_refs)


# ─── Helpers SSE (formatage reste dans server.py, mais factorisé ici pour DRY) ─

def openai_tool_events_to_chunks(cid: str, created: int, model_name: str, event: tuple, call_ids: dict) -> list:
    """Convertit un event StreamToolCallParser en chunks OpenAI. Utilisé par server.py."""
    kind = event[0]
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name}
    if kind == "text":
        delta = {"content": event[1]}
    elif kind == "tool_start":
        _, idx, name = event
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        call_ids[idx] = call_id
        delta = {"tool_calls": [{"index": idx, "id": call_id, "type": "function",
                                 "function": {"name": name, "arguments": ""}}]}
    elif kind == "tool_args":
        _, idx, args_json = event
        delta = {"tool_calls": [{"index": idx, "function": {"arguments": args_json}}]}
    else:  # tool_end
        return []
    return [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}]


def usage(prompt: str, text: str) -> dict:
    p = len(prompt) // 4
    c = len(text or "") // 4
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
