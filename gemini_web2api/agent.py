"""Server-side agent loop.

When a request includes ``tool_executor_url``, the proxy runs the full
model <-> tool loop itself:

    model call -> parse tool calls -> POST each to the executor URL
    -> feed results back -> repeat until the model stops calling tools

The client makes a single API call and receives the final answer, while its
own tool implementations live behind the executor webhook. Contract:

    POST {tool_executor_url}   (server -> client)
    {"call_id": "...", "name": "get_weather", "arguments": {"city": "Tokyo"}}

    Response (client -> server): a JSON object whose ``result``, ``output``,
    ``content`` or ``response`` field carries the result, or a plain-text body.
"""
import json
import urllib.request
import ssl

from .config import CONFIG
from .gemini import generate, _get_ssl_ctx, log
from .tools import messages_to_prompt, generate_validated, extract_openai_tool_defs


def _parse_args(arguments):
    """Arguments arrive as a JSON string from the model; hand the executor a dict."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments
    return arguments


def execute_tool_call(executor_url: str, call: dict, timeout: int = 30) -> str:
    """POST one tool call to the executor webhook and return the result text."""
    payload = {
        "call_id": call.get("id"),
        "name": call["function"]["name"],
        "arguments": _parse_args(call["function"].get("arguments")),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "gemini-web2api/1.1.0",
    }
    req = urllib.request.Request(executor_url, data=body, headers=headers, method="POST")
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")
    try:
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx),
            )
            resp = opener.open(req, timeout=timeout)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=timeout)
        raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"tool executor error: {e}") from e

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw.strip()
    if isinstance(data, dict):
        for key in ("result", "output", "content", "response"):
            if key in data and data[key] is not None:
                value = data[key]
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False)


def _execute_calls_parallel(calls: list, executor_url: str, timeout: int) -> list:
    """Exécute plusieurs tool calls en parallèle (ThreadPool). Retourne list[str] results dans l'ordre.

    Chaque call est indépendant ; une erreur/timeout d'un outil n'empêche pas les autres.
    Utilise max_workers = min(len(calls), 8) pour limiter la pression.
    """
    if not calls:
        return []
    if len(calls) == 1:
        try:
            return [execute_tool_call(executor_url, calls[0], timeout=timeout)]
        except Exception as e:
            return [f"tool executor error: {e}"]

    import concurrent.futures

    results = [None] * len(calls)
    max_workers = min(len(calls), 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(execute_tool_call, executor_url, call, timeout): idx
            for idx, call in enumerate(calls)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                # L'erreur est capturée comme résultat texte pour que Gemini puisse la voir et se corriger
                results[idx] = f"tool executor error: {e}"
                log(f"Tool {calls[idx]['function']['name']} failed: {e}")
    return results


def run_agent_loop(
    messages: list,
    tools: list,
    tool_choice,
    executor_url: str,
    model_id: int,
    think_mode: int,
    extra_fields: dict = None,
    file_refs: list = None,
) -> tuple:
    """Run the model <-> webhook tool loop until the model finishes.

    Returns (final_text, steps) where steps is a list of executed tool calls
    with their results, for observability.

    Améliorations Phase 5 :
    - appels parallèles quand un tour contient 2+ tool calls
    - validation des arguments déjà faite via generate_validated
    - timeout par outil, erreurs capturées comme résultats texte
    - protection boucle infinie via max_agent_turns + détection de répétition ?
    """
    max_turns = int(CONFIG.get("max_agent_turns", 8))
    tool_timeout = int(CONFIG.get("agent_tool_timeout_sec", 30))
    tool_defs = extract_openai_tool_defs(tools) if tools and tool_choice != "none" else []
    history = list(messages)
    steps = []
    nudged = False
    last_text = ""
    # Petite protection contre boucle infinie : si le même tool est appelé avec les mêmes args 3 fois
    seen_calls: dict = {}
    for turn in range(max_turns):
        prompt, _ = messages_to_prompt(history, tools, tool_choice)
        text, calls = generate_validated(
            prompt, tool_defs, tool_choice, generate,
            model_id, think_mode, file_refs, extra_fields,
        )
        last_text = text or ""
        if calls:
            # Détection boucle : même (name, args) répété trop souvent
            for call in calls:
                key = (call["function"]["name"], call["function"].get("arguments"))
                seen_calls[key] = seen_calls.get(key, 0) + 1
                if seen_calls[key] > 3:
                    log(f"Agent loop: stopping duplicate call loop for {key}")
                    return last_text, steps

            history.append({"role": "assistant", "content": text or None, "tool_calls": calls})

            # Exécution parallèle si plusieurs calls
            if len(calls) > 1:
                log(f"Agent loop turn {turn+1}: executing {len(calls)} tool calls in parallel")
            results = _execute_calls_parallel(calls, executor_url, tool_timeout)

            for call, result in zip(calls, results):
                steps.append({
                    "name": call["function"]["name"],
                    "arguments": call["function"].get("arguments"),
                    "result": result,
                })
                history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": result,
                })
            continue
        # No tool call this turn.
        if tool_choice == "required" and not nudged and turn < max_turns - 1:
            nudged = True
            history.append({
                "role": "user",
                "content": "[System: You MUST call a tool now. Output ONLY a ```tool_call``` block.]",
            })
            continue
        return last_text, steps
    log(f"Agent loop reached max turns ({max_turns})")
    return last_text, steps
