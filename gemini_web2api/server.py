"""HTTP server: OpenAI-compatible API endpoints."""
import json
import time
import uuid
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .config import CONFIG
from .models import MODELS, resolve_model
from .gemini import generate, generate_stream, log
from .tools import (
    messages_to_prompt, parse_tool_calls, google_contents_to_prompt,
    parse_google_function_calls, parse_google_calls_as_tool_calls,
    extract_openai_tool_defs, extract_google_tool_defs,
    generate_validated, StreamToolCallParser, validate_tool_arguments,
)
from .multimodal import detect_image_mime, fetch_image_bytes, upload_image
from .agent import run_agent_loop
from . import __version__


def _usage(prompt: str, text: str) -> dict:
    p = len(prompt) // 4
    c = len(text or "") // 4
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _openai_tool_event_chunks(cid: str, created: int, model_name: str, event: tuple, call_ids: dict) -> list:
    """Convert a StreamToolCallParser event into OpenAI streaming chunk dicts.

    call_ids maps parser tool index -> generated call id (assigned on tool_start).
    Returns a list (possibly empty) of chunk payloads.
    """
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
    else:  # tool_end: nothing to emit for OpenAI
        return []
    return [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}]


def _upload_images(images: list) -> list:
    """Upload images and return list of file references. Returns None if no images."""
    if not images:
        return None
    file_refs = []
    for item in images:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        data, mime = item
        if isinstance(data, str):
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


class GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        client_ip = self.client_address[0] if self.client_address else "-"
        log(f"{client_ip} {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _parse_body(self, body: bytes) -> dict:
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _read_request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in transfer_encoding.lower():
            chunks = []
            while True:
                size_line = self.rfile.readline()
                if not size_line:
                    break
                size_text = size_line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError:
                    raise ValueError("invalid chunked request body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)

        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _authorized(self):
        keys = CONFIG.get("api_keys") or []
        if not keys:
            return True
        # Authorization: Bearer <key>
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] in keys:
            return True
        # header keys (OpenAI x-api-key / Google x-goog-api-key)
        for h in ("x-api-key", "x-goog-api-key"):
            if self.headers.get(h, "") in keys:
                return True
        # query param ?key= (Gemini CLI native style)
        if "?" in self.path:
            for pair in self.path.split("?", 1)[1].split("&"):
                if pair.startswith("key=") and pair[4:] in keys:
                    return True
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            if self.path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in MODELS.items()
                ]})
            elif self.path.startswith("/v1beta/models"):
                self.send_json({"models": [
                    {"name": f"models/{n}", "displayName": n, "description": c["desc"],
                     "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]}
                    for n, c in MODELS.items()
                ]})
            elif self.path in ("/", "/health"):
                # /health est l'endpoint léger utilisé par le keepalive Render
                # et les sondes UptimeRobot/cron. Pas d'auth, réponse < 1KB.
                self.send_json({"status": "ok", "version": __version__, "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            body = self._read_request_body()
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            elif self.path == "/v1/responses":
                self._handle_responses(body)
            elif self.path == "/v1/messages":
                self._handle_anthropic(body)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

    # ─── /v1/chat/completions ─────────────────────────────────────────────────

    def _handle_chat(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        messages = req.get("messages", [])
        tool_defs = extract_openai_tool_defs(tools) if tools and tool_choice != "none" else []
        prompt, images = messages_to_prompt(messages, tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        executor_url = req.get("tool_executor_url")
        if executor_url and tool_defs:
            # Agent mode: the server runs the full model <-> tool loop via the
            # executor webhook and returns the final answer like a normal call.
            try:
                text, steps = run_agent_loop(
                    messages, tools, tool_choice, executor_url,
                    model_id, think_mode, extra_fields, file_refs,
                )
            except Exception as e:
                self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
                return
            msg = {"role": "assistant", "content": text or None}
            if stream:
                self._start_sse()
                role_chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                              "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
                content_chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                 "model": model_name, "choices": [{"index": 0, "delta": {"content": text or ""}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n".encode())
                end_chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(end_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.send_json({
                    "id": cid, "object": "chat.completion", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text or "")//4,
                              "total_tokens": (len(prompt)+len(text or ""))//4},
                    "agent_tool_calls": steps,
                })
            return

        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        created = int(time.time())

        if stream and not tool_defs:
            try:
                self._start_sse()
                first_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode())
                self.wfile.flush()
                full_text = []
                for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                    full_text.append(delta)
                end = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
                if include_usage:
                    usage_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                   "model": model_name, "choices": [],
                                   "usage": _usage(prompt, "".join(full_text))}
                    self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
            return

        if stream and tool_defs:
            # Real-time streaming with tool calls: text is forwarded as it
            # arrives, tool_call deltas are emitted as soon as each block is
            # recognized (no buffering of the full response).
            try:
                self._start_sse()
                first_chunk = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode())
                self.wfile.flush()
                parser = StreamToolCallParser()
                call_ids = {}
                out_chars = 0
                for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                    for event in parser.feed(delta):
                        if event[0] == "text":
                            out_chars += len(event[1])
                        elif event[0] == "tool_args":
                            out_chars += len(event[2])
                        for chunk in _openai_tool_event_chunks(cid, created, model_name, event, call_ids):
                            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                for event in parser.finish():
                    if event[0] == "text":
                        out_chars += len(event[1])
                    elif event[0] == "tool_args":
                        out_chars += len(event[2])
                    for chunk in _openai_tool_event_chunks(cid, created, model_name, event, call_ids):
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                finish = "tool_calls" if parser.tool_count else "stop"
                end = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
                if include_usage:
                    usage_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                   "model": model_name, "choices": [],
                                   "usage": {"prompt_tokens": len(prompt) // 4,
                                             "completion_tokens": out_chars // 4,
                                             "total_tokens": (len(prompt) + out_chars) // 4}}
                    self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
            return

        try:
            text, tool_calls = generate_validated(
                prompt, tool_defs, tool_choice, generate,
                model_id, think_mode, file_refs, extra_fields,
            )
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            self._start_sse()
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": model_name, "choices": [{"index": 0, "delta": msg, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            if include_usage:
                usage_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": model_name, "choices": [],
                               "usage": _usage(prompt, text or "")}
                self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text or "")//4,
                          "total_tokens": (len(prompt)+len(text or ""))//4},
            })

    # ─── /v1/responses (Codex CLI) ───────────────────────────────────────────

    def _handle_responses(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        input_items = req.get("input", [])
        tools = req.get("tools")
        messages = []
        if req.get("instructions"):
            messages.append({"role": "system", "content": req["instructions"]})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                         "name": item.get("name", ""), "content": item.get("output", "")})
                    elif item.get("type") in ("input_text", "input_image", "image"):
                        messages.append({"role": "user", "content": [item]})
                    elif item.get("role") == "assistant" or (item.get("type") == "message" and item.get("role") == "assistant"):
                        cp = item.get("content", [])
                        text_acc, tc_list = "", []
                        if isinstance(cp, list):
                            for c in cp:
                                if isinstance(c, dict):
                                    if c.get("type") == "output_text":
                                        text_acc += c.get("text", "")
                                    elif c.get("type") == "function_call":
                                        tc_list.append(c)
                        elif isinstance(cp, str):
                            text_acc = cp
                        m = {"role": "assistant", "content": text_acc or None}
                        if tc_list:
                            m["tool_calls"] = [{"id": tc.get("call_id", f"call_{i}"), "type": "function",
                                                "function": {"name": tc.get("name",""), "arguments": tc.get("arguments","{}")}}
                                               for i, tc in enumerate(tc_list)]
                        messages.append(m)
                    else:
                        role = item.get("role", "user")
                        messages.append({"role": role, "content": item.get("content", "")})

        if tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                     if t.get("type") == "function" and "function" not in t else t for t in tools]

        tool_choice = req.get("tool_choice", "auto")
        prompt, images = messages_to_prompt(messages, tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        tool_defs = extract_openai_tool_defs(tools) if tools and tool_choice != "none" else []
        agent_steps = []
        executor_url = req.get("tool_executor_url")
        agent_mode = bool(executor_url) and bool(tool_defs)
        try:
            file_refs = _upload_images(images)
            if agent_mode:
                text, agent_steps = run_agent_loop(
                    messages, tools, tool_choice, executor_url,
                    model_id, think_mode, extra_fields, file_refs,
                )
            else:
                text, tool_calls = generate_validated(
                    prompt, tool_defs, tool_choice, generate,
                    model_id, think_mode, file_refs, extra_fields,
                )
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return
        if agent_mode:
            tool_calls = None

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or not tool_calls:
            output.append({"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text or "", "annotations": []}]})

        if req.get("stream"):
            self._start_sse()
            sequence_number = 0

            def emit(event_type, **fields):
                nonlocal sequence_number
                sequence_number += 1
                event = {
                    "type": event_type,
                    "sequence_number": sequence_number,
                    **fields,
                }
                self.wfile.write(
                    f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()
                )

            usage = {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(text or "") // 4,
                "total_tokens": (len(prompt) + len(text or "")) // 4,
            }
            base_response = {
                "id": rid,
                "object": "response",
                "created_at": int(time.time()),
                "model": model_name,
            }
            emit(
                "response.created",
                response={
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            )
            emit(
                "response.in_progress",
                response={
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            )
            for output_index, item in enumerate(output):
                if item["type"] == "function_call":
                    pending_item = {
                        "type": "function_call",
                        "id": item["id"],
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": "",
                        "status": "in_progress",
                    }
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item=pending_item,
                    )
                    emit(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=item["arguments"],
                    )
                    emit(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=output_index,
                        arguments=item["arguments"],
                    )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
                elif item["type"] == "message":
                    pending_item = {
                        "type": "message",
                        "id": item["id"],
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item=pending_item,
                    )
                    for content_index, content_part in enumerate(item["content"]):
                        event_fields = {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "content_index": content_index,
                        }
                        emit(
                            "response.content_part.added",
                            **event_fields,
                            part={
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        )
                        emit(
                            "response.output_text.delta",
                            **event_fields,
                            delta=content_part["text"],
                        )
                        emit(
                            "response.output_text.done",
                            **event_fields,
                            text=content_part["text"],
                        )
                        emit(
                            "response.content_part.done",
                            **event_fields,
                            part=content_part,
                        )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
            emit(
                "response.completed",
                response={
                    **base_response,
                    "status": "completed",
                    "output": output,
                    "usage": usage,
                },
            )
            self.wfile.flush()
        else:
            resp_json = {"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed",
                         "model": model_name, "output": output,
                         "usage": {"input_tokens": len(prompt)//4, "output_tokens": len(text or "")//4, "total_tokens": (len(prompt)+len(text or ""))//4}}
            if agent_steps:
                resp_json["agent_tool_calls"] = agent_steps
            self.send_json(resp_json)

    # ─── /v1beta/models (Google Gemini CLI) ──────────────────────────────────

    def _handle_google_generate(self, body: bytes, stream: bool):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        model_name = m.group(1) if m else CONFIG["default_model"]
        model_name, model_id, think_mode, err, extra_fields = resolve_model(model_name)
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tool_defs = extract_google_tool_defs(req)
        has_tools = bool(tool_defs)
        prompt, images = google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return
        log(f"Google API: model={model_name} stream={stream} tools={has_tools} prompt_len={len(prompt)}")

        if stream and not has_tools:
            try:
                self._start_sse()
                full_text = ""
                for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                    if not delta:
                        continue
                    full_text += delta
                    chunk_obj = {
                        "candidates": [{"content": {"parts": [{"text": delta}], "role": "model"}, "index": 0}],
                        "modelVersion": model_name,
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                final_chunk = {
                    "candidates": [{"finishReason": "STOP", "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": len(prompt) // 4,
                        "candidatesTokenCount": len(full_text) // 4,
                        "totalTokenCount": (len(prompt) + len(full_text)) // 4,
                    },
                    "modelVersion": model_name,
                }
                self.wfile.write(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Google stream error: {e}")
            return

        if stream and has_tools:
            # Streaming with function calls: text deltas stream as they arrive;
            # each completed function_call block is emitted as one full part
            # (Google protocol expects complete args in a single functionCall).
            try:
                self._start_sse()
                parser = StreamToolCallParser()
                out_chars = 0
                pending_calls = {}  # parser index -> {"name":..., "args": dict}

                def _write(obj):
                    self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())

                def _handle_event(event):
                    nonlocal out_chars
                    kind = event[0]
                    if kind == "text":
                        out_chars += len(event[1])
                        _write({"candidates": [{"content": {"parts": [{"text": event[1]}], "role": "model"}, "index": 0}],
                                "modelVersion": model_name})
                    elif kind == "tool_start":
                        pending_calls[event[1]] = {"name": event[2], "args": {}}
                    elif kind == "tool_args":
                        out_chars += len(event[2])
                        try:
                            pending_calls[event[1]]["args"] = json.loads(event[2])
                        except (json.JSONDecodeError, KeyError):
                            pass
                    elif kind == "tool_end":
                        call = pending_calls.pop(event[1], None)
                        if call:
                            _write({"candidates": [{"content": {
                                "parts": [{"functionCall": {"name": call["name"], "args": call["args"]}}],
                                "role": "model"}, "index": 0}],
                                "modelVersion": model_name})

                for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                    for event in parser.feed(delta):
                        _handle_event(event)
                    self.wfile.flush()
                for event in parser.finish():
                    _handle_event(event)
                final_chunk = {
                    "candidates": [{"finishReason": "STOP", "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": len(prompt) // 4,
                        "candidatesTokenCount": out_chars // 4,
                        "totalTokenCount": (len(prompt) + out_chars) // 4,
                    },
                    "modelVersion": model_name,
                }
                _write(final_chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Google stream error: {e}")
            return

        try:
            text, tool_calls = generate_validated(
                prompt, tool_defs, "auto", generate,
                model_id, think_mode, file_refs, extra_fields,
                parse_fn=parse_google_calls_as_tool_calls, block_format="function_call",
            )
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        if not text and not tool_calls:
            log("Warning: empty response from Gemini")

        response_parts = []
        if tool_calls:
            if text:
                response_parts.append({"text": text})
            for call in tool_calls:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                response_parts.append({"functionCall": {"name": call["function"]["name"], "args": args}})
        else:
            response_parts.append({"text": text or "I apologize, but I was unable to generate a response. Please try again."})

        candidate = {
            "content": {"parts": response_parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text or "") // 4,
            "totalTokenCount": (len(prompt) + len(text or "")) // 4,
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self._start_sse()
            self.wfile.write(f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)

    # ─── /v1/messages (Anthropic-compatible) ─────────────────────────────────

    def _anthropic_to_internal(self, req: dict) -> tuple:
        """Convert an Anthropic /v1/messages request into (messages, tools, tool_choice).

        messages/tools use the internal OpenAI shape so the shared prompt
        builder and validation loop can be reused.
        """
        messages = []
        system = req.get("system")
        if system:
            if isinstance(system, list):
                system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            messages.append({"role": "system", "content": system})

        for msg in req.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            text_parts, tool_calls, tool_results = [], [], []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    })
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        inner = " ".join(b.get("text", "") for b in inner if isinstance(b, dict))
                    if not isinstance(inner, str):
                        inner = json.dumps(inner, ensure_ascii=False)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "name": block.get("name", ""),
                        "content": inner,
                    })
                elif btype == "image":
                    text_parts.append("[Image attached]")
            if tool_calls:
                messages.append({"role": "assistant", "content": " ".join(text_parts) or None,
                                 "tool_calls": tool_calls})
            elif text_parts:
                messages.append({"role": role, "content": " ".join(text_parts)})
            for tr in tool_results:
                messages.append(tr)

        tools = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        } for t in req.get("tools") or [] if isinstance(t, dict) and t.get("name")]

        tc = req.get("tool_choice") or {}
        tc_type = tc.get("type", "auto") if isinstance(tc, dict) else "auto"
        if tc_type == "none":
            tool_choice = "none"
        elif tc_type == "any":
            tool_choice = "required"
        elif tc_type == "tool" and tc.get("name"):
            tool_choice = {"type": "function", "function": {"name": tc["name"]}}
        else:
            tool_choice = "auto"
        return messages, tools, tool_choice

    def _handle_anthropic(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"type": "invalid_request_error", "message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"type": "invalid_request_error", "message": err}}, 400)
            return

        messages, tools, tool_choice = self._anthropic_to_internal(req)
        tool_defs = extract_openai_tool_defs(tools) if tools and tool_choice != "none" else []
        prompt, images = messages_to_prompt(messages, tools or None, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"type": "invalid_request_error", "message": "empty content"}}, 400)
            return

        try:
            file_refs = _upload_images(images)
        except RuntimeError as e:
            self.send_json({"error": {"type": "api_error", "message": f"upstream error: {e}"}}, 502)
            return
        log(f"Anthropic API: model={model_name} stream={req.get('stream')} tools={bool(tool_defs)} prompt_len={len(prompt)}")

        mid = f"msg_{uuid.uuid4().hex[:16]}"

        if req.get("stream"):
            self._start_sse()

            def _emit(event_type, payload):
                self.wfile.write(f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())

            _emit("message_start", {"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "model": model_name,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": len(prompt) // 4, "output_tokens": 0},
            }})
            parser = StreamToolCallParser()
            block_index = -1
            text_open = False
            tool_ids = {}
            out_chars = 0
            for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                for event in parser.feed(delta):
                    kind = event[0]
                    if kind == "text":
                        out_chars += len(event[1])
                        if not text_open:
                            block_index += 1
                            text_open = True
                            _emit("content_block_start", {"type": "content_block_start", "index": block_index,
                                                          "content_block": {"type": "text", "text": ""}})
                        _emit("content_block_delta", {"type": "content_block_delta", "index": block_index,
                                                      "delta": {"type": "text_delta", "text": event[1]}})
                    elif kind == "tool_start":
                        if text_open:
                            _emit("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            text_open = False
                        block_index += 1
                        tool_ids[event[1]] = (block_index, f"toolu_{uuid.uuid4().hex[:16]}")
                        bidx, call_id = tool_ids[event[1]]
                        _emit("content_block_start", {"type": "content_block_start", "index": bidx,
                                                      "content_block": {"type": "tool_use", "id": call_id,
                                                                        "name": event[2], "input": {}}})
                    elif kind == "tool_args":
                        out_chars += len(event[2])
                        bidx, _ = tool_ids[event[1]]
                        _emit("content_block_delta", {"type": "content_block_delta", "index": bidx,
                                                      "delta": {"type": "input_json_delta", "partial_json": event[2]}})
                    elif kind == "tool_end":
                        bidx, _ = tool_ids[event[1]]
                        _emit("content_block_stop", {"type": "content_block_stop", "index": bidx})
            for event in parser.finish():
                if event[0] == "text":
                    out_chars += len(event[1])
                    if not text_open:
                        block_index += 1
                        text_open = True
                        _emit("content_block_start", {"type": "content_block_start", "index": block_index,
                                                      "content_block": {"type": "text", "text": ""}})
                    _emit("content_block_delta", {"type": "content_block_delta", "index": block_index,
                                                  "delta": {"type": "text_delta", "text": event[1]}})
            if text_open:
                _emit("content_block_stop", {"type": "content_block_stop", "index": block_index})
            stop_reason = "tool_use" if parser.tool_count else "end_turn"
            _emit("message_delta", {"type": "message_delta",
                                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                    "usage": {"output_tokens": out_chars // 4}})
            _emit("message_stop", {"type": "message_stop"})
            self.wfile.flush()
            return

        try:
            text, tool_calls = generate_validated(
                prompt, tool_defs, tool_choice, generate,
                model_id, think_mode, file_refs, extra_fields,
            )
        except Exception as e:
            self.send_json({"error": {"type": "api_error", "message": f"upstream error: {e}"}}, 502)
            return

        usage = {"input_tokens": len(prompt) // 4, "output_tokens": (len(text or "")) // 4}
        for call in tool_calls or []:
            usage["output_tokens"] += len(call["function"].get("arguments", "")) // 4
        stop_reason = "tool_use" if tool_calls else "end_turn"

        def _content_blocks():
            blocks = []
            if text:
                blocks.append({"type": "text", "text": text})
            for call in tool_calls or []:
                try:
                    input_obj = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    input_obj = {}
                blocks.append({"type": "tool_use", "id": call["id"],
                               "name": call["function"]["name"], "input": input_obj})
            return blocks

        self.send_json({
            "id": mid, "type": "message", "role": "assistant", "model": model_name,
            "content": _content_blocks(),
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": usage,
        })


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
