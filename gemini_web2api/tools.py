"""Tool calling and multimodal message parsing."""
import json
import re
import uuid
import base64
import binascii
import io
from urllib.parse import unquote_to_bytes

from .config import CONFIG

MAX_IMAGE_B64_SIZE = 50000  # ~37KB raw image


def _minify_schema(schema):
    """Recursively strip schema keys that add prompt bytes without helping the
    model follow the schema ($schema/$id/title boilerplate, redundant
    additionalProperties)."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in ("$schema", "$id", "title"):
                continue
            if k == "additionalProperties" and v is False:
                continue
            out[k] = _minify_schema(v)
        return out
    if isinstance(schema, list):
        return [_minify_schema(v) for v in schema]
    return schema


def extract_openai_tool_defs(tools: list) -> list:
    """Normalize OpenAI-style tool definitions into compact dicts for the prompt.

    Schema minification keeps large agent toolsets (60+ tools) within a usable
    prompt budget without silently changing what the tools do.
    """
    defs = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool) if tool.get("type") == "function" else tool
        name = fn.get("name", tool.get("name", ""))
        if not name:
            continue
        defs.append({
            "name": name,
            "description": fn.get("description", tool.get("description", "")),
            "parameters": _minify_schema(fn.get("parameters", tool.get("parameters", {})) or {}),
        })
    return defs


def extract_google_tool_defs(req: dict) -> list:
    """Extract tool definitions from a Google-native request body."""
    defs = []
    fc_mode = req.get("toolConfig", {}).get("functionCallingConfig", {}).get("mode", "AUTO")
    if fc_mode == "NONE":
        return defs
    for tool_group in req.get("tools") or []:
        for fn in tool_group.get("functionDeclarations", []):
            td = {"name": fn.get("name", ""), "description": fn.get("description", "")}
            params = fn.get("parameters") or fn.get("parametersJsonSchema")
            if params:
                td["parameters"] = _minify_schema(params)
            defs.append(td)
    return defs


def validate_tool_arguments(args, schema, path="$") -> list:
    """Validate parsed tool arguments against a (subset of) JSON Schema.

    Checks: type, enum, required properties, and recurses into object
    properties and array items. Returns a list of human-readable errors
    (empty list = valid). Intentionally lenient on unknown properties.
    """
    errors = []
    if not isinstance(schema, dict):
        return errors
    expected = schema.get("type")
    if expected:
        type_map = {
            "object": dict, "array": list, "string": str,
            "boolean": bool, "null": type(None),
        }
        if expected in type_map:
            py = type_map[expected]
            ok = isinstance(args, py) and not (py is bool and not isinstance(args, bool))
            if expected == "boolean":
                ok = isinstance(args, bool)
            if not ok:
                return [f"{path}: expected {expected}, got {type(args).__name__}"]
        elif expected == "integer":
            if not isinstance(args, int) or isinstance(args, bool):
                return [f"{path}: expected integer, got {type(args).__name__}"]
        elif expected == "number":
            if not isinstance(args, (int, float)) or isinstance(args, bool):
                return [f"{path}: expected number, got {type(args).__name__}"]
    if "enum" in schema and isinstance(args, (str, int, float, bool)) and args not in schema["enum"]:
        errors.append(f"{path}: value {args!r} not in enum {schema['enum']}")
    if isinstance(args, dict):
        for req_key in schema.get("required", []):
            if req_key not in args:
                errors.append(f"{path}: missing required property '{req_key}'")
        props = schema.get("properties", {})
        for k, v in args.items():
            if k in props:
                errors.extend(validate_tool_arguments(v, props[k], f"{path}.{k}"))
    if isinstance(args, list) and "items" in schema:
        for i, item in enumerate(args):
            errors.extend(validate_tool_arguments(item, schema["items"], f"{path}[{i}]"))
    return errors


def _compress_b64_if_needed(b64: str) -> str:
    """Compress image if base64 is too large for text embedding."""
    if len(b64) <= MAX_IMAGE_B64_SIZE:
        return b64
    try:
        from PIL import Image
        img_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_data))
        # Resize to max 256px on longest side
        max_dim = 256
        ratio = min(max_dim / img.width, max_dim / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        # Convert to JPEG with quality reduction
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=60)
        compressed = base64.b64encode(buf.getvalue()).decode()
        return compressed
    except Exception:
        # If PIL not available, truncate (model will get partial data)
        return b64[:MAX_IMAGE_B64_SIZE]


def _build_tool_choice_instruction(tool_choice, tool_defs: list) -> str:
    """Build tool_choice constraint instruction.

    tool_choice values:
      - "none": do not call any tool
      - "auto": decide whether to call tools (default)
      - "required": must call at least one tool
      - {"type": "function", "function": {"name": "xxx"}}: must call specific tool
    """
    if tool_choice == "none":
        return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
    if tool_choice == "required":
        return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
    if isinstance(tool_choice, dict):
        fn_name = tool_choice.get("function", {}).get("name", "")
        if fn_name:
            return f'\n\nIMPORTANT: You MUST call the tool "{fn_name}". Do not call other tools.'
    return ""


def _decode_data_url(url: str):
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
    if not match:
        return None
    mime = match.group(1) or "image/png"
    is_base64 = bool(match.group(2))
    data = match.group(3)
    try:
        if is_base64:
            return base64.b64decode(data, validate=True), mime
        return unquote_to_bytes(data), mime
    except (ValueError, TypeError, binascii.Error):
        return None


def _image_from_url(url: str, mime: str = None):
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        return _decode_data_url(url)
    return url, mime or "image/png"


def _image_from_part(part: dict):
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url", {})
        if isinstance(image_url, dict):
            return _image_from_url(image_url.get("url"), image_url.get("mime_type"))
        return _image_from_url(image_url)
    if part_type in ("input_image", "image"):
        image_url = part.get("image_url") or part.get("url")
        if isinstance(image_url, dict):
            return _image_from_url(image_url.get("url"), image_url.get("mime_type"))
        if image_url:
            return _image_from_url(image_url, part.get("mime_type"))
        image_data = part.get("data") or part.get("base64")
        if isinstance(image_data, str):
            mime = part.get("mime_type") or part.get("media_type") or "image/png"
            if image_data.startswith("data:"):
                return _decode_data_url(image_data)
            try:
                return base64.b64decode(image_data, validate=True), mime
            except (ValueError, TypeError, binascii.Error):
                return None
    return None


def messages_to_prompt(messages: list, tools: list = None, tool_choice=None) -> tuple:
    """Convert OpenAI messages to (prompt_str, images_list).

    Returns (prompt, images) where images is a list of (bytes, mime_type) tuples.
    """
    parts = []
    images = []

    if tools and tool_choice != "none":
        tool_defs = extract_openai_tool_defs(tools)
        if tool_defs:
            # Budgets come from CONFIG so large agent toolsets (opencode sends
            # ~80 tools with MCP extras) fit without silent truncation.
            # Core tools are always kept; remaining budget is filled smallest-first.
            max_tools = int(CONFIG.get("tool_max_tools", 64))
            max_chars = int(CONFIG.get("tool_max_prompt_chars", 120000))
            CORE_TOOLS = ("bash", "edit", "write", "read", "grep", "glob", "webfetch",
                          "question", "lsp", "task", "todowrite", "skill",
                          "list_mcp_resources", "read_mcp_resource",
                          "list_mcp_resource_templates")
            forced = None
            if isinstance(tool_choice, dict):
                forced = tool_choice.get("function", {}).get("name")

            def _rank(t):
                n = t.get("name", "")
                if forced and n == forced:
                    return (0, 0)
                if n in CORE_TOOLS:
                    return (1, CORE_TOOLS.index(n))
                return (2, len(json.dumps(t)))

            tool_defs = sorted(tool_defs, key=_rank)
            kept, size = [], 0
            for t in tool_defs:
                tsize = len(json.dumps(t))
                if len(kept) >= max_tools or (kept and size + tsize > max_chars):
                    break
                kept.append(t)
                size += tsize
            dropped = [t["name"] for t in tool_defs[len(kept):]]
            if dropped:
                from .gemini import log
                log(f"Tool budget exceeded: dropped {len(dropped)} tools: {', '.join(dropped[:10])}"
                    + ("..." if len(dropped) > 10 else ""))
            tool_defs = kept
            tool_json = json.dumps(tool_defs)  # compact block: stays under Gemini's safety thresholds
            constraint = _build_tool_choice_instruction(tool_choice, tool_defs)
            parts.append(
                "# Tool Use\n\n"
                "You can call the following tools. Call format:\n"
                '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
                "When calling tools, output ONLY the tool_call block(s).\n"
                "Rules:\n"
                '- To call several tools in one turn, output multiple tool_call blocks.\n'
                '- "arguments" must be a JSON object satisfying the tool\'s parameters schema,\n'
                '  including every "required" field.\n'
                "- After receiving a [Tool result for ...], use that data to continue.\n\n"
                f"Available tools:\n{tool_json}"
                f"{constraint}"
            )

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for c in content:
                if c.get("type") in ("text", "input_text"):
                    text_parts.append(c.get("text", ""))
                else:
                    image = _image_from_part(c)
                    if image:
                        images.append(image)
                        text_parts.append("[Image attached]")
            content = " ".join(text_parts)

        if role == "system":
            parts.append(f"[System instruction]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_strs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tc_strs.append(
                        f'```tool_call\n{{"name": "{fn.get("name")}", '
                        f'"arguments": {fn.get("arguments", "{}")}}}\n```'
                    )
                parts.append(f"[Assistant]: {content or ''}\n" + "\n".join(tc_strs))
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            parts.append(f"[Tool result for {msg.get('name', '')}]: {content}")
        else:
            parts.append(content if content else "")

    prompt = "\n\n".join(p for p in parts if p)
    return prompt, images


def _repair_json(raw: str) -> str:
    """Fix common Gemini JSON issues: raw control chars and unescaped quotes
    inside strings (the model emits these when arguments are large, e.g. full
    file contents). Safe on valid JSON: it only changes invalid constructs.
    """
    out = []
    i = 0
    n = len(raw)
    in_str = False
    esc = False
    while i < n:
        c = raw[i]
        if in_str:
            if esc:
                out.append(c)
                esc = False
            elif c == "\\":
                out.append(c)
                esc = True
            elif c == '"':
                j = i + 1
                while j < n and raw[j] in ' \t\r\n':
                    j += 1
                nxt = raw[j] if j < n else ""
                if nxt in ',}]:':
                    out.append(c)
                    in_str = False
                else:
                    out.append('\\"')
            elif c == '\n':
                out.append('\\n')
            elif c == '\r':
                out.append('\\r')
            elif c == '\t':
                out.append('\\t')
            elif ord(c) < 32:
                out.append('\\u%04x' % ord(c))
            else:
                out.append(c)
        else:
            if c == '"':
                out.append(c)
                in_str = True
            elif c in '\r\n\t':
                out.append(c)
            elif ord(c) < 32:
                out.append('\\u%04x' % ord(c))
            else:
                out.append(c)
        i += 1
    return "".join(out)


def _scan_json_object(text: str, start: int):
    """String-aware brace matching. Returns (json_str, end_index) or (None, None)."""
    depth = 0
    in_str = False
    esc = False
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], i + 1
        i += 1
    return None, None


def parse_tool_calls(text: str) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list).

    Robust to how the model closes the block: ``` fences with 1-3 backticks,
    no fence at all, or a response cut off mid-JSON. Uses string-aware brace
    matching instead of a strict regex, so large arguments (e.g. full file
    contents with quotes/newlines) parse correctly.
    """
    tool_calls = []
    out = []
    pos = 0
    for m in re.finditer(r'```tool_call', text):
        out.append(text[pos:m.start()])
        rest = text[m.end():]
        stripped = len(rest) - len(rest.lstrip("\r\n "))
        body = rest[stripped:]
        if not body.startswith("{"):
            out.append(text[m.start():m.end()])
            pos = m.end()
            continue
        json_str, end = _scan_json_object(body, 0)
        if json_str is None:
            json_str = body
            end = len(body)
        try:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = json.loads(_repair_json(json_str))
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                },
            })
            after = body[end:]
            fence = len(after) - len(after.lstrip("`\r\n "))
            pos = m.end() + stripped + end + fence
        except (json.JSONDecodeError, KeyError, TypeError):
            # Don't silently drop unparseable blocks - keep the original text.
            out.append(text[m.start():m.end()])
            pos = m.end()
    out.append(text[pos:])
    return "".join(out).strip(), tool_calls


# ─── Google Native API helpers ─────────────────────────────────────────────────


def build_tool_prompt(tool_defs: list) -> str:
    """Build natural tool-use prompt for Gemini Web that avoids prompt-injection detection."""
    tool_spec = json.dumps(tool_defs, indent=2, ensure_ascii=False)
    return (
        "# Tool Use\n\n"
        "You can call the following tools to help accomplish tasks. "
        "These tools connect to the user's local environment and will execute when called.\n\n"
        "Call format (use this exact format):\n"
        "```function_call\n"
        '{"name": "<tool_name>", "args": {<arguments>}}\n'
        "```\n\n"
        "When calling tools:\n"
        "- Output ONLY the function_call block(s), nothing else\n"
        "- You may call multiple tools with multiple blocks\n"
        "- After receiving a [Tool result for ...], use that data to answer the user\n\n"
        f"Available tools:\n{tool_spec}"
    )


def _google_tool_choice_instruction(req: dict) -> str:
    """Extract tool_choice constraint from Google API toolConfig."""
    tool_config = req.get("toolConfig", {})
    fc_config = tool_config.get("functionCallingConfig", {})
    mode = fc_config.get("mode", "AUTO")
    allowed = fc_config.get("allowedFunctionNames", [])

    if mode == "NONE":
        return "\n\nIMPORTANT: Do NOT call any tools. Respond with text only."
    if mode == "ANY":
        if allowed:
            names = ", ".join(f'"{n}"' for n in allowed)
            return f"\n\nIMPORTANT: You MUST call one of these tools: {names}. Do not respond with text only."
        return "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
    return ""


def google_contents_to_prompt(req: dict) -> tuple:
    """Convert Google API contents/tools/systemInstruction to (prompt_str, images_list).

    Returns (prompt, images) where images is a list of (bytes, mime_type) tuples.
    """
    parts = []
    images = []

    tool_config = req.get("toolConfig", {})
    fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")

    tools = req.get("tools")
    tool_defs = []
    if tools and fc_mode != "NONE":
        for tool_group in tools:
            for fn in tool_group.get("functionDeclarations", []):
                td = {"name": fn.get("name", ""), "description": fn.get("description", "")}
                params = fn.get("parameters") or fn.get("parametersJsonSchema")
                if params:
                    td["parameters"] = params
                tool_defs.append(td)

    sys_inst = req.get("systemInstruction")
    if sys_inst:
        sys_parts = sys_inst.get("parts", [])
        sys_text = " ".join(p.get("text", "") for p in sys_parts if p.get("text"))
        if sys_text:
            if tool_defs:
                constraint = _google_tool_choice_instruction(req)
                parts.append(sys_text + "\n\n" + build_tool_prompt(tool_defs) + constraint)
            else:
                parts.append(sys_text)
    elif tool_defs:
        constraint = _google_tool_choice_instruction(req)
        parts.append(build_tool_prompt(tool_defs) + constraint)

    for content in req.get("contents", []):
        role = content.get("role", "user")
        msg_parts = []
        for p in content.get("parts", []):
            if p.get("text"):
                msg_parts.append(p["text"])
            elif p.get("inlineData"):
                data = p["inlineData"]
                try:
                    images.append((
                        base64.b64decode(data["data"], validate=True),
                        data.get("mimeType", "image/png"),
                    ))
                    msg_parts.append("[Image attached]")
                except (KeyError, ValueError, TypeError, binascii.Error):
                    pass
            elif p.get("functionCall"):
                fc = p["functionCall"]
                msg_parts.append(
                    f'```function_call\n{json.dumps({"name": fc["name"], "args": fc.get("args", {})}, ensure_ascii=False)}\n```'
                )
            elif p.get("functionResponse"):
                fr = p["functionResponse"]
                msg_parts.append(
                    f'[Tool result for {fr.get("name", "")}]: {json.dumps(fr.get("response", {}), ensure_ascii=False)}'
                )
        text = "\n".join(msg_parts)
        if role == "model":
            parts.append(f"[Assistant]: {text}")
        else:
            parts.append(text)

    return "\n\n".join(p for p in parts if p), images


def parse_google_function_calls(text: str) -> tuple:
    """Extract function_call blocks from model output.

    Handles 3 formats:
    1. ```function_call\\n{...}\\n``` (standard)
    2. function_call\\n{...} (without backticks)
    3. Raw JSON with "name" + "args" keys

    Returns (clean_text, [{"name": ..., "args": ...}])
    """
    function_calls = []
    pattern1 = r'```function_call\s*\n(.*?)\n```'
    pattern2 = r'(?:^|\n)function_call\s*\n(\{[^`]*?\})'
    clean = text
    for pattern in [pattern1, pattern2]:
        for match in re.findall(pattern, clean, re.DOTALL):
            try:
                data = json.loads(match.strip())
                if "name" in data:
                    function_calls.append({
                        "name": data["name"],
                        "args": data.get("args", data.get("arguments", {})),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        clean = re.sub(pattern, '', clean, flags=re.DOTALL).strip()
    if not function_calls and clean.strip().startswith("{"):
        try:
            data = json.loads(clean.strip())
            if "name" in data and ("args" in data or "arguments" in data):
                function_calls.append({
                    "name": data["name"],
                    "args": data.get("args", data.get("arguments", {})),
                })
                clean = ""
        except (json.JSONDecodeError, KeyError):
            pass
    return clean, function_calls


def parse_google_calls_as_tool_calls(text: str) -> tuple:
    """Parse Google function_call blocks and normalize to OpenAI tool_calls shape.

    Returns (clean_text, [{"id", "type": "function", "function": {"name", "arguments"}}]).
    Used so the shared validation/repair loop works for both API surfaces.
    """
    clean, calls = parse_google_function_calls(text)
    normalized = []
    for c in calls:
        normalized.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c.get("args", {}), ensure_ascii=False),
            },
        })
    return clean, normalized


# ─── Streaming tool-call parser ───────────────────────────────────────────────


class StreamToolCallParser:
    """Incremental parser that extracts fenced tool-call blocks from a
    streamed model response in real time.

    feed(chunk) returns events as soon as they are recognized:
      ("text", str)                    - plain text to stream to the client
      ("tool_start", index, name)      - a tool call opened, name known
      ("tool_args", index, args_json)  - complete arguments for that call
      ("tool_end", index)              - call finished

    finish() flushes the tail: buffered text, or a truncated JSON block that
    gets auto-closed and repaired before falling back to raw text.

    Handles both ```tool_call (OpenAI surface) and ```function_call (Google
    surface) openers, and is robust to openers/JSON split across chunks.
    """

    OPENERS = ("```tool_call", "```function_call")
    _TAIL = max(len(o) for o in OPENERS) - 1

    def __init__(self):
        self.index = 0
        self.pending = ""    # text-mode buffer (holds possible partial opener)
        self.body = None     # {"buf", "scanned", "depth", "in_str", "esc"} inside a block
        self.await_fence = False  # a block just closed: consume ``` before text

    # ── public API ────────────────────────────────────────────────────────

    def feed(self, chunk: str) -> list:
        events = []
        if self.body is not None:
            self.body["buf"] += chunk
            events.extend(self._drain_body())
            if self.body is None:
                events.extend(self._scan_text())
            return events
        self.pending += chunk
        events.extend(self._scan_text())
        return events

    def finish(self) -> list:
        events = []
        if self.body is not None:
            raw = self.body["buf"]
            depth, in_str = self.body["depth"], self.body["in_str"]
            self.body = None
            events.extend(self._emit_tool(raw, depth, in_str))
            self.pending = ""
            return events
        if self.await_fence:
            self._consume_fence()
            # A stream ending right at the closing fence must not leak the
            # leftover backticks into the content text.
            if self.await_fence and not self.pending.strip("`\r\n \t"):
                self.pending = ""
        if self.pending:
            events.append(("text", self.pending))
            self.pending = ""
        return events

    @property
    def tool_count(self) -> int:
        return self.index

    # ── internals ─────────────────────────────────────────────────────────

    def _consume_fence(self) -> bool:
        """Consume the closing ``` (possibly split across chunks) after a block.

        Returns True when text scanning may continue, False when more data is
        needed before the decision can be made."""
        stripped = self.pending.lstrip(" \t\r\n")
        self.pending = stripped
        if not stripped:
            return False
        if stripped.startswith("{"):
            # Another block body follows without an opener.
            self.await_fence = False
            return True
        if stripped.startswith("`"):
            count = len(stripped) - len(stripped.lstrip("`"))
            after_bt = stripped[count:]
            if after_bt.startswith(("tool_call", "function_call")):
                # Opening fence of another block: let the scanner handle it.
                self.await_fence = False
                return True
            if count >= 3:
                # Fence consumed; what follows is content text - keep it
                # verbatim (no further whitespace stripping).
                self.pending = stripped[3:]
                self.await_fence = False
                return True
            if after_bt:
                # 1-2 backticks followed by content: inline code, not a fence.
                self.await_fence = False
                return True
            return False  # partial fence: wait for more data
        self.await_fence = False
        return True

    def _scan_text(self) -> list:
        events = []
        while self.body is None:
            if self.await_fence:
                if self._consume_fence():
                    continue
                return events
            found = None
            for opener in self.OPENERS:
                i = self.pending.find(opener)
                if i != -1 and (found is None or i < found[0]):
                    found = (i, opener)
            if found is None:
                safe = len(self.pending) - self._TAIL
                if safe > 0:
                    events.append(("text", self.pending[:safe]))
                    self.pending = self.pending[safe:]
                return events
            i, opener = found
            events.append(("text", self.pending[:i]))
            rest = self.pending[i + len(opener):]
            stripped = rest.lstrip()
            if stripped == "":
                # Opener at a chunk boundary: wait for more data to decide.
                self.pending = self.pending[i:]
                return events
            if not stripped.startswith("{"):
                # Not a tool block: literal text.
                events.append(("text", opener))
                self.pending = rest
                continue
            self.body = {"buf": stripped, "scanned": 0, "depth": 0, "in_str": False, "esc": False}
            self.pending = ""
            events.extend(self._drain_body())
        return events

    def _drain_body(self) -> list:
        events = []
        b = self.body
        buf = b["buf"]
        i = b["scanned"]
        n = len(buf)
        while i < n:
            c = buf[i]
            if b["in_str"]:
                if b["esc"]:
                    b["esc"] = False
                elif c == "\\":
                    b["esc"] = True
                elif c == '"':
                    b["in_str"] = False
            elif c == '"':
                b["in_str"] = True
            elif c == "{":
                b["depth"] += 1
            elif c == "}":
                b["depth"] -= 1
                if b["depth"] == 0:
                    events.extend(self._emit_tool(buf[:i + 1]))
                    self.body = None
                    # Await the closing fence: it may arrive in a later chunk,
                    # so delegate its consumption to _scan_text/_consume_fence.
                    self.pending = buf[i + 1:]
                    self.await_fence = True
                    return events
            i += 1
        b["scanned"] = i
        return events

    def _emit_tool(self, json_str: str, depth: int = 0, in_str: bool = False) -> list:
        try:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = json.loads(_repair_json(json_str))
        except (json.JSONDecodeError, ValueError):
            if depth or in_str:
                # Truncated mid-JSON: auto-close using the live scanner state.
                closing = ('"' if in_str else "") + "}" * depth
                try:
                    data = json.loads(_repair_json(json_str + closing))
                except (json.JSONDecodeError, ValueError):
                    data = None
            else:
                data = None
        if not isinstance(data, dict) or "name" not in data:
            # Keep the raw block as text rather than dropping it silently.
            opener = "```tool_call"
            return [("text", opener + json_str + "```")]
        args = data.get("arguments", data.get("args", {}))
        args_json = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        idx = self.index
        self.index += 1
        return [("tool_start", idx, data["name"]), ("tool_args", idx, args_json), ("tool_end", idx)]


# ─── Validation + transparent repair loop ────────────────────────────────────


def generate_validated(
    prompt: str,
    tool_defs: list,
    tool_choice,
    generate_fn,
    *gen_args,
    parse_fn=None,
    block_format: str = "tool_call",
    **gen_kwargs,
) -> tuple:
    """Run generate_fn, parse tool calls and validate their arguments against
    the tool schemas.

    Invalid arguments trigger up to ``tool_validate_retry`` transparent repair
    round-trips (the validation errors are fed back to the model together with
    the original prompt). Returns (clean_text, tool_calls) in OpenAI shape.
    """
    if not tool_defs or tool_choice == "none":
        return (generate_fn(prompt, *gen_args, **gen_kwargs) or "", [])
    parse_fn = parse_fn or parse_tool_calls

    def _round(p):
        text = generate_fn(p, *gen_args, **gen_kwargs)
        if not text:
            return "", []
        return parse_fn(text)

    clean, calls = _round(prompt)
    retries = int(CONFIG.get("tool_validate_retry", 1))
    by_name = {d.get("name", ""): d for d in tool_defs}
    attempt = 0
    while calls and attempt < retries:
        invalid = []
        valid = []
        for call in calls:
            fn = call.get("function", {})
            schema = (by_name.get(fn.get("name", ""), {}) or {}).get("parameters") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                invalid.append((call, f"{fn.get('name')}: arguments are not valid JSON"))
                continue
            errs = validate_tool_arguments(args, schema)
            if errs:
                invalid.append((call, f"{fn.get('name')}: " + "; ".join(errs)))
            else:
                valid.append(call)
        if not invalid:
            break
        attempt += 1
        from .gemini import log
        log(f"Invalid tool arguments (repair round {attempt}): "
            + "; ".join(msg for _, msg in invalid))
        repair_prompt = (
            f"{prompt}\n\n"
            f"[System correction]: Your previous answer contained {block_format} blocks "
            "with invalid arguments:\n"
            + "".join(f"- {msg}\n" for _, msg in invalid)
            + f"\nPrevious answer:\n{clean or ''}\n\n"
            f"Re-output corrected {block_format} blocks ONLY, with arguments that satisfy "
            "the tool schemas. If no tool call is needed, answer in plain text."
        )
        clean, calls = _round(repair_prompt)
    # Keep only schema-valid calls; never hand broken arguments to the client.
    final = []
    for call in calls:
        fn = call.get("function", {})
        schema = (by_name.get(fn.get("name", ""), {}) or {}).get("parameters") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = None
        if args is None or validate_tool_arguments(args, schema):
            from .gemini import log
            log(f"Dropping invalid tool call: {fn.get('name')}")
            continue
        final.append(call)
    return clean, final
