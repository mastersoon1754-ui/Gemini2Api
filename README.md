# gemini-web2api

<p align="center">
  <img src="logo.png" width="200" alt="gemini-web2api logo">
</p>

[中文文档](README_CN.md)

Convert Google Gemini's web interface into an OpenAI-compatible API. Zero cost, cross-platform, single file.

## Features

- **Optional API Keys**: no auth when `api_keys` is empty, OpenAI-style Bearer auth when configured
- **OpenAI Compatible**: Drop-in replacement for `/v1/chat/completions` and `/v1/models`
- **Tool Calling**: Full function calling support (OpenAI format)
- **Multiple Models**: Flash (3.6), Extended Thinking (20k+ char output), Pro, Auto, Lite
- **Thinking Depth**: Adjustable via `@think=N` suffix (0=deepest, 4=shallowest)
- **Web Search**: Built-in internet access (Gemini's native search)
- **Cross-Platform**: Pure Python, single optional dependency (`httpx` for streaming)
- **Streaming**: SSE streaming support via `httpx`
- **Codex CLI**: Responses API (`/v1/responses`) for OpenAI Codex integration
- **Gemini CLI**: Google native API (`/v1beta/models`) for Gemini CLI compatibility
- **Anthropic clients**: Messages API (`/v1/messages`) for Claude-style SDKs

## Quick Start

```bash
pip install httpx
python gemini_web2api.py
```

Server starts at `http://localhost:8081/v1`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` , `/health` | Health check (no auth, <1KB, for keepalive) |
| `GET` | `/v1/models` | List models (OpenAI) |
| `POST` | `/v1/chat/completions` | Chat completions, `stream` SSE, `tools`, `tool_choice`, `tool_executor_url` (agent) |
| `POST` | `/v1/responses` | Responses API (Codex) — same, `stream` events `response.*` |
| `POST` | `/v1/messages` | Anthropic-compatible — `tools` with `input_schema`, `tool_choice` |
| `GET` | `/v1beta/models` | List models (Google) |
| `POST` | `/v1beta/models/{model}:generateContent` | Google non-streaming |
| `POST` | `/v1beta/models/{model}:streamGenerateContent` | Google streaming SSE |

All `/v1/*` respect `api_keys` (`Authorization: Bearer` or `x-api-key` or `?key=`). `/health` is always open.

## Client Configuration

### Cherry Studio / ChatBox / any OpenAI client

| Field | Value |
|-------|-------|
| Base URL | `http://localhost:8081/v1` |
| API Key | any `api_keys` value from `config.json`; anything if not configured |
| Model | `gemini-3.5-flash-thinking` |

### curl

#### bash / macOS / Linux

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello!"}]}'
```

#### PowerShell (Windows)

```powershell
curl.exe --% http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer sk-your-key" -d "{\"model\":\"gemini-3.5-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

> Note: On Windows PowerShell, use `curl.exe` and `--%` so PowerShell does not reinterpret JSON quoting or curl options.

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")
resp = client.chat.completions.create(
    model="gemini-3.5-flash-thinking",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(resp.choices[0].message.content)

# Streaming (SSE compatible, tool_calls streamed as deltas)
stream = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Gemini CLI

```bash
export GEMINI_API_KEY=none
export GOOGLE_GEMINI_BASE_URL=http://localhost:8081
gemini
```

Supports Google native API endpoints:
- `GET /v1beta/models` — list models
- `POST /v1beta/models/{model}:generateContent` — non-streaming
- `POST /v1beta/models/{model}:streamGenerateContent` — streaming (SSE)

## Available Models

| Model | Description | Output |
|-------|-------------|--------|
| `gemini-3.6-flash` | All-around model (latest) | ~12k chars |
| `gemini-3.5-flash` | Alias for gemini-3.6-flash | ~12k chars |
| `gemini-3.5-flash-thinking` | Extended thinking, longest output | **~20k chars** |
| `gemini-3.5-flash-thinking-lite` | Adaptive thinking depth | ~15k chars |
| `gemini-3.1-pro` | Advanced math & code (needs cookie) | ~12k chars |
| `gemini-auto` | Auto model selection | varies |
| `gemini-flash-lite` | Fastest answers, lightweight | ~10k chars |

### Thinking Depth

Append `@think=N` to any model name:

```
gemini-3.5-flash-thinking@think=0   # deepest (default)
gemini-3.5-flash-thinking@think=2   # medium
gemini-3.5-flash-thinking@think=4   # shallowest
```

## Optional: Cookie for Pro

Anonymous access works for all models, but `gemini-3.1-pro` routes to Flash without authentication. To get real Pro routing, you need a **Gemini Advanced (paid subscription)** account cookie:

```bash
python gemini_web2api.py --cookie-file cookie.txt
```

### How to get cookies

1. Open Chrome, go to [gemini.google.com](https://gemini.google.com) and sign in with a **Gemini Advanced** Google account
2. Open DevTools (F12) → Application → Cookies → `https://gemini.google.com`
3. Copy these cookie values: `SID`, `HSID`, `SSID`, `APISID`, `SAPISID`, `__Secure-1PSID`
4. Create `cookie.txt` in this format:

```
SID=your_sid_value; HSID=your_hsid_value; SSID=your_ssid_value; APISID=your_apisid_value; SAPISID=your_sapisid_value; __Secure-1PSID=your_1psid_value
```

Or use the JSON format:
```json
{"cookie": "SID=xxx; HSID=xxx; SSID=xxx; APISID=xxx; SAPISID=xxx; __Secure-1PSID=xxx", "sapisid": "your_sapisid_value"}
```

**Alternative (browser extension)**: Use any "Export Cookies" extension to export cookies for `gemini.google.com` in Netscape format, then convert to the single-line format above.

### Authenticated account path and XSRF token

If the signed-in Gemini page URL contains an account index, such as:

```
https://gemini.google.com/u/1/app/...
```

set `auth_user` to that index. Authenticated web requests may also require the page XSRF token. In the rendered Gemini page source, this token is exposed as `SNlM0e`; pass it as `xsrf_token` in `config.json`. The server sends it as the `at` form field.

Example:

```json
{
  "cookie_file": "/app/cookie.txt",
  "auth_user": "1",
  "xsrf_token": "AOOh0P...",
  "gemini_bl": "boq_assistant-bard-web-server_YYYYMMDD.xx_p0"
}
```

If authenticated requests return HTTP 400 with an `xsrf` error, refresh Gemini Web, update `xsrf_token`, and make sure `auth_user` matches the `/u/<index>/` part of the browser URL.

Pro routing requires **Gemini Advanced** (paid subscription). A free Google account cookie will authenticate but silently fall back to Flash.

## Configuration

Create `config.json` in the same directory:

```json
{
  "port": 8081,
  "host": "0.0.0.0",
  "retry_attempts": 3,
  "retry_delay_sec": 2,
  "request_timeout_sec": 180,
  "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
  "auth_user": null,
  "xsrf_token": null,
  "api_keys": ["sk-your-key"],
  "cookie_file": null,
  "proxy": null,
  "log_requests": true,
  "temporary_chats": false
}
```

Set `temporary_chats` to `true` to use Gemini Web temporary chats instead of
persisting conversations to the account history.

When `api_keys` is `[]`, authentication is disabled. When one or more keys are set, `/v1/*` endpoints require `Authorization: Bearer <key>` or `x-api-key: <key>`.

## Docker

```bash
cp config.example.json config.json
docker build -t gemini-web2api .
docker run -d --name gemini-web2api -p 8081:8081 -v ./config.json:/app/config.json gemini-web2api
```

Or use Docker Compose:

```bash
cp config.example.json config.json
docker compose up -d
```

To mount a cookie file:

```bash
docker run -d --name gemini-web2api -p 8081:8081 -v ./config.json:/app/config.json -v ./cookie.txt:/app/cookie.txt gemini-web2api
```

Set `"cookie_file": "/app/cookie.txt"` in `config.json`.

> **Note**: If you get empty responses (`content: null`) with Docker's default bridge network, switch to host networking: `docker run --network host ...` or add `network_mode: host` in your compose file. This is caused by Gemini's upstream rejecting requests from certain Docker NAT IP ranges.

## Proxy

If you cannot access `gemini.google.com` directly (connection timeout), configure a proxy:

**Method 1: CLI argument**
```bash
python gemini_web2api.py --proxy http://127.0.0.1:7890
```

**Method 2: config.json**
```json
{"proxy": "http://127.0.0.1:7890"}
```

**Method 3: Environment variable** (auto-detected)
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
python gemini_web2api.py
```

Works with Clash, V2Ray, Shadowsocks, or any HTTP proxy.

## Tool Calling

```python
resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }]
)
```

## Agent Mode (server-side tool loop)

Want a **real agent** — one API call that keeps calling tools until it
finishes? Add `tool_executor_url` to a normal chat or responses request. The
server then runs the whole loop itself:

1. Model decides to call a tool
2. Server POSTs the call to your webhook: `{"call_id", "name", "arguments"}`
3. Your webhook executes the tool and returns `{"result": ...}` (or plain text)
4. Result is fed back to the model; repeat until it stops calling tools
5. Server returns the **final answer** — like any normal API call

The response includes an `agent_tool_calls` array listing every executed step
(name, arguments, result) so you can see what the agent did.

```python
resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo and should I bring an umbrella?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }],
    # The proxy calls this URL to run tools; your tool implementations live here.
    tool_executor_url="http://localhost:3000/exec",
)
print(resp.choices[0].message.content)  # final answer, loop already handled
```

Your webhook receives:

```json
{"call_id": "call_xxx", "name": "get_weather", "arguments": {"city": "Tokyo"}}
```

and responds with the result under `result`, `output`, `content`, or `response`
(any JSON value is fine, or a plain-text body).

Configuration:
- `max_agent_turns` (default `8`) — loop safety cap
- `agent_tool_timeout_sec` (default `30`) — per-tool-call webhook timeout

`tool_choice: "required"` is enforced: if the model refuses to call a tool,
the loop nudges it once before returning.

## Tool Calling Reliability

Gemini Web has no native function calling, so the proxy translates tool
definitions into the prompt and parses tool-call blocks from the model output.
To make this behave like a real API for coding agents:

- **Real-time streaming**: `tool_calls` deltas are emitted as soon as each
  block is recognized (no buffering of the full response), with OpenAI-style
  `index`/`id`/`arguments` chunks and `finish_reason: "tool_calls"`.
- **Parallel tool calls**: multiple blocks in one response become multiple
  `tool_calls` entries.
- **Large toolsets**: up to `tool_max_tools` (default `96`) tools and
  `tool_max_prompt_chars` (default `120000`) of schema JSON per prompt;
  schemas are minified first. Core coding tools (bash/edit/read/...) are
  prioritized if the budget is hit, and drops are logged.
- **Validation + auto-repair**: parsed arguments are validated against each
  tool's JSON schema. Invalid arguments trigger up to `tool_validate_retry`
  (default `1`) transparent repair round-trips with the original context;
  calls that stay broken are never sent to the client.

## OpenCode (coding agent with its own tools)

OpenCode is itself the agent — it runs its own tools (bash, file edits, etc.)
and handles the tool loop client-side. You just point it at the proxy. No
webhook needed.

1. Start the server:
   ```bash
   python gemini_web2api.py
   ```
2. Copy `opencode.example.json` to `opencode.json` in your project root
   (or merge it into `~/.config/opencode/opencode.json` for global use):
   ```bash
   cp opencode.example.json opencode.json
   ```
   If you set `api_keys` in your proxy config, make the `apiKey` here match.
3. Run OpenCode and select the model:
   ```bash
   opencode
   /models   # pick e.g. gemini-web2api/gemini-3.6-flash
   ```

The example config includes **all 9 models** the proxy serves (Flash 3.7/3.6/3.5,
Thinking, Thinking Lite, Pro, Pro Enhanced, Auto, Flash Lite), sets
`gemini-3.6-flash` as the default, and sets `small_model` to
`gemini-flash-lite` so OpenCode's session-title tasks also run through the
proxy instead of falling back to an external provider.

OpenCode's built-in tools (bash, edit, grep, etc.) work through the proxy's
function-calling support, including multi-turn tool result roundtrips.

## Image Input

OpenAI-style multimodal messages are supported for Chat Completions and the
Responses API. Use either HTTP(S) image URLs or base64 data URLs:

```python
resp = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
    }]
)
```

## Limitations

- **Image upload may require cookies**: Multimodal input uses Gemini Web's image upload endpoint. If anonymous upload fails, configure a Gemini cookie.
- **Not real Pro/Ultra**: Without a paid subscription cookie, `gemini-3.1-pro` routes to the same Flash model. The "Pro" label is a UI preference, not a backend model switch.
- **Single-turn only**: Each request is an independent conversation. Multi-turn context is simulated by including previous messages in the prompt.
- **Rate limits**: Google may throttle high-frequency requests. The server retries automatically but sustained heavy use may be blocked.

## Requirements

- Python 3.8+
- `httpx` (`pip install httpx`) — used for streaming requests
- Network access to `gemini.google.com` (proxy/VPN may be needed in some regions)

## Architecture

```
Client (OpenAI SDK / Codex / Gemini CLI / Anthropic)
        ↓  POST /v1/chat/completions | /v1/responses | /v1/messages | /v1beta/*
API Layer (gemini_web2api/server.py) — OpenAI / Google / Anthropic translation, SSE, auth, validation
        ↓  prompt + tool_defs + images
Orchestrator (gemini_web2api/orchestrator.py + agent.py) — prompt building, tool budget, validation, agent loop
        ↓  prompt string + file_refs
Gemini Web Backend (gemini_web2api/gemini.py) — f.req / bl / SAPISIDHASH / wrb.fr parsing / streaming
        ↓  POST https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate
Gemini Web
        ↓  wrb.fr chunks
Tool System (gemini_web2api/tools.py) — ```tool_call``` blocks, StreamToolCallParser, JSON Schema validation, repair loop
        ↓  tool_calls → Tool Executor (webhook or client) → [Tool result for ...] → next turn
```

- **API Layer** ne connaît pas le détail Gemini (`bl`, `inner[79]`), seulement la traduction.
- **Backend** ne connaît pas OpenAI (`tool_calls`, `finish_reason`), seulement `prompt` et `file_refs`.
- **Tool System** est un fallback texte (` ```tool_call` ) avec abstraction `parse_fn`/`block_format` pour remplacer par un futur protocole natif si découvert (voir `docs/GEMINI_WEB_PROTOCOL.md`).

## How It Works

This project **does not use the official Google Gemini API** (`generativelanguage.googleapis.com`). It provides its **own OpenAI-compatible API** (`/v1/*`) and uses **Gemini Web as a backend**.

1. Your app calls `POST /v1/chat/completions` with OpenAI format (messages, tools, stream).
2. The server (`server.py`) translates `messages` → a single prompt string (with `[System instruction]:`, `[Tool result for ...]:`) and `tools` → a JSON block injected in the prompt (`Available tools: [...]`). See `tools.py:207`.
3. Images are uploaded via Scotty `content-push.googleapis.com` (`multimodal.py:88`) → `file_ref`.
4. The backend (`gemini.py:195`) builds `inner[79]=MODE_CATEGORY` (1 FAST, 2 THINKING, 3 PRO, 4 AUTO... from `028-*.js`), `inner[17]=think`, `inner[41]=temporary_chats`, and POSTs `f.req` to `StreamGenerate?bl={bl}&_reqid={rand}&rt=c` with `SAPISIDHASH` if a cookie exists.
5. Gemini Web streams `wrb.fr` lines (`)]}'` + `[[null,"wrb.fr",null,"<inner_json>"]]`). The proxy parses `inner[4]` for cumulative text and `inner[2]{"11","44"}` for end-of-generation, with a 60s idle fallback (`STALL_TIMEOUT`).
6. If the model emitted ` ```tool_call\n{"name":...}\n``` ` blocks, `StreamToolCallParser` extracts them in real time, validates against JSON Schema, and (optionally) repairs via `generate_validated`. In agent mode (`tool_executor_url`), `agent.py` executes them (parallel) and loops.
7. The API layer re-formats the result as OpenAI SSE (`choices[].delta.tool_calls`, `finish_reason: tool_calls`) / Google `functionCall` / Anthropic `tool_use`.

Model selection is field `[79]` in the request payload, mapped from Gemini's frontend JS (`MODE_CATEGORY` enum). All tool calling is currently **simulated via prompt injection**, not a native Gemini Web field — see `REVERSE_ENGINEERING.md:14` and `docs/GEMINI_WEB_PROTOCOL.md:5` for levels `CONFIRMED`/`PROBABLE`/`UNKNOWN`.

### Keepalive (Render free tier)

Render sleeps free services after ~15 min without inbound traffic. The server self-pings `GET /health` every `keepalive_interval_sec` (default 600s) via `RENDER_EXTERNAL_URL` or `KEEPALIVE_URL` (`keepalive.py:1` + `__main__.py:36`). Set `KEEPALIVE_URL=https://your-app.onrender.com` or disable with `KEEPALIVE_INTERVAL_SEC=0`. Health endpoint is `GET /` and `GET /health` (no auth, <1KB).

## Acknowledgments

- Inspired by the open-source API proxy ecosystem

## License

MIT

---

## 致谢

本项目的开发 agent 能力由 [GenericAgent](https://github.com/lsdefine/GenericAgent) 提供。

### 🚩 友情链接

[![GenericAgent](https://img.shields.io/badge/Agent_Framework-GenericAgent-orange?style=for-the-badge&logo=github)](https://github.com/lsdefine/GenericAgent)
[![LinuxDo](https://img.shields.io/badge/社区-LinuxDo-blue?style=for-the-badge)](https://linux.do/)
