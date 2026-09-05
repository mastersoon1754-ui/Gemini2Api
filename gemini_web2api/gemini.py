"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import socket
import urllib.request
import urllib.parse
import ssl
import os
import hashlib

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_httpx_client = None


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None and HAS_HTTPX:
        proxy = CONFIG.get("proxy")
        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        # Streaming responses are never closed by Gemini (no terminating chunk),
        # so cap the per-read idle time to detect stream completion instead of
        # hanging until request_timeout_sec.
        if hasattr(httpx, "Timeout"):
            timeout = httpx.Timeout(CONFIG["request_timeout_sec"], read=STALL_TIMEOUT)
        else:
            timeout = CONFIG["request_timeout_sec"]
        _httpx_client = httpx.Client(transport=transport, timeout=timeout, verify=True)
    return _httpx_client


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "", None
    try:
        mtime = os.path.getmtime(cookie_file)
        if mtime == _cookie_cache["mtime"] and _cookie_cache["str"]:
            return _cookie_cache["str"], _cookie_cache["sapisid"]
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return _cookie_cache["str"], _cookie_cache["sapisid"]


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


# Gemini keeps the StreamGenerate chunked response open indefinitely (no
# terminating `0\r\n\r\n` chunk), so any read-to-EOF blocks forever. Data is
# always delivered in a short burst; a brief idle on the socket means we have
# received everything. This is the max time we wait for the next byte before
# treating the response as complete. Gemini pauses up to ~60s between bursts
# while composing long tool calls / files, so the idle timeout must be generous.
STALL_TIMEOUT = 60


def _is_terminal_line(line: str) -> bool:
    """True when a wrb.fr line is the final response metadata (title + done flag)."""
    if '"wrb.fr"' not in line or len(line) < 60:
        return False
    try:
        arr = json.loads(line)
        inner = json.loads(arr[0][2])
        return (isinstance(inner, list) and len(inner) > 2
                and isinstance(inner[2], dict)
                and "11" in inner[2] and "44" in inner[2])
    except Exception:
        return False


def _has_terminal_chunk(buf: bytes) -> bool:
    """Scan a byte buffer for the terminal response-metadata line."""
    try:
        for line in buf.split(b"\n"):
            if _is_terminal_line(line.decode("utf-8", errors="replace")):
                return True
    except Exception:
        pass
    return False


def fetch_latest_bl() -> str:
    """Fetch the latest gemini_bl build token from the Gemini web app page."""
    try:
        req = urllib.request.Request(
            "https://gemini.google.com/app",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        ctx = _get_ssl_ctx()
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx),
            )
            resp = opener.open(req, timeout=15)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
        m = re.search(r'(boq_assistant-bard-web-server_\d+\.\d+_p\d+)', html)
        if m:
            return m.group(1)
    except Exception as e:
        log(f"BL fetch failed: {e}")
    return ""


def update_bl_if_needed() -> bool:
    """Try to fetch and adopt the latest gemini_bl. Returns True if updated."""
    new_bl = fetch_latest_bl()
    if new_bl and new_bl != CONFIG["gemini_bl"]:
        log(f"BL updated: {CONFIG['gemini_bl']} -> {new_bl}")
        CONFIG["gemini_bl"] = new_bl
        return True
    return False


def _build_headers() -> dict:
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    return headers


def _apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        # Match Gemini Web temporary-chat requests.
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    return urllib.parse.urlencode(params)


def _get_url() -> str:
    reqid = int(time.time()) % 1000000
    account_prefix = _account_prefix()
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


def _extract_texts_from_line(line: str) -> list:
    """Parse a single wrb.fr line and return list of text strings found."""
    if '"wrb.fr"' not in line or len(line) < 200:
        return []
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str or len(inner_str) < 50:
            return []
        inner = json.loads(inner_str)
        if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
            return []
        texts = []
        for part in inner[4]:
            if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                for t in part[1]:
                    if isinstance(t, str) and t:
                        texts.append(t)
        return texts
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text."""
    bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', raw)
    if bard_err:
        raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]")
    last_text = ""
    for line in raw.split("\n"):
        for t in _extract_texts_from_line(line):
            if len(t) > len(last_text):
                last_text = t
    return clean_text(last_text)


def _read_all(resp) -> bytes:
    """Read a response body, tolerating a connection that never closes.

    Gemini streams the answer in bursts with pauses up to ~60s (it composes
    long tool calls / files between bursts) and leaves the chunked connection
    open (no terminating chunk). We stop as soon as the final response-metadata
    line arrives (title + done flag), falling back to a socket idle exceeding
    STALL_TIMEOUT as a safety net. Falls back to read() (EOF based) otherwise.
    """
    if hasattr(resp, "read1"):
        sock = None
        fp = getattr(resp, "fp", None)
        raw = getattr(fp, "raw", None) if fp is not None else None
        if raw is not None and hasattr(raw, "_sock"):
            sock = raw._sock
        if sock is not None:
            prev_timeout = sock.gettimeout()
            sock.settimeout(STALL_TIMEOUT)
            try:
                buf = bytearray()
                while True:
                    try:
                        chunk = resp.read1(65536)
                    except (socket.timeout, TimeoutError):
                        break
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if _has_terminal_chunk(bytes(buf)):
                        break
                return bytes(buf)
            finally:
                sock.settimeout(prev_timeout)
    return resp.read()


def _urlopen(req, ctx=None, proxy=None):
    ctx = ctx or _get_ssl_ctx()
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=ctx),
        )
        return opener.open(req, timeout=CONFIG["request_timeout_sec"])
    return urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])


def _should_retry_bl(e) -> bool:
    return getattr(e, "code", None) == 405


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    """Non-streaming generation with retry and BL refresh."""
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    for attempt in range(CONFIG["retry_attempts"] + 1):
        url = _get_url()
        headers = _build_headers()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            resp = _urlopen(req, ctx, proxy)
            raw = _read_all(resp).decode("utf-8", errors="replace")
            return extract_response_text(raw)
        except Exception as e:
            last_err = e
            if _should_retry_bl(e) and update_bl_if_needed():
                log("Retrying with updated BL...")
                continue
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None):
    """Streaming generation via httpx with retry on connection failure."""
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        if text:
            yield text
        return

    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)

    last_err = None
    emitted_raw_text = ""
    for attempt in range(CONFIG["retry_attempts"] + 1):
        url = _get_url()
        headers = _build_headers()
        client = _get_httpx_client()
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                try:
                    iter_chunks = resp.iter_text()
                    while True:
                        try:
                            chunk = next(iter_chunks)
                        except StopIteration:
                            break
                        buf += chunk
                        if "BardErrorInfo" in buf:
                            bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                            if bard_err:
                                raise RuntimeError(
                                    f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]"
                                )
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            if _is_terminal_line(line):
                                return
                            for t in _extract_texts_from_line(line):
                                if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                    continue
                                if not t.startswith(emitted_raw_text):
                                    raise RuntimeError("Gemini stream content changed during retry")
                                delta = clean_text(t[len(emitted_raw_text):], strip=False)
                                emitted_raw_text = t
                                if delta:
                                    yield delta
                except httpx.ReadTimeout as _e:
                    # Upstream keeps the connection open; treat an idle stream
                    # as end-of-response rather than an error.
                    pass
                return
        except httpx.HTTPStatusError as e:
            last_err = e.response
            if e.response.status_code == 405 and update_bl_if_needed():
                log("Retrying stream with updated BL...")
                continue
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Stream retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
            raise
        except Exception as e:
            if isinstance(e, (httpx.ReadTimeout, ImportError)):
                raise
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Stream retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
