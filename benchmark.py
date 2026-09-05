#!/usr/bin/env python3
"""Benchmark the proxy's Gemini models: latency + estimated tokens/s.

Measures each model through the /v1/chat/completions streaming endpoint and
prints an ASCII table. With --update-md, rewrites MODELS.md with the results.

Usage:
    python benchmark.py                          # all models, 1 run each, own proxy
    python benchmark.py --runs 3                 # 3 runs per model (averaged)
    python benchmark.py --models gemini-3.6-flash,gemini-flash-lite
    python benchmark.py --url http://127.0.0.1:8081/v1   # benchmark a running proxy
    python benchmark.py --update-md              # regenerate MODELS.md with results
    python benchmark.py --prompt "My custom prompt"

Notes:
    - Tokens are estimated as chars / 4 (same convention as the proxy's usage).
    - Measurements use streaming requests and include proxy overhead.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from collections import defaultdict

DEFAULT_PROMPT = (
    "Write a detailed three-paragraph explanation of how the internet works, "
    "covering DNS, TCP/IP, HTTP and routing. Be thorough."
)


# Per-model description, best-for, and notes (kept in sync with MODELS.md).
MODEL_DETAILS = {
    "gemini-3.7-flash": {
        "what": "the newest all-around Flash model.",
        "best": "everyday agent work, coding, quick answers with good quality.",
        "notes": "the current top pick if you want the latest backend.",
    },
    "gemini-3.6-flash": {
        "what": "the all-around Flash model (previous default).",
        "best": "the safe default — fast, capable, handles tools and agents well.",
        "notes": "fastest first-byte of the flash family; good balance.",
    },
    "gemini-3.5-flash": {
        "what": "alias that routes to the gemini-3.6-flash backend.",
        "best": "compatibility with old configs/scripts that hardcode the name.",
        "notes": "same backend as 3.6, so prefer gemini-3.6-flash for new configs.",
    },
    "gemini-3.5-flash-thinking": {
        "what": "deep thinking mode; longest output of the family (~20k chars).",
        "best": "hard problems — debugging, architecture, math, long codegen.",
        "notes": "pays the extra wait for quality; ideal for tough agent tasks.",
    },
    "gemini-3.5-flash-thinking-lite": {
        "what": "adaptive thinking depth.",
        "best": "a middle ground — smarter than plain flash, faster than full thinking.",
        "notes": "good 'thinking without the full wait' option.",
    },
    "gemini-3.1-pro": {
        "what": "Pro-tier model.",
        "best": "advanced math & code — requires a Gemini Advanced cookie for real Pro routing.",
        "notes": "without a cookie it silently falls back to Flash (see README).",
    },
    "gemini-3.1-pro-enhanced": {
        "what": "Pro with an experimental output-enhancement flag.",
        "best": "Pro users who want maximum output quality over speed.",
        "notes": "also needs the Pro cookie; the boost is experimental.",
    },
    "gemini-auto": {
        "what": "lets Gemini pick the model per request.",
        "best": "when you don't want to choose — trade-off handled upstream.",
        "notes": "quality/price decision is made by Gemini.",
    },
    "gemini-flash-lite": {
        "what": "lightweight model.",
        "best": "high-volume simple Q&A, titles, summarization, cheap/fast loops.",
        "notes": "the recommended small_model in OpenCode configs.",
    },
}

PRACTICAL_PICKS = [
    ("Fast default for everything", "gemini-3.6-flash"),
    ("Hard coding / reasoning", "gemini-3.5-flash-thinking"),
    ("Smarter than flash, faster than thinking", "gemini-3.5-flash-thinking-lite"),
    ("Cheapest bulk Q&A", "gemini-flash-lite"),
    ("Best possible quality (with cookie)", "gemini-3.1-pro-enhanced"),
    ("OpenCode small_model (titles etc.)", "gemini-flash-lite"),
]


def start_local_proxy():
    """Start an in-process proxy on a random port. Returns (server, base_url)."""
    from gemini_web2api.config import CONFIG
    import gemini_web2api.server as srv

    CONFIG["api_keys"] = []
    CONFIG["log_requests"] = False
    server = srv.ThreadedServer(("127.0.0.1", 0), srv.GeminiHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def list_models():
    """Model names in the order the server reports them (own-proxy mode)."""
    import urllib.request as _urllib

    # Preferred: import the package's MODELS if available.
    try:
        from gemini_web2api.models import MODELS
        return list(MODELS.keys())
    except ImportError:
        pass
    return list(MODEL_DETAILS.keys())


def bench_once(base_url, model, prompt, timeout):
    """One streaming request. Returns a metrics dict or raises."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=timeout)
    t_first = None
    content = ""
    chunks = 0
    for raw_line in resp:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
        except Exception:
            delta = ""
        if delta:
            if t_first is None:
                t_first = time.time()
            content += delta
            chunks += 1
    t_done = time.time()

    elapsed = t_done - t0
    ttfb = (t_first - t0) if t_first else elapsed
    est_tokens = max(len(content) // 4, 0)
    stream_secs = max(elapsed - ttfb, 0.1)
    return {
        "model": model,
        "ttfb": ttfb,
        "elapsed": elapsed,
        "chars": len(content),
        "est_tokens": est_tokens,
        "stream_tps": est_tokens / stream_secs,
        "overall_tps": est_tokens / elapsed if elapsed > 0 else 0.0,
        "chunks": chunks,
    }


def aggregate(models, base_url, prompt, runs, timeout):
    """Run each model `runs` times. Returns {model: [metrics]}."""
    results = defaultdict(list)
    for model in models:
        sys.stderr.write(f"Benchmarking {model} ({runs} run{'s' if runs != 1 else ''})...\n")
        for i in range(runs):
            sys.stderr.write(f"  run {i + 1}/{runs}\n")
            try:
                results[model].append(bench_once(base_url, model, prompt, timeout))
            except Exception as e:
                sys.stderr.write(f"  ERROR: {type(e).__name__}: {e}\n")
                results[model].append(None)
    return results


def avg(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else float("nan")


def summarize(results):
    """Flatten to one row per model (averages over successful runs)."""
    rows = []
    for model, runs in results.items():
        ok = [r for r in runs if r is not None]
        if not ok:
            rows.append({"model": model, "error": True})
            continue
        rows.append({
            "model": model,
            "error": False,
            "ttfb": avg([r["ttfb"] for r in ok]),
            "elapsed": avg([r["elapsed"] for r in ok]),
            "chars": avg([r["chars"] for r in ok]),
            "est_tokens": avg([r["est_tokens"] for r in ok]),
            "stream_tps": avg([r["stream_tps"] for r in ok]),
            "overall_tps": avg([r["overall_tps"] for r in ok]),
        })
    return rows


def print_table(rows):
    header = f"{'Model':40s} {'First byte':>10s} {'Total':>7s} {'Chars':>6s} {'Tok~':>6s} {'Tok/s strm':>11s} {'Tok/s all':>10s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["error"]:
            print(f"{row['model']:40s} {'ERROR':>10s}")
            continue
        print(
            f"{row['model']:40s} {row['ttfb']:8.2f}s {row['elapsed']:6.1f}s "
            f"{int(row['chars']):6d} {int(row['est_tokens']):6d} "
            f"{row['stream_tps']:9.1f} {row['overall_tps']:9.1f}"
        )


def round_or_dash(value, digits=1):
    if value is None or value != value:  # None or NaN
        return "-"
    return f"{value:.{digits}f}"


def generate_markdown(rows, prompt, runs, base_url):
    """Build the full MODELS.md content from measured averages."""
    lines = []
    lines.append("# Gemini Models — Reference & Benchmarks\n")
    lines.append(
        "All 9 models served by this proxy, what each is best for, and measured\n"
        "performance **from this proxy** (live benchmark, streaming request, anonymous mode).\n"
    )
    lines.append("\n> Generated by `benchmark.py` — re-run `python benchmark.py --update-md` to refresh.\n")
    lines.append("## Quick summary\n")
    lines.append(
        "| Model | What it is | Best for | First byte | Total time* | Tokens/s (stream) | Tokens/s (overall) |\n"
        "|---|---|---|---|---|---|---|"
    )
    for row in rows:
        if row["error"]:
            lines.append(f"| `{row['model']}` | — | — | ERROR | — | — | — |")
            continue
        d = MODEL_DETAILS.get(row["model"], {})
        lines.append(
            f"| `{row['model']}` | {d.get('what', '—')} | {d.get('best', '—')} "
            f"| ~{round_or_dash(row['ttfb'])} s | ~{round_or_dash(row['elapsed'])} s "
            f"| ~{round_or_dash(row['stream_tps'])} | ~{round_or_dash(row['overall_tps'])} |"
        )
    lines.append(
        "\n\\* Total time for a ~"
        f"{int(avg([r['chars'] for r in rows if not r['error']]) if any(not r['error'] for r in rows) else 0)}"
        " char (~"
        f"{int(avg([r['est_tokens'] for r in rows if not r['error']]) if any(not r['error'] for r in rows) else 0)}"
        " token) answer. Your numbers will vary with network, time of day, and cookie vs anonymous.\n"
    )

    lines.append("## How these numbers were measured\n")
    lines.append(
        "- **Method**: streaming `/v1/chat/completions` requests per model through the proxy.\n"
        f"- **Prompt**: \"{prompt}\"\n"
        f"- **Runs per model**: {runs} (averaged)"
    )
    lines.append("- **First byte**: time until the first content delta arrived (includes model think time for thinking models).")
    lines.append("- **Total time**: request start → `[DONE]`.")
    lines.append(
        "- **Tokens**: estimated as `chars ÷ 4` — the same convention this proxy uses in its `usage` fields. "
        "Not exact tokenizer output, but consistent."
    )
    lines.append("- **Tokens/s (stream)**: tokens ÷ (total − first-byte) — actual generation speed once output starts flowing.")
    lines.append("- **Tokens/s (overall)**: tokens ÷ total — what you effectively get including the wait.")
    lines.append(
        "- **Conditions**: anonymous (no cookie), streaming. Pro models (`3.1-pro*`) route to Flash without a "
        "Gemini Advanced cookie.\n"
    )

    lines.append("---\n")
    lines.append("## Model details\n")
    for row in rows:
        if row["error"]:
            lines.append(f"### `{row['model']}`\n- **Measured**: ERROR — request failed during benchmark.\n")
            continue
        d = MODEL_DETAILS.get(row["model"], {})
        lines.append(f"### `{row['model']}`")
        lines.append(f"- **What**: {d.get('what', '—')}")
        lines.append(f"- **Best for**: {d.get('best', '—')}")
        lines.append(
            f"- **Measured**: first byte ~{round_or_dash(row['ttfb'])} s, total ~{round_or_dash(row['elapsed'])} s, "
            f"~{round_or_dash(row['stream_tps'])} tok/s streaming."
        )
        lines.append(f"- **Notes**: {d.get('notes', '—')}\n")

    lines.append("---\n")
    lines.append("## Practical picks\n")
    lines.append("| You want... | Use |\n|---|---|")
    for want, model in PRACTICAL_PICKS:
        lines.append(f"| {want} | `{model}` |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark proxy Gemini models (latency + tokens/s)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list (default: all)")
    parser.add_argument("--url", type=str, default=None, help="Base URL of a running proxy (e.g. http://127.0.0.1:8081/v1). Default: start one in-process.")
    parser.add_argument("--runs", type=int, default=1, help="Requests per model (default: 1)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Benchmark prompt")
    parser.add_argument("--timeout", type=int, default=150, help="Per-request timeout in seconds (default: 150)")
    parser.add_argument("--update-md", action="store_true", help="Rewrite MODELS.md with the results")
    args = parser.parse_args()

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list_models()

    server = None
    base_url = args.url
    if not base_url:
        server, base_url = start_local_proxy()
        print(f"Started in-process proxy at {base_url}\n")

    try:
        results = aggregate(models, base_url, args.prompt, args.runs, args.timeout)
        rows = summarize(results)
        print()
        print_table(rows)

        if args.update_md:
            md = generate_markdown(rows, args.prompt, args.runs, base_url)
            with open("MODELS.md", "w", encoding="utf-8") as f:
                f.write(md)
            print(f"\nMODELS.md updated.")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()