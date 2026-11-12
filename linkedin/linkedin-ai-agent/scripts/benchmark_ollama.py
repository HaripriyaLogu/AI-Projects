
#!/usr/bin/env python3
"""Benchmark an Ollama model on a GitHub-hosted runner."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

PROMPT = """Write a professional LinkedIn post for an AI Engineer about
why retrieval-augmented generation (RAG) needs evaluation, not just a vector
database. Use a clear hook, explain two practical points, give one short
takeaway, and add 4 relevant hashtags. Avoid fabricated statistics and
unsupported claims. Keep it between 120 and 180 words."""


def post_json(path: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    print(f"Benchmarking model: {MODEL}")
    print(f"Ollama endpoint: {BASE_URL}")

    started = time.perf_counter()

    try:
        result = post_json(
            "/api/generate",
            {
                "model": MODEL,
                "prompt": PROMPT,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 400,
                },
            },
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Benchmark request failed: {exc}", file=sys.stderr)
        return 1

    wall_seconds = time.perf_counter() - started

    response_text = result.get("response") or ""
    thinking_text = result.get("thinking") or ""

    eval_count = int(result.get("eval_count") or 0)
    eval_duration_ns = int(result.get("eval_duration") or 0)
    prompt_eval_duration_ns = int(result.get("prompt_eval_duration") or 0)
    load_duration_ns = int(result.get("load_duration") or 0)

    tokens_per_second = (
        eval_count / (eval_duration_ns / 1_000_000_000)
        if eval_duration_ns > 0
        else None
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "runner": os.getenv(
            "RUNNER_NAME", "GitHub-hosted Ubuntu runner"
        ),
        "wall_clock_seconds": round(wall_seconds, 2),
        "model_load_seconds": round(load_duration_ns / 1e9, 2),
        "prompt_tokens": result.get("prompt_eval_count"),
        "generated_tokens": eval_count,
        "prompt_eval_seconds": round(prompt_eval_duration_ns / 1e9, 2),
        "generation_seconds": round(eval_duration_ns / 1e9, 2),
        "generation_tokens_per_second": (
            round(tokens_per_second, 2)
            if tokens_per_second is not None
            else None
        ),
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "thinking_characters": len(thinking_text),
        "response_characters": len(response_text),
        "response": response_text,
    }

    Path("benchmark-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== BENCHMARK RESULTS ==========")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=======================================")

    if not response_text.strip():
        print("ERROR: Model returned an empty final response.")
        return 1

    print("\n========== GENERATED LINKEDIN POST ==========")
    print(response_text)
    print("=============================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())