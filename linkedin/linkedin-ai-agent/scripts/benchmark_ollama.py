
#!/usr/bin/env python3
"""Benchmark an Ollama chat model on a GitHub-hosted runner."""

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

SYSTEM_PROMPT = """You are a professional LinkedIn content writer
specializing in Artificial Intelligence and Generative AI.

Your task is to write polished LinkedIn posts.

STRICT RULES:
- Output only the final LinkedIn post.
- Never describe your planning or thought process.
- Never mention what the user asked you to do.
- Do not include introductory commentary.
- Do not include explanations outside the post.
- Follow the requested word count.
- Do not invent statistics or facts.
"""

PROMPT = """Write a professional LinkedIn post for an AI Engineer
about why Retrieval-Augmented Generation (RAG) needs evaluation,
not just a vector database.

Requirements:
1. Start with an engaging hook.
2. Explain two practical evaluation points.
3. Include one short takeaway.
4. Add four relevant hashtags.
5. Keep the post between 120 and 180 words.

Return only the finished LinkedIn post."""


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
    print("API endpoint: /api/chat")

    started = time.perf_counter()

    try:
        result = post_json(
            "/api/chat",
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": PROMPT,
                    },
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.2,
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

    # Extract the final response from Ollama Chat API.
    message = result.get("message") or {}

    response_text = message.get("content") or ""
    thinking_text = message.get("thinking") or ""

    # Extract performance metrics.
    eval_count = int(result.get("eval_count") or 0)
    eval_duration_ns = int(result.get("eval_duration") or 0)
    prompt_eval_duration_ns = int(
        result.get("prompt_eval_duration") or 0
    )
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
        "api_endpoint": "/api/chat",
        "wall_clock_seconds": round(wall_seconds, 2),
        "model_load_seconds": round(load_duration_ns / 1e9, 2),
        "prompt_tokens": result.get("prompt_eval_count"),
        "generated_tokens": eval_count,
        "prompt_eval_seconds": round(
            prompt_eval_duration_ns / 1e9, 2
        ),
        "generation_seconds": round(
            eval_duration_ns / 1e9, 2
        ),
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

    # Save benchmark results.
    Path("benchmark-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== BENCHMARK RESULTS ==========")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=======================================")

    # Validate the generated response.
    if not response_text.strip():
        print(
            "ERROR: Model returned an empty final response.",
            file=sys.stderr,
        )
        return 1

    print("\n========== GENERATED LINKEDIN POST ==========")
    print(response_text)
    print("=============================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())