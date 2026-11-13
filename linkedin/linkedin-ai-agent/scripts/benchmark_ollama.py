
#!/usr/bin/env python3
"""
Benchmark Qwen3 using Ollama on a GitHub-hosted runner.
Inspect model configuration and generate a LinkedIn post.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# OLLAMA API HELPER
# --------------------------------------------------

def post_json(path: str, payload: dict, timeout: int = 900) -> dict:
    """Send a POST request to the Ollama API."""

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def extract_final_answer(response_text: str) -> str:
    """
    Extract the final answer after Qwen3's thinking marker.
    """

    marker = "</think>"

    if marker in response_text:
        return response_text.rsplit(marker, 1)[-1].strip()

    return response_text.strip()
# --------------------------------------------------
# MAIN BENCHMARK
# --------------------------------------------------

def main() -> int:

    print("=" * 55)
    print("OLLAMA CLOUD MODEL BENCHMARK")
    print("=" * 55)

    print(f"Model: {MODEL}")
    print(f"Ollama endpoint: {BASE_URL}")

    # ----------------------------------------------
    # STEP 1: INSPECT MODEL CONFIGURATION
    # ----------------------------------------------

    print("\n[STEP 1] Inspecting model configuration...")

    try:
        model_info = post_json(
            "/api/show",
            {
                "model": MODEL
            },
        )

        print("\n========== MODEL CONFIGURATION ==========")

        print("\nThinking configuration:")
        print(
            json.dumps(
                model_info.get("thinking"),
                indent=2
            )
        )

        print("\nModel details:")
        print(
            json.dumps(
                model_info.get("details", {}),
                indent=2
            )
        )

        print("\nModel parameters:")
        print(model_info.get("parameters", "Not available"))

        print("=========================================")

    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:

        print(
            f"Model inspection failed: {exc}",
            file=sys.stderr,
        )
        return 1

    # ----------------------------------------------
    # STEP 2: GENERATE LINKEDIN POST
    # ----------------------------------------------

    print("\n[STEP 2] Generating LinkedIn post...")

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

        print(
            f"Generation request failed: {exc}",
            file=sys.stderr,
        )
        return 1

    wall_seconds = time.perf_counter() - started

    # ----------------------------------------------
    # STEP 3: EXTRACT MODEL RESPONSE
    # ----------------------------------------------

    

    message = result.get("message") or {}

    raw_response = message.get("content") or ""
    thinking_text = message.get("thinking") or ""

    response_text = extract_final_answer(raw_response)

    # ----------------------------------------------
    # STEP 4: EXTRACT PERFORMANCE METRICS
    # ----------------------------------------------

    eval_count = int(result.get("eval_count") or 0)

    eval_duration_ns = int(
        result.get("eval_duration") or 0
    )

    prompt_eval_duration_ns = int(
        result.get("prompt_eval_duration") or 0
    )

    load_duration_ns = int(
        result.get("load_duration") or 0
    )

    tokens_per_second = (
        eval_count / (eval_duration_ns / 1_000_000_000)
        if eval_duration_ns > 0
        else None
    )

    # ----------------------------------------------
    # STEP 5: PREPARE BENCHMARK REPORT
    # ----------------------------------------------

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": MODEL,

        "runner": os.getenv(
            "RUNNER_NAME",
            "GitHub-hosted Ubuntu runner",
        ),

        "api_endpoint": "/api/chat",

        "wall_clock_seconds": round(
            wall_seconds, 2
        ),

        "model_load_seconds": round(
            load_duration_ns / 1e9, 2
        ),

        "prompt_tokens": result.get(
            "prompt_eval_count"
        ),

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
        "raw_response_characters": len(raw_response),
        "final_response_characters": len(response_text),
        "thinking": thinking_text,
        "response": response_text,
    }
    

    # ----------------------------------------------
    # STEP 6: SAVE RESULTS
    # ----------------------------------------------

    output_file = Path("benchmark-results.json")

    output_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== BENCHMARK RESULTS ==========")

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=======================================")

    # ----------------------------------------------
    # STEP 7: DISPLAY GENERATED CONTENT
    # ----------------------------------------------

    if not response_text.strip():

        print(
            "\nERROR: Model returned an empty response.",
            file=sys.stderr,
        )

        return 1

    print("\n========== GENERATED LINKEDIN POST ==========")
    print(response_text)
    print("=============================================")

    # ----------------------------------------------
    # STEP 8: BASIC OUTPUT VALIDATION
    # ----------------------------------------------

    word_count = len(response_text.split())

    print("\n========== CONTENT VALIDATION ==========")

    word_count = len(response_text.split())

    print(f"Word count: {word_count}")

    if 120 <= word_count <= 180:
        print("Word count check: PASSED")
    else:
        print("Word count check: FAILED")


    if result.get("done_reason") == "length":
        print("WARNING: Generation stopped at the token limit.")
        print("The generated post may be incomplete.")
    else:
        print("Generation stop reason:", result.get("done_reason"))

    if "</think>" in raw_response:
        print("Thinking marker detected and final answer extracted.")
    else:
        print("No thinking marker detected.")

    print("=========================================")

    

if __name__ == "__main__":
    raise SystemExit(main())
