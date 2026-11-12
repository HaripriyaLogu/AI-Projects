import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2:3b",
    "system": """
    You are a technical content writer specializing in
    Generative AI, Agentic AI, RAG, Python and AI projects.

    Explain concepts accurately.
    Use simple language.
    Never invent technical facts.
    If unsure, say so.
    """,
    "prompt": "Explain the difference between RAG and fine-tuning in 5 bullet points.",
    "stream": False
}

try:
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    print("\nGenerated Response:\n")
    print(result["response"])

except requests.exceptions.ConnectionError:
    print("Could not connect to Ollama. Check whether Ollama is running.")

except requests.exceptions.Timeout:
    print("Ollama took too long to respond.")

except requests.exceptions.RequestException as error:
    print("Request failed:", error)
