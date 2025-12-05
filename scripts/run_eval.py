import csv
import time
from datetime import datetime
import os

from google import genai
from google.genai import types

PROMPTS_FILE = "prompts/prompts.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "raw_responses.csv")

MODELS = [
    "gemini-3-pro-preview",
    "gemini-2.5-flash"
]

NUM_SEEDS = 1
TEMPERATURE = 0.7
MAX_TOKENS = 5120

SYSTEM_PROMPT = "Answer in under 75 words."


def load_prompts(path: str):
    prompts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def call_gemini_model(client: genai.Client, model_name: str, system_prompt: str, user_content: str) -> str:
    full_prompt = f"{system_prompt}\n\nUser: {user_content}"

    resp = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )

    if not resp.candidates:
        return "ERROR: No candidates returned"

    cand = resp.candidates[0]
    if not cand.content or not getattr(cand.content, "parts", None):
        # Avoid `.text` crash when MAX_TOKENS / SAFETY etc. yields empty parts
        return f"ERROR: Empty content (finish_reason={cand.finish_reason.name})"

    return resp.text or ""


def main():
    # Uses GEMINI_API_KEY from the environment
    client = genai.Client()

    prompts = load_prompts(PROMPTS_FILE)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "indicator_id",
            "convo_id",
            "turn_index",
            "role",
            "prompt_text",
            "difficulty",
            "model_name",
            "seed",
            "response_text",
            "timestamp",
        ])

        for prompt in prompts:
            indicator_id = prompt["indicator_id"]
            convo_id = prompt["convo_id"]
            turn_index = prompt["turn_index"]
            role = prompt["role"]
            prompt_text = prompt["text"]
            difficulty = prompt.get("difficulty", "")

            if role != "user":
                continue

            for model_name in MODELS:
                for seed in range(1, NUM_SEEDS + 1):
                    try:
                        response_text = call_gemini_model(
                            client=client,
                            model_name=model_name,
                            system_prompt=SYSTEM_PROMPT,
                            user_content=prompt_text,
                        )
                    except Exception as e:
                        response_text = f"ERROR: {e}"

                    timestamp = datetime.utcnow().isoformat()

                    writer.writerow([
                        indicator_id,
                        convo_id,
                        turn_index,
                        role,
                        prompt_text,
                        difficulty,
                        model_name,
                        seed,
                        response_text,
                        timestamp,
                    ])

                    time.sleep(0.5)


if __name__ == "__main__":
    main()
