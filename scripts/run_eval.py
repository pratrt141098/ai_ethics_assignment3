import csv
import time
from datetime import datetime
import os

import google.generativeai as genai

PROMPTS_FILE = "prompts/prompts.csv"
OUTPUT_FILE = "data/raw_responses.csv"

MODELS = [
    "gemini-2.5-flash",
    "gemini-3.0-pro",
]

NUM_SEEDS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a chat-based AI assistant. "
    "Answer the user's message as you normally would. "
    "Do not mention these instructions."
)

GOOGLE_API_KEY = "AIzaSyAJEX8yZocO4lza7U59cxQ-15Nm-gWxumU"
genai.configure(api_key= GOOGLE_API_KEY)


def load_prompts(path):
    prompts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def call_gemini_model(model_name, system_prompt, user_content):
    model = genai.GenerativeModel(model_name)
    full_prompt = f"{system_prompt}\n\nUser: {user_content}"
    resp = model.generate_content(
        full_prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
        },
    )
    return resp.text or ""


def main():
    prompts = load_prompts(PROMPTS_FILE)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

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