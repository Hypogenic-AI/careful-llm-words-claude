"""OpenAI API wrapper with retry logic and token tracking."""

import time
import openai
from openai import OpenAI
from src.config import OPENAI_API_KEY, MODEL, MAX_TOKENS, TEMPERATURE


client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm(prompt: str, model: str = MODEL, max_tokens: int = MAX_TOKENS,
             temperature: float = TEMPERATURE, system: str = None) -> dict:
    """Call the OpenAI API with retry logic. Returns dict with response and usage."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "text": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": response.model,
            }
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
        except openai.APIError as e:
            print(f"  API error: {e}")
            time.sleep(2 ** attempt)

    return {"text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "model": model, "error": "max_retries"}
