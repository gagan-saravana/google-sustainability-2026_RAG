# gemini_client.py: the one place that talks to the Gemini API.

# Shared by embeddings.py (indexing + query embedding) and synthesize.py (answers)

from __future__ import annotations

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src import config

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise SystemExit(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key "
                "(free key: https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _is_retryable(exc: BaseException) -> bool:
    # google.genai.errors.APIError exposes the HTTP status as `.code`
    return getattr(exc, "code", None) in (429, 500, 503, 504)


# Free tiers have rate limits: retry 429/5xx with exponential backoff (2s, 4s, ... up to 60s).
gemini_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


@gemini_retry
def generate_text(system_instruction: str, user_prompt: str) -> str:
    """One grounded-answer call. Low temperature: we want faithful, repeatable answers."""
    response = get_client().models.generate_content(
        model=config.GENERATION_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=config.TEMPERATURE,
        ),
    )
    return (response.text or "").strip()
