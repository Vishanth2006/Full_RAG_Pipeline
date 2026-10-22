"""
Gemini LLM service using the official google-genai SDK.
Reads API key from environment. Never exposes secrets in logs or UI.
"""

import logging
import os
from typing import Optional

from google import genai
from google.genai import types

from config.settings import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS,
)

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Return API key from environment. Raises if missing."""
    key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your API key."
        )
    return key


def validate_api_config() -> tuple[bool, str]:
    """
    Check whether the API key and model name are configured.
    Returns (is_valid, message).  Never exposes the key value.
    """
    try:
        key = _get_api_key()
        if not key:
            return False, "API key is empty."
        model = GEMINI_MODEL
        if not model:
            return False, "GEMINI_MODEL is not configured."
        return True, f"API key present ({len(key)} chars). Model: {model}"
    except ValueError as exc:
        return False, str(exc)


def generate_answer(
    system_prompt: str,
    user_message: str,
    model_name: str = GEMINI_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """
    Call the Gemini API and return a dict with:
    {
        "answer": str,
        "model": str,
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "error": str | None,
    }
    """
    result = {
        "answer": "",
        "model": model_name,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "error": None,
    }

    try:
        api_key = _get_api_key()
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        result["answer"] = response.text or ""

        # Extract token usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            result["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
            result["completion_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
            result["total_tokens"] = getattr(usage, "total_token_count", 0) or 0

    except Exception as exc:
        error_msg = str(exc)
        # Sanitise: remove any substring that looks like an API key
        api_key_val = GEMINI_API_KEY or ""
        if api_key_val and api_key_val in error_msg:
            error_msg = error_msg.replace(api_key_val, "[REDACTED]")
        logger.error("Gemini API call failed: %s", error_msg)
        result["error"] = _classify_error(error_msg)

    return result


def _classify_error(raw_error: str) -> str:
    """Return a user-friendly error description without leaking secrets."""
    low = raw_error.lower()
    if "api_key" in low or "api key" in low or "invalid key" in low or "401" in low:
        return "Invalid or missing API key. Check your GEMINI_API_KEY in the .env file."
    if "quota" in low or "429" in low or "resource_exhausted" in low:
        return "API quota exceeded. Please wait and retry, or upgrade your Gemini plan."
    if "model" in low and ("not found" in low or "404" in low):
        return (
            f"Model not found. Check GEMINI_MODEL in your .env. "
            f"Supported: {', '.join(SUPPORTED_GEMINI_MODELS)}"
        )
    if "network" in low or "connection" in low or "timeout" in low:
        return "Network error. Check your internet connection and retry."
    return f"Gemini API error: {raw_error}"
