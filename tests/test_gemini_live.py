"""
Live Gemini API integration test.
Requires a valid GEMINI_API_KEY in .env.
Skipped automatically if the key is missing or a placeholder.
"""

import pytest


def _get_key_or_skip():
    """Return the API key or skip the test if not set."""
    try:
        from config.settings import GEMINI_API_KEY
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            pytest.skip("No real GEMINI_API_KEY configured in .env")
        return GEMINI_API_KEY
    except Exception as exc:
        pytest.skip(f"Could not load settings: {exc}")


def test_live_validate_api_config():
    """Confirms validate_api_config() reports valid when real key is present."""
    _get_key_or_skip()
    from services.gemini_service import validate_api_config

    is_valid, msg = validate_api_config()
    assert is_valid, f"API config reported invalid: {msg}"


def test_live_gemini_generate_answer():
    """
    Makes a real Gemini API call with a short prompt.
    Verifies the answer is non-empty and no error is returned.
    """
    _get_key_or_skip()
    from services.gemini_service import generate_answer

    result = generate_answer(
        system_prompt="You are a helpful assistant. Answer concisely.",
        user_message="Reply with exactly the word: WORKING",
        max_output_tokens=20,
    )

    assert result["error"] is None, f"API returned an error: {result['error']}"
    assert len(result["answer"]) > 0, "API returned an empty answer"
    assert result["model"] is not None
    print(f"\n[OK] Live Gemini response: '{result['answer'].strip()}'")
    print(f"     Tokens used - prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']}")

