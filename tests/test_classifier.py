"""
Integration tests for classify_intent — require GROQ_API_KEY in env or Streamlit secrets.
Tests are skipped automatically when no API key is available.
"""
from __future__ import annotations

import os
import pytest

# Skip all tests in this module if GROQ_API_KEY is not available.
def _groq_key_available() -> bool:
    if os.getenv("GROQ_API_KEY"):
        return True
    try:
        import streamlit as st
        return bool(st.secrets.get("GROQ_API_KEY"))
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _groq_key_available(),
    reason="GROQ_API_KEY not set",
)

from intent_classifier import classify_intent  # noqa: E402 (imported after skip guard)


def test_nutrition():
    result = classify_intent("potrzebuję porady dotyczącej żywienia", [])
    assert result["intent"] == "NUTRITION"


def test_gear():
    result = classify_intent("jakie buty na mokre szlaki", [])
    assert result["intent"] == "GEAR"


def test_question():
    result = classify_intent("jak zrobić long run", [])
    assert result["intent"] == "QUESTION"
    assert "Q_LONG_RUN" not in str(result)  # no subtype leak


def test_frustration():
    result = classify_intent("znów coś mylisz", [])
    assert result["subtype"] == "Q_OTHER"


def test_plan():
    result = classify_intent("zrób mi plan na maraton za 12 tygodni", [])
    assert result["intent"] == "TRAINING_PLAN"
