from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.7):
    """
    Zwraca LLM na podstawie AI_PROVIDER w .env.
    Domyślnie: groq (darmowy, szybki).
    """
    provider = os.getenv("AI_PROVIDER", "groq")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-5",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:
        raise ValueError(f"Nieznany AI_PROVIDER: {provider}")


def invoke_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Wywołuje LLM z prostym string promptem.
    Zwraca string. Rzuca wyjątek gdy LLM niedostępny.
    """
    provider = os.getenv("AI_PROVIDER", "groq")
    logger.info(f"[LLM] Wywołanie przez provider: {provider}")
    try:
        llm = get_llm(temperature=temperature)
        response = llm.invoke(prompt)
        logger.info(f"[LLM] Sukces — {len(response.content)} znaków")
        return response.content
    except Exception as e:
        logger.error(f"[LLM] Błąd: {e}")
        raise
