from __future__ import annotations

import os
import re
import httpx
import logging
import secrets
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import Any, Optional, Sequence, List

from core.pipeline.api_gateway import APIRequestGateway

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
BASE_OPENAI_URL = os.getenv("BASE_OPENAI_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 1024))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 12))

INSUFFICIENT_CONTEXT_MESSAGE = (
    "This topic isn't sufficiently covered in the available text."
)
GENERATION_ERROR_MESSAGE = (
    "I'm sorry, I encountered an error while processing your request."
)

ALLOWED_HISTORY_ROLES = {"user", "assistant"}


def get_llm_client() -> OpenAI:
    """
    Create one cached OpenAI-compatible client.
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_OPENAI_URL,
        timeout=httpx.Timeout(
            timeout=120.0,
            connect=30.0,
            read=120.0,
            write=30.0,
            pool=30.0,
        ),
        max_retries=0,
    )

    return client


def get_api_gateway() -> APIRequestGateway:
    """
    Create one shared request gateway for chat and embedding calls.
    """
    return APIRequestGateway(
        client=get_llm_client(),
        chat_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        retry_delay=30,
    )


def call_llm(
    prompt: Optional[str] = None,
    messages: Optional[Sequence[dict[str, str]]] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    json_response: bool = False,
    **request_options: Any,
) -> str:
    """
    Send a prompt or message history through the shared API gateway.
    """
    if bool(prompt) == bool(messages):
        raise ValueError("Provide exactly one of prompt or messages.")

    gateway = get_api_gateway()

    if max_tokens is not None:
        request_options["max_tokens"] = max_tokens

    request_options["temperature"] = temperature

    if json_response:
        request_options["response_format"] = {"type": "json_object"}

    if prompt is not None:
        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError("LLM prompt cannot be empty.")

        return gateway.chat(
            prompt=prompt,
            **request_options,
        )

    normalized_messages = [
        {
            "role": str(message.get("role", "")).strip(),
            "content": str(message.get("content", "")).strip(),
        }
        for message in messages or []
        if str(message.get("role", "")).strip()
        and str(message.get("content", "")).strip()
    ]

    if not normalized_messages:
        raise ValueError("LLM messages cannot be empty.")

    return gateway.chat_messages(
        messages=normalized_messages,
        **request_options,
    )


def generate_rag_answer(
    messages: list[dict[str, str]],
    context: str,
) -> str:
    """
    Answer the latest user message using retrieved GraphRAG context.
    """
    history = [
        {
            "role": str(message.get("role", "")).strip().lower(),
            "content": str(message.get("content", "")).strip(),
        }
        for message in messages
        if str(message.get("role", "")).strip().lower() in ALLOWED_HISTORY_ROLES
        and str(message.get("content", "")).strip()
    ][-CHAT_HISTORY_LIMIT:]

    if not history or history[-1]["role"] != "user":
        raise ValueError("The latest valid message must be from the user.")

    context = str(context).strip()

    if not context or context == "No relevant content found.":
        return INSUFFICIENT_CONTEXT_MESSAGE

    current_question = history[-1]["content"]
    previous_messages = history[:-1]

    system_prompt = """
        You are an expert Q&A system. You must answer questions using ONLY the provided GraphRAG evidence.

        The retrieved context often contains dense texts with many overlapping names and events. To avoid mixing up entities:
        1. Find the exact sentence or clause that matches the user's question.
        2. Identify the specific person, place, or thing associated with that exact action.

        Rules:
        1. You MUST write your step-by-step reasoning inside <thinking> tags.
        2. You MUST write your final, concise answer inside <answer> tags.
        3. If the evidence is insufficient, your <answer> must be exactly: "This topic isn't sufficiently covered in the available text."
        4. Do not add inline citations, citation numbers, or a Sources section in your answer.
    """.strip()

    user_prompt = f"""
        <retrieved_context>
        {context}
        </retrieved_context>

        <question>
        {current_question}
        </question>
    """.strip()

    api_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *previous_messages,
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:
        raw_response = call_llm(
            messages=api_messages,
            max_tokens=CHAT_MAX_TOKENS,
        )

        match = re.search(
            r"<answer>(.*?)</answer>", raw_response, re.DOTALL | re.IGNORECASE
        )
        if match:
            final_answer = match.group(1).strip()
        else:
            final_answer = re.sub(
                r"<thinking>.*?</thinking>",
                "",
                raw_response,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()

        return final_answer

    except Exception as exc:
        logging.exception(
            "LLM chat completion failed | error=%s",
            exc,
        )
        return GENERATION_ERROR_MESSAGE


def context_to_list(
    contexts: List[str] | str,
) -> List[str]:
    """
    Convert context input into a clean list of non-empty strings.
    """
    if isinstance(contexts, str):
        contexts = [contexts]

    return [str(context).strip() for context in contexts if str(context).strip()]


def safe_mean(
    values: List[float],
) -> float:
    """
    Calculate the average of valid scores.

    Returns NaN when there are no valid scores.
    """
    if not values:
        return float("nan")

    return float(np.mean(values))


def safe_float(value: Any, default: float = 1.0) -> float:
    """
    Convert graph values to JSON-safe floats.
    """
    try:
        return float(value)

    except (TypeError, ValueError, OverflowError):
        return default


def source_filename(
    title: str = None,
) -> str:
    random_hash = secrets.token_hex(6)

    if not title:
        return f"PastFile_{random_hash}"

    original_name = Path(str(title).strip()).name

    if not original_name or original_name in {".", ".."}:
        raise ValueError("Invalid source filename.")

    return f"{original_name}_{random_hash}"
