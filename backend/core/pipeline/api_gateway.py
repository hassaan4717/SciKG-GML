from __future__ import annotations

import time
import logging
import threading
from typing import Any, Callable, Optional, Sequence, TypeVar
from openai import APIConnectionError, APITimeoutError, OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


T = TypeVar("T")


class APIRequestGateway:
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}

    def __init__(
        self,
        client: OpenAI,
        chat_model: str,
        embedding_model: Optional[str] = None,
        retry_delay: int = 30,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.retry_delay = max(1, int(retry_delay))

        self._cooldown_lock = threading.Lock()
        self._retry_after = 0.0

    @staticmethod
    def get_status_code(exc: Exception) -> Optional[int]:
        status_code = getattr(exc, "status_code", None)

        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)

        try:
            return int(status_code)
        except (TypeError, ValueError):
            return None

    @classmethod
    def should_retry(cls, exc: Exception) -> bool:
        status_code = cls.get_status_code(exc)

        if status_code in cls.RETRYABLE_STATUS_CODES:
            return True

        return isinstance(
            exc,
            (
                APITimeoutError,
                APIConnectionError,
            ),
        )

    def _wait_for_cooldown(self) -> None:
        while True:
            with self._cooldown_lock:
                remaining = self._retry_after - time.monotonic()

            if remaining <= 0:
                return

            time.sleep(remaining)

    def _start_cooldown(self) -> None:
        retry_after = time.monotonic() + self.retry_delay

        with self._cooldown_lock:
            self._retry_after = max(
                self._retry_after,
                retry_after,
            )

    def request(
        self,
        operation: str,
        request_fn: Callable[[], T],
    ) -> T:
        while True:
            self._wait_for_cooldown()

            try:
                return request_fn()

            except Exception as exc:
                status_code = self.get_status_code(exc)

                if not self.should_retry(exc):
                    logging.error(
                        "%s request failed and stop - type=%s - status=%s - error=%r",
                        operation,
                        type(exc).__name__,
                        status_code,
                        exc,
                    )
                    raise

                logging.warning(
                    "%s request failed. Retry - type=%s - status=%s - error=%r",
                    operation,
                    type(exc).__name__,
                    status_code,
                    exc,
                )

                try:
                    self.client.close()
                except Exception:
                    pass

                from core.utils import get_llm_client
                self.client = get_llm_client()

                self._start_cooldown()

    @staticmethod
    def _completion_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected LLM response format.") from exc

        return content.strip() if isinstance(content, str) else ""

    def chat_completion(
        self,
        messages: Sequence[dict[str, str]],
        **request_options: Any,
    ) -> Any:
        normalized_messages = [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in messages
            if message.get("role") and message.get("content")
        ]

        if not normalized_messages:
            raise ValueError("Chat messages cannot be empty.")

        return self.request(
            operation="Chat request",
            request_fn=lambda: self.client.chat.completions.create(
                model=self.chat_model,
                messages=normalized_messages,
                **request_options,
            ),
        )

    def chat_messages(
        self,
        messages: Sequence[dict[str, str]],
        **request_options: Any,
    ) -> str:
        """
        Send a full system/user/assistant message history.
        """
        response = self.chat_completion(
            messages=messages,
            **request_options,
        )

        text = self._completion_text(response)

        if not text:
            finish_reason = None

            try:
                finish_reason = response.choices[0].finish_reason
            except (AttributeError, IndexError, TypeError):
                pass

            raise RuntimeError(
                "The LLM returned an empty response "
                f"(finish_reason={finish_reason!r})."
            )

        return text

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        previous_prompt: Optional[str] = None,
        assistant_answer: Optional[str] = None,
        **request_options: Any,
    ) -> str:
        """
        Send one prompt, optionally with a system prompt.
        """
        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError("Chat prompt cannot be empty.")

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": str(system_prompt).strip(),
                }
            )

        if previous_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": str(previous_prompt).strip(),
                }
            )

        if assistant_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": str(assistant_answer).strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return self.chat_messages(
            messages=messages,
            **request_options,
        )

    def embed_texts(
        self,
        texts: Sequence[str],
        **request_options: Any,
    ) -> list[list[float]]:
        if not self.embedding_model:
            raise RuntimeError("No embedding model has been configured.")

        normalized_texts = [str(text) for text in texts if str(text).strip()]

        if not normalized_texts:
            return []

        response = self.request(
            operation="Embedding request",
            request_fn=lambda: self.client.embeddings.create(
                model=self.embedding_model,
                input=normalized_texts,
                **request_options,
            ),
        )

        return [item.embedding for item in response.data]

    def embed_query(
        self,
        text: str,
        **request_options: Any,
    ) -> list[float]:
        text = str(text).strip()

        if not text:
            raise ValueError("Embedding query cannot be empty.")

        return self.embed_texts(
            [text],
            **request_options,
        )[0]
