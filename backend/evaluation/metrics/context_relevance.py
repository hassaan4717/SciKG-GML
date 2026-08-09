import json
import logging
from typing import List, Optional

from core.utils import get_api_gateway

CONTEXT_RELEVANCE_PROMPT = """
### Task
Evaluate the relevance of the Context for answering the Question using ONLY the information provided.
Respond ONLY with a number from 0-2. Do not explain.

### Rating Scale
0: Context has NO relevant information
1: Context has PARTIAL relevance
2: Context has RELEVANT information

### Question
{question}

### Context
{context}

output should be like this:
{{
  "relevance": 0 or 1 or 2
}}
"""


def _get_llm_rating(
    prompt: str,
    max_attempts: int,
) -> Optional[float]:
    """
    Request one relevance rating through APIRequestGateway.
    """
    gateway = get_api_gateway()

    repeat = max_attempts
    last_error: Exception | None = None
    response = None

    while repeat > 0:
        attempt = max_attempts - repeat + 1
        repeat -= 1

        try:
            logging.info(
                "Sending context relevance request " "(attempt %d/%d)",
                attempt,
                max_attempts,
            )

            if attempt > 1:
                fix_res_prompt = """
                    The previous response was invalid.
                    Return exactly one valid JSON object:
                    {
                    "relevance": 0
                    }
                    The relevance value must be 0, 1, or 2.
                    Do not include markdown or additional text.
                """

                response = gateway.chat(
                    prompt=fix_res_prompt,
                    assistant_answer=response,
                    previous_prompt=prompt,
                )
            else:
                response = gateway.chat(prompt=prompt)

            data = json.loads(response)
            rating = int(data.get("relevance"))
            if rating not in [0, 1, 2]:
                raise ValueError(f"Invalid relevance rating")

            return rating

        except Exception as exc:
            last_error = exc

            logging.warning(
                "Context relevance request failed " "(attempt %d/%d). %s",
                attempt,
                max_attempts,
                exc,
            )

    logging.error(
        "Context relevance evaluation failed " "after %d attempts. %s",
        max_attempts,
        last_error,
    )

    return None


def compute_context_relevance(
    question: str,
    contexts: List[str],
    max_attempts: int = 3,
) -> float:
    """
    Evaluate how relevant the retrieved context is to the question.
    The LLM returns a value on a 0-2 scale.
    """
    context_str = "\n\n".join(contexts)

    prompt = CONTEXT_RELEVANCE_PROMPT.format(
        question=question,
        context=context_str,
    )
    rating = _get_llm_rating(
        prompt=prompt,
        max_attempts=max_attempts,
    )

    if rating is None:
        return float("nan")

    normalized_score = rating / 2.0

    return float(normalized_score)
