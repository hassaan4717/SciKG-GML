import json
import logging
from typing import Any, Dict, List, Optional

from core.utils import get_api_gateway

CONTEXT_RECALL_PROMPT = """
### Task
Analyze each sentence in the Answer and determine if it can be attributed to the Context.
Respond STRICTLY with a JSON object containing a "classifications" list. 
DO NOT include markdown formatting (like ```json), DO NOT include any conversational text before or after the JSON.

### Example
Input:
Context: "Einstein won the Nobel Prize in 1921 for physics."
Answer: "Einstein received the Nobel Prize. He was born in Germany."

Output:
{{
  "classifications": [
    {{
      "statement": "Einstein received the Nobel Prize",
      "reason": "Matches context about Nobel Prize",
      "attributed": 1
    }},
    {{
      "statement": "He was born in Germany",
      "reason": "Birth information not in context",
      "attributed": 0
    }}
  ]
}}

### Actual Input
Context: "{context}"

Answer: "{answer}"

Question: "{question}" (for reference only)

### Your Response:
"""

import json
import logging
import re
from typing import Any, Dict, List


def _validate_classifications(response: Any) -> List[Dict[str, Any]]:
    """
    Validate and normalize classification objects with robust JSON extraction.
    """
    text = str(response).strip()

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        raise ValueError("No JSON object found in the response.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw text was: {text}")

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dictionary.")

    classifications = data.get("classifications", [])
    if not classifications:
        raise ValueError("No 'classifications' array found in JSON.")

    valid: List[Dict[str, Any]] = []

    for item in classifications:
        if not isinstance(item, dict):
            continue

        statement = str(item.get("statement", "")).strip()
        reason = str(item.get("reason", "")).strip()

        try:
            attributed = int(item.get("attributed", 0))
        except (ValueError, TypeError):
            attributed = 0

        valid.append(
            {
                "statement": statement,
                "reason": reason,
                "attributed": attributed,
            }
        )

    return valid


def _get_classifications(
    prompt: str,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    """
    Request context-recall classifications through
    APIRequestGateway.
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
                "Sending context recall request " "(attempt %d/%d)",
                attempt,
                max_attempts,
            )
            if attempt > 1:
                fix_res_prompt = """
                    The previous response contained invalid JSON or markdown formatting.
                    Generate the evaluation again.
                    Return EXACTLY one valid JSON object containing a non-empty "classifications" list.
                    NO markdown formatting (do not use ```). NO text before or after the JSON.
                    Each classification must contain:
                    {
                        "statement": "non-empty string",
                        "reason": "brief string",
                        "attributed": 0 or 1
                    }
                """
                response = gateway.chat(
                    prompt=prompt + fix_res_prompt, assistant_answer=response
                )
            else:
                response = gateway.chat(prompt=prompt)

            classifications = _validate_classifications(response)

            if not classifications:
                raise ValueError("The LLM returned no valid classifications.")

            return classifications

        except Exception as exc:
            last_error = exc

            logging.warning(
                "Context recall request failed " "(attempt %d/%d). %s",
                attempt,
                max_attempts,
                exc,
            )

    logging.error(
        "Context recall evaluation failed " "after %d attempts. %s",
        max_attempts,
        last_error,
    )

    return []


def compute_context_recall(
    question: str,
    contexts: List[str],
    reference_answer: str,
    max_attempts: int = 3,
) -> float:
    """
    Calculate context recall.

    The score represents the proportion of factual statements
    in the reference answer that are supported by the context.
    """
    context_str = "\n\n".join(contexts)

    prompt = CONTEXT_RECALL_PROMPT.format(
        question=question,
        context=context_str,
        answer=reference_answer,
    )

    classifications = _get_classifications(
        prompt=prompt,
        max_attempts=max_attempts,
    )

    if not classifications:
        return float("nan")

    attributed_values = [
        classification["attributed"] for classification in classifications
    ]

    return float(sum(attributed_values) / len(attributed_values))
