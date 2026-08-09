import json
import logging
from typing import Any, Dict, List, Optional

from core.utils import get_api_gateway

FACT_EXTRACTION_PROMPT = """
### Task
Extract distinct factual statements from the Reference Answer that can be independently verified.

### Question
{question}

### Reference Answer
{reference}

Output should be like this:
{{
  "facts": [
    "First factual statement",
    "Second factual statement"
  ]
}}
"""


FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the Reference Answer, determine whether it is covered by the Generated Response.

Use only the meaning expressed in the Generated Response.

Set "attributed" to:
- 1 when the fact is directly stated or clearly implied.
- 0 when the fact is missing, contradicted, or unsupported.

### Question
{question}

### Generated Response
{response}

### Reference Facts
{facts}

Output should be like this:
{{
  "classifications": [
    {{
      "statement": "exact reference fact",
      "attributed": 1
    }},
    {{
      "statement": "exact reference fact",
      "attributed": 0
    }}
  ]
}}
"""


def _request_json(
    prompt: str,
    fix_res_prompt: str,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    """
    Request one valid JSON object through APIRequestGateway.
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
                "Sending coverage request (attempt %d/%d)", attempt, max_attempts
            )

            if attempt > 1:
                response = gateway.chat(
                    prompt=fix_res_prompt,
                    assistant_answer=response,
                    previous_prompt=prompt,
                )
            else:
                response = gateway.chat(prompt=prompt)

            data = json.loads(response)

            if not isinstance(data, dict):
                raise TypeError("The LLM response must be a JSON object.")

            return data

        except Exception as exc:
            last_error = exc
            logging.warning(
                "Coverage request failed (attempt %d/%d).",
                attempt,
                max_attempts,
            )

    logging.error(
        "Coverage request failed after %d attempts. %s",
        max_attempts,
        last_error,
    )
    return None


def _validate_facts(
    facts: Any,
) -> List[str]:
    """
    Validate extracted factual statements.
    """
    if not isinstance(facts, list):
        return []

    return [
        str(fact).strip() for fact in facts if isinstance(fact, str) and fact.strip()
    ]


def _normalize_attributed(
    value: Any,
) -> Optional[int]:
    """
    Normalize attributed values into 0 or 1.
    """
    if value is True:
        return 1

    if value is False:
        return 0

    if isinstance(value, str):
        value = value.strip()

        if value == "1":
            return 1

        if value == "0":
            return 0

        return None

    if isinstance(value, (int, float)):
        if value == 1:
            return 1

        if value == 0:
            return 0

    return None


def _validate_classifications(
    classifications: Any,
) -> List[Dict[str, Any]]:
    """
    Validate fact coverage classifications.
    """
    if not isinstance(classifications, list):
        return []

    valid: List[Dict[str, Any]] = []

    for item in classifications:
        if not isinstance(item, dict):
            continue

        statement = str(item.get("statement", "")).strip()
        attributed = _normalize_attributed(item.get("attributed"))

        if statement and attributed is not None:
            valid.append({"statement": statement, "attributed": attributed})

    return valid


def _extract_facts(
    question: str,
    reference: str,
    max_attempts: int,
) -> Optional[List[str]]:
    """
    Extract factual statements from the reference answer.
    """
    prompt = FACT_EXTRACTION_PROMPT.format(question=question, reference=reference)

    fix_res_prompt = """
The previous response was invalid.

Return exactly one valid JSON object:

{
  "facts": [
    "First factual statement",
    "Second factual statement"
  ]
}

The "facts" value must be a JSON list of strings.
Do not include markdown or additional text.
"""

    data = _request_json(
        prompt=prompt,
        fix_res_prompt=fix_res_prompt,
        max_attempts=max_attempts,
    )

    if data is None:
        return None

    facts = _validate_facts(data.get("facts"))

    if not facts:
        logging.error("Fact extraction returned no valid facts.")
        return None

    return facts


def _check_fact_coverage(
    question: str,
    facts: List[str],
    response: str,
    max_attempts: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Determine which reference facts are covered by the response.
    """
    prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response,
        facts=json.dumps(facts, ensure_ascii=False),
    )

    fix_res_prompt = """
The previous response was invalid.

Return exactly one valid JSON object:

{
  "classifications": [
    {
      "statement": "exact reference fact",
      "attributed": 1
    }
  ]
}

The "classifications" value must be a JSON list.
Each item must contain:
- "statement": a non-empty string
- "attributed": 0 or 1

Evaluate every reference fact.
Do not include markdown or additional text.
"""

    data = _request_json(
        prompt=prompt,
        fix_res_prompt=fix_res_prompt,
        max_attempts=max_attempts,
    )

    if data is None:
        return None

    classifications = _validate_classifications(data.get("classifications"))

    if not classifications:
        logging.error("Coverage evaluation returned no valid classifications.")
        return None

    if len(classifications) != len(facts):
        logging.error(
            "Coverage classification count does not match fact count: facts=%d, classifications=%d",
            len(facts),
            len(classifications),
        )
        return None

    return classifications


def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    max_attempts: int = 3,
) -> float:
    """
    Measure what percentage of reference facts are covered in the response.
    """
    question = str(question or "").strip()
    reference = str(reference or "").strip()
    response = str(response or "").strip()

    if not reference:
        return 1.0

    facts = _extract_facts(
        question=question,
        reference=reference,
        max_attempts=max_attempts,
    )

    if facts is None:
        return float("nan")

    classifications = _check_fact_coverage(
        question=question,
        facts=facts,
        response=response,
        max_attempts=max_attempts,
    )

    if classifications is None:
        return float("nan")

    attributed = [item["attributed"] for item in classifications]
    return float(sum(attributed) / len(attributed))
