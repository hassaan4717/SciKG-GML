import json
import logging
from typing import Any, Dict, List, Optional

from core.utils import context_to_list, get_api_gateway

STATEMENT_GENERATION_PROMPT = """
### Task
Break the Answer into atomic factual statements that are fully understandable without pronouns or missing context.

Each statement must:
- Contain one factual claim.
- Preserve the meaning of the Answer.
- Be understandable independently.

### Question
{question}

### Answer
{answer}

Output should be like this:
{{
  "statements": [
    "First atomic factual statement",
    "Second atomic factual statement"
  ]
}}
"""


FAITHFULNESS_EVALUATION_PROMPT = """
### Task
Determine whether each Answer Statement can be directly supported or clearly inferred from the Context.

Use only the provided Context.

Set "verdict" to:
- 1 when the statement is supported.
- 0 when the statement is unsupported, contradicted, or cannot be inferred.

### Context
{context}

### Answer Statements
{statements}

Output should be like this:
{{
  "verdicts": [
    {{
      "statement": "exact answer statement",
      "verdict": 1,
      "reason": "brief explanation"
    }},
    {{
      "statement": "exact answer statement",
      "verdict": 0,
      "reason": "brief explanation"
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
                "Sending faithfulness request (attempt %d/%d)", attempt, max_attempts
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
                "Faithfulness request failed (attempt %d/%d).",
                attempt,
                max_attempts,
            )

    logging.error(
        "Faithfulness request failed after %d attempts. %s",
        max_attempts,
        last_error
    )
    return None


def _validate_statements(
    statements: Any,
) -> List[str]:
    """
    Validate generated atomic statements.
    """
    if not isinstance(statements, list):
        return []

    return [
        str(statement).strip()
        for statement in statements
        if isinstance(statement, str) and statement.strip()
    ]


def _normalize_verdict(
    value: Any,
) -> Optional[int]:
    """
    Normalize verdict values into 0 or 1.
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


def _validate_verdicts(
    verdicts: Any,
) -> List[Dict[str, Any]]:
    """
    Validate statement faithfulness verdicts.
    """
    if not isinstance(verdicts, list):
        return []

    valid: List[Dict[str, Any]] = []

    for item in verdicts:
        if not isinstance(item, dict):
            continue

        statement = str(item.get("statement", "")).strip()
        reason = str(item.get("reason", "")).strip()
        verdict = _normalize_verdict(item.get("verdict"))

        if statement and verdict is not None:
            valid.append({"statement": statement, "verdict": verdict, "reason": reason})

    return valid


def _generate_statements(
    question: str,
    answer: str,
    max_attempts: int,
) -> Optional[List[str]]:
    """
    Break the generated answer into atomic factual statements.
    """
    prompt = STATEMENT_GENERATION_PROMPT.format(question=question, answer=answer)

    fix_res_prompt = """
        The previous response was invalid.

        Return exactly one valid JSON object:

        {
        "statements": [
            "First atomic factual statement",
            "Second atomic factual statement"
        ]
        }

        The "statements" value must be a non-empty JSON list of strings.
        Do not include markdown or additional text.
    """

    data = _request_json(
        prompt=prompt,
        fix_res_prompt=fix_res_prompt,
        max_attempts=max_attempts,
    )

    if data is None:
        return None

    statements = _validate_statements(data.get("statements"))

    if not statements:
        logging.error("Faithfulness statement generation returned no valid statements.")
        return None

    return statements


def _evaluate_statements(
    statements: List[str],
    context: str,
    max_attempts: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Evaluate whether each answer statement is supported by context.
    """
    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        context=context,
        statements=json.dumps(statements, ensure_ascii=False),
    )

    fix_res_prompt = """
        The previous response was invalid.

        Return exactly one valid JSON object:

        {
        "verdicts": [
            {
            "statement": "exact answer statement",
            "verdict": 1,
            "reason": "brief explanation"
            }
        ]
        }

        The "verdicts" value must be a JSON list.
        Each item must contain:
        - "statement": a non-empty string
        - "verdict": 0 or 1
        - "reason": a brief string

        Evaluate every Answer Statement.
        Do not include markdown or additional text.
    """

    data = _request_json(
        prompt=prompt,
        fix_res_prompt=fix_res_prompt,
        max_attempts=max_attempts,
    )

    if data is None:
        return None

    verdicts = _validate_verdicts(data.get("verdicts"))

    if not verdicts:
        logging.error("Faithfulness evaluation returned no valid verdicts.")
        return None

    if len(verdicts) != len(statements):
        logging.error(
            "Faithfulness verdict count does not match statement count: statements=%d, verdicts=%d",
            len(statements),
            len(verdicts),
        )
        return None

    return verdicts


def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    max_attempts: int = 3,
) -> float:
    """
    Calculate the percentage of answer statements supported by context.
    """
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    contexts = context_to_list(contexts)

    if not answer:
        return 1.0

    statements = _generate_statements(
        question=question,
        answer=answer,
        max_attempts=max_attempts,
    )

    if statements is None:
        return float("nan")

    context = "\n\n".join(contexts)

    if not context:
        return 0.0

    verdicts = _evaluate_statements(
        statements=statements,
        context=context,
        max_attempts=max_attempts,
    )

    if verdicts is None:
        return float("nan")

    supported = [item["verdict"] for item in verdicts]
    return float(sum(supported) / len(supported))
