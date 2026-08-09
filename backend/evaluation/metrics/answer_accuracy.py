import re
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.utils import get_api_gateway

STATEMENT_GENERATOR_PROMPT = """
### Task
Generate concise and independent factual statements from the given text.

Each statement must:
- Contain one factual claim.
- Be understandable independently.
- Preserve the meaning of the original text.

### Input Text
{text}

Output should be like this:
{{
  "statements": [
    "First factual statement",
    "Second factual statement"
  ]
}}
"""


CORRECTNESS_PROMPT = """
### Task
Compare the Answer Statements with the Ground Truth Statements.

CRITICAL RULE FOR LENIENCY:
If the Answer is a short entity or phrase (e.g., "Tom Pepys") and the Ground Truth is a full sentence (e.g., "Samuel Pepys had dinner with Tom Pepys"), DO NOT penalize the answer for missing contextual words like verbs or locations. If the core entity or main subject matches, classify it as TP (True Positive). Ignore grammatical completeness.

Classify statements as:

- TP: Present in the answer and supported by the ground truth.
- FP: Present in the answer but unsupported or contradicted.
- FN: Present in the ground truth but missing from the answer.

### Question
{question}

### Answer Statements
{answer_statements}

### Ground Truth Statements
{ground_truth_statements}

Output should be like this:
{{
  "TP": [
    {{
      "statement": "statement text",
      "reason": "brief reason"
    }}
  ],
  "FP": [
    {{
      "statement": "statement text",
      "reason": "brief reason"
    }}
  ],
  "FN": [
    {{
      "statement": "statement text",
      "reason": "brief reason"
    }}
  ]
}}
"""


def fbeta_score(
    tp: int,
    fp: int,
    fn: int,
    beta: float = 1.0,
) -> float:
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(recall)


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
                "Sending answer accuracy request (attempt %d/%d)", attempt, max_attempts
            )

            if attempt > 1:
                response = gateway.chat(
                    prompt=fix_res_prompt,
                    assistant_answer=response,
                    previous_prompt=prompt,
                )
            else:
                response = gateway.chat(prompt=prompt)

            if response is None:
                 raise ValueError("Received empty response from API.")
            
            text = str(response).strip()
            
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError("No JSON object found in the response.")

            data = json.loads(json_str)

            if not isinstance(data, dict):
                raise TypeError("The LLM response must be a JSON object.")

            return data

        except Exception as exc:
            last_error = exc
            logging.warning(
                "Answer accuracy request failed (attempt %d/%d). Error: %s",
                attempt,
                max_attempts,
                exc
            )

    logging.error(
        "Answer accuracy request failed after %d attempts. %s",
        max_attempts,
        last_error,
    )
    print("-------------------")
    print(response)
    print("-------------------")

    return None


def generate_statements(
    text: str,
    max_attempts: int = 3,
) -> Optional[List[str]]:
    """
    Generate factual statements from text.
    """
    text = str(text or "").strip()

    if not text:
        return []

    prompt = STATEMENT_GENERATOR_PROMPT.format(text=text)

    fix_res_prompt = """
        The previous response was invalid.

        Return exactly one valid JSON object:

        {
        "statements": [
            "First factual statement",
            "Second factual statement"
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

    statements = data.get("statements")

    if not isinstance(statements, list):
        logging.error(
            "Statement generation response does not contain a valid statements list."
        )
        return None

    valid_statements = [
        str(statement).strip() for statement in statements if str(statement).strip()
    ]

    if not valid_statements:
        logging.error("Statement generation returned no valid statements.")
        return None

    return valid_statements


def _validate_classification_items(
    items: Any,
) -> List[Dict[str, str]]:
    """
    Validate TP, FP or FN items.
    """
    if not isinstance(items, list):
        return []

    valid: List[Dict[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        statement = str(item.get("statement", "")).strip()
        reason = str(item.get("reason", "")).strip()

        if statement:
            valid.append({"statement": statement, "reason": reason})

    return valid


def calculate_factuality(
    question: str,
    answer_statements: List[str],
    ground_truth_statements: List[str],
    beta: float = 1.0,
    max_attempts: int = 3,
) -> float:
    """
    Calculate factuality using TP, FP and FN classifications.
    """
    if not answer_statements and not ground_truth_statements:
        return 1.0

    prompt = CORRECTNESS_PROMPT.format(
        question=str(question or "").strip(),
        answer_statements=json.dumps(answer_statements, ensure_ascii=False),
        ground_truth_statements=json.dumps(ground_truth_statements, ensure_ascii=False),
    )

    fix_res_prompt = """
        The previous response was invalid.

        Return exactly one valid JSON object:

        {
        "TP": [
            {
            "statement": "statement text",
            "reason": "brief reason"
            }
        ],
        "FP": [
            {
            "statement": "statement text",
            "reason": "brief reason"
            } 
        ],
        "FN": [
            {
            "statement": "statement text",
            "reason": "brief reason"
            } 
        ]
        }

        The response must contain the keys "TP", "FP", and "FN" and at least one statement with reason.
        Each value must be a JSON list.
        Do not include markdown or additional text.
    """

    data = _request_json(
        prompt=prompt,
        fix_res_prompt=fix_res_prompt,
        max_attempts=max_attempts,
    )

    if data is None:
        return float("nan")

    if not all(key in data for key in ["TP", "FP", "FN"]):
        logging.error("Correctness response is missing TP, FP or FN.")
        return float("nan")

    tp_items = _validate_classification_items(data["TP"])
    fp_items = _validate_classification_items(data["FP"])
    fn_items = _validate_classification_items(data["FN"])

    tp = len(tp_items)
    fp = len(fp_items)
    fn = len(fn_items)

    return fbeta_score(tp=tp, fp=fp, fn=fn, beta=beta)


def calculate_semantic_similarity(
    answer: str,
    ground_truth: str,
) -> float:
    """
    Calculate normalized cosine similarity between embeddings.
    """
    answer = str(answer or "").strip()
    ground_truth = str(ground_truth or "").strip()

    if not answer and not ground_truth:
        return 1.0

    if not answer or not ground_truth:
        return 0.0

    gateway = get_api_gateway()
    embeddings = gateway.embed_texts([answer, ground_truth])

    if not isinstance(embeddings, list) or len(embeddings) != 2:
        raise ValueError("Embedding gateway must return exactly two embeddings.")

    answer_embedding = np.asarray(embeddings[0], dtype=np.float32)
    ground_truth_embedding = np.asarray(embeddings[1], dtype=np.float32)

    denominator = np.linalg.norm(answer_embedding) * np.linalg.norm(
        ground_truth_embedding
    )

    if denominator == 0:
        return 0.0

    cosine_similarity = float(
        np.dot(answer_embedding, ground_truth_embedding) / denominator
    )
    cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))

    return float((cosine_similarity + 1.0) / 2.0)


def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    weights: Sequence[float] = (0.50, 0.50),
    beta: float = 1.0,
    max_attempts: int = 3,
) -> float:
    """
    Combine factuality and semantic similarity.
    """
    answer = str(answer or "").strip()
    ground_truth = str(ground_truth or "").strip()

    if len(weights) != 2:
        raise ValueError("weights must contain factuality and similarity weights.")

    factuality_weight = float(weights[0])
    similarity_weight = float(weights[1])

    if factuality_weight < 0 or similarity_weight < 0:
        raise ValueError("Metric weights cannot be negative.")

    if factuality_weight + similarity_weight == 0:
        raise ValueError("At least one metric weight must be greater than zero.")

    if not answer and not ground_truth:
        return 1.0

    answer_statements = generate_statements(answer, max_attempts=max_attempts)
    ground_truth_statements = generate_statements(
        ground_truth, max_attempts=max_attempts
    )

    if answer_statements is None or ground_truth_statements is None:
        return float("nan")

    factuality_score = (
        calculate_factuality(
            question=question,
            answer_statements=answer_statements,
            ground_truth_statements=ground_truth_statements,
            beta=beta,
            max_attempts=max_attempts,
        )
        if factuality_weight > 0
        else 0.0
    )

    similarity_score = (
        calculate_semantic_similarity(answer=answer, ground_truth=ground_truth)
        if similarity_weight > 0
        else 0.0
    )

    if np.isnan(factuality_score) or np.isnan(similarity_score):
        return float("nan")

    return float(
        np.average(
            [factuality_score, similarity_score],
            weights=[factuality_weight, similarity_weight],
        )
    )