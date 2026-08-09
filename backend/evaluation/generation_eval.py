import os
import json
import logging
from dotenv import load_dotenv
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from core.utils import context_to_list, safe_mean
from evaluation.metrics.rouge import compute_rouge_score
from evaluation.metrics.coverage import compute_coverage_score
from evaluation.metrics.faithfulness import compute_faithfulness_score
from evaluation.metrics.answer_accuracy import compute_answer_correctness

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DATASET_PREFIXES = os.getenv("DATASET_FILES", []).split(",")
WORKER_LIMIT = max(1, int(os.getenv("WORKER_LIMIT", 5)))

QUESTION_TYPES = [
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
]
METRIC_CONFIG = {
    "Fact Retrieval": [
        "rouge_score",
        "answer_correctness",
    ],
    "Complex Reasoning": [
        "answer_correctness",
        "coverage_score",
        "faithfulness",
    ],
    "Contextual Summarize": [
        "rouge_score",
        "answer_correctness",
        "coverage_score",
        "faithfulness",
    ],
    "Creative Generation": [
        "answer_correctness",
        "coverage_score",
        "faithfulness",
    ]
}


def compute_metric(
    metric: str,
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> float:
    """
    Compute one generation metric.
    """
    if metric == "rouge_score":
        return float(compute_rouge_score(answer=answer, ground_truth=ground_truth))

    if metric == "answer_correctness":
        return float(
            compute_answer_correctness(
                question=question, answer=answer, ground_truth=ground_truth
            )
        )

    if metric == "coverage_score":
        return float(
            compute_coverage_score(
                question=question, reference=ground_truth, response=answer
            )
        )

    if metric == "faithfulness":
        return float(
            compute_faithfulness_score(
                question=question, answer=answer, contexts=contexts
            )
        )

    raise ValueError(f"Unsupported generation metric: {metric}")


def process_evaluation_job(
    job: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate all configured metrics for one question.
    """
    sample_id = str(job["id"])
    question = str(job["question"]).strip()
    answer = str(job["answer"]).strip()
    contexts = context_to_list(job["contexts"])
    ground_truth = str(job["ground_truth"]).strip()
    metrics = job["metrics"]

    logging.info("Evaluating question: %s", sample_id)

    scores: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    for metric in metrics:
        try:
            logging.info("Computing %s for question %s", metric, sample_id)

            scores[metric] = compute_metric(
                metric=metric,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

        except Exception as exc:
            logging.exception("Metric %s failed for question %s", metric, sample_id)
            scores[metric] = float("nan")
            errors[metric] = f"{type(exc).__name__}: {exc}"

    return {"id": sample_id, "scores": scores, "errors": errors}


def evaluate_dataset(
    dataset: Dict[str, List[Any]],
    metrics: List[str],
) -> Dict[str, float]:
    """
    Evaluate generation metrics using ThreadPoolExecutor.
    """
    results: Dict[str, List[float]] = {metric: [] for metric in metrics}

    ids = dataset["ids"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]

    if not (
        len(ids)
        == len(questions)
        == len(answers)
        == len(contexts_list)
        == len(ground_truths)
    ):
        raise ValueError("Dataset fields do not have equal lengths.")

    jobs = [
        {
            "id": ids[index],
            "question": questions[index],
            "answer": answers[index],
            "contexts": contexts_list[index],
            "ground_truth": ground_truths[index],
            "metrics": metrics,
        }
        for index in range(len(questions))
    ]

    for start in range(0, len(jobs), WORKER_LIMIT):
        batch = jobs[start : start + WORKER_LIMIT]

        max_workers = min(WORKER_LIMIT, len(batch))

        logging.info(
            "Sending generation evaluation batch of %d requests with %d workers",
            len(batch),
            max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(process_evaluation_job, job): job for job in batch
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]

                try:
                    sample = future.result()

                except Exception as exc:
                    logging.exception(
                        "Unhandled evaluation error for question %s: %s", job["id"], exc
                    )
                    continue

                sample_id = sample["id"]

                for metric in metrics:
                    score = float(sample["scores"].get(metric, float("nan")))

                    if np.isnan(score):
                        logging.warning(
                            "Question %s | %s skipped: %s",
                            sample_id,
                            metric,
                            sample["errors"].get(metric, "Invalid score"),
                        )
                        continue

                    results[metric].append(score)

    return {metric: safe_mean(results[metric]) for metric in metrics}


def main() -> None:
    if not DATASET_PREFIXES:
        raise RuntimeError("DATASET_FILES cannot be empty.")

    for subset in DATASET_PREFIXES:
        data_path = f"evaluation/results/{subset}/predictions_{subset}.json"
        report_path = f"evaluation/results/{subset}/report_{subset}.json"

        logging.info("Evaluating subset: %s", subset)

        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as file:
                file_data = json.load(file)
        else:
            file_data = []

        if not isinstance(file_data, list) or not file_data:
            logging.error(
                "Prediction file must contain a list or is empty: %s", data_path
            )
            continue

        all_results: Dict[str, Dict[str, float]] = {}

        for question_type in QUESTION_TYPES:
            logging.info("Evaluating question type: %s", question_type)

            filtered_items = [
                item
                for item in file_data
                if isinstance(item, dict)
                and "error" not in item
                and item.get("question_type") == question_type
            ]

            if not filtered_items:
                logging.warning(
                    "No valid samples found for question type: %s", question_type
                )
                continue

            dataset = {
                "ids": [str(item.get("id", "")) for item in filtered_items],
                "question": [
                    str(item.get("question", "")).strip() for item in filtered_items
                ],
                "answer": [
                    str(item.get("generated_answer", "")).strip()
                    for item in filtered_items
                ],
                "contexts": [
                    context_to_list(item.get("context", "")) for item in filtered_items
                ],
                "ground_truth": [
                    str(item.get("gold_answer", "")).strip() for item in filtered_items
                ],
            }

            all_results[question_type] = evaluate_dataset(
                dataset=dataset,
                metrics=METRIC_CONFIG[question_type],
            )

        logging.info("Final Generation Evaluation Summary:")

        for question_type, metrics in all_results.items():
            logging.info("Question Type: %s", question_type)
            for metric, score in metrics.items():
                logging.info("  %s: %.4f", metric, score)

        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as file:
                try:
                    report = json.load(file)
                except json.JSONDecodeError:
                    report = {}
        else:
            report = {}

        if not isinstance(report, dict):
            report = {}

        report.setdefault(subset, {})
        report[subset].setdefault("generation", {})

        for q_type, metrics_dict in all_results.items():
            report[subset]["generation"].setdefault(q_type, {})
            for metric_name, score in metrics_dict.items():
                report[subset]["generation"][q_type][metric_name] = score

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        logging.info("Generation report saved for %s: %s", subset, report_path)


if __name__ == "__main__":
    main()
