from __future__ import annotations

import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import context_to_list, safe_mean
from evaluation.metrics.context_recall import compute_context_recall
from evaluation.metrics.context_relevance import compute_context_relevance

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DATASET_PREFIXES = os.getenv("DATASET_FILES", []).split(",")
WORKER_LIMIT = max(1, int(os.getenv("WORKER_LIMIT", 5)))


def process_evaluation_job(
    job: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate one question.

    Network/API retries are handled by APIRequestGateway.
    Invalid evaluator outputs are retried inside each metric.
    """
    sample_id = str(job["id"])
    question = str(job["question"]).strip()
    contexts = context_to_list(job["contexts"])
    ground_truth = str(job["ground_truth"]).strip()

    try:
        logging.info(
            "Evaluating question: %s",
            sample_id,
        )

        relevance_score = compute_context_relevance(
            question=question,
            contexts=contexts,
        )

        recall_score = compute_context_recall(
            question=question,
            contexts=contexts,
            reference_answer=ground_truth,
        )

        return {
            "id": sample_id,
            "success": True,
            "context_relevancy": float(relevance_score),
            "context_recall": float(recall_score),
        }

    except Exception as exc:
        logging.exception(
            "Failed to evaluate question %s",
            sample_id,
        )

        return {
            "id": sample_id,
            "success": False,
            "context_relevancy": float("nan"),
            "context_recall": float("nan"),
            "error": (f"{type(exc).__name__}: {exc}"),
        }


def evaluate_dataset(
    dataset: Dict[str, List[Any]],
) -> Dict[str, float]:
    """
    Evaluate context relevance and context recall
    using ThreadPoolExecutor.
    """
    results: Dict[str, List[float]] = {
        "context_relevancy": [],
        "context_recall": [],
    }

    ids = dataset["ids"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]

    num_samples = len(questions)

    if not (len(ids) == len(questions) == len(contexts_list) == len(ground_truths)):
        raise ValueError("Dataset fields do not have equal lengths.")

    jobs: List[Dict[str, Any]] = []

    for index in range(num_samples):
        jobs.append(
            {
                "id": ids[index],
                "question": questions[index],
                "contexts": contexts_list[index],
                "ground_truth": ground_truths[index],
            }
        )

    for start in range(0, len(jobs), WORKER_LIMIT):
        batch = jobs[start : start + WORKER_LIMIT]

        max_workers = min(WORKER_LIMIT, len(batch))

        logging.info(
            "Sending evaluation batch of %d jobs with %d workers",
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
                        "Unhandled evaluation error for question %s: %s",
                        job["id"],
                        exc,
                    )
                    continue

                sample_id = sample["id"]

                if not sample["success"]:
                    logging.error(
                        "Evaluation failed for %s: %s",
                        sample_id,
                        sample.get(
                            "error",
                            "Unknown error",
                        ),
                    )
                    continue

                relevance_score = float(sample["context_relevancy"])
                recall_score = float(sample["context_recall"])

                logging.info(
                    "Question %s | Context Relevance: %.4f | Context Recall: %.4f",
                    sample_id,
                    relevance_score,
                    recall_score,
                )

                if not np.isnan(relevance_score):
                    results["context_relevancy"].append(relevance_score)

                if not np.isnan(recall_score):
                    results["context_recall"].append(recall_score)

    return {
        "context_relevancy": safe_mean(results["context_relevancy"]),
        "context_recall": safe_mean(results["context_recall"]),
    }


def main() -> None:

    if not DATASET_PREFIXES:
        raise RuntimeError("DATASET_FILES cannot be empty.")

    for prefix in DATASET_PREFIXES:
        data_path = f"evaluation/results/{prefix}/predictions_{prefix}.json"
        report_path = f"evaluation/results/{prefix}/report_{prefix}.json"

        logging.info(
            "Evaluating dataset: %s",
            prefix,
        )

        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as file:
                file_data = json.load(file)
        else:
            file_data = []

        if not isinstance(file_data, list):
            logging.error(
                "Prediction file must contain a list: %s",
                data_path,
            )
            continue

        question_types = sorted(
            {
                str(item.get("question_type", "Unknown"))
                for item in file_data
                if (isinstance(item, dict) and "error" not in item)
            }
        )

        all_results: Dict[str, Dict[str, float]] = {}

        for question_type in question_types:
            logging.info(
                "Evaluating question type: %s",
                question_type,
            )

            filtered_items = [
                item
                for item in file_data
                if (
                    isinstance(item, dict)
                    and "error" not in item
                    and str(item.get("question_type", "Unknown")) == question_type
                )
            ]

            if not filtered_items:
                logging.warning(
                    "No valid samples found for question type: %s",
                    question_type,
                )
                continue

            ids = [str(item.get("id", "")) for item in filtered_items]

            questions = [
                str(item.get("question", "")).strip() for item in filtered_items
            ]

            ground_truths = [
                str(item.get("gold_answer", "")).strip() for item in filtered_items
            ]

            contexts = [
                context_to_list(item.get("context", "")) for item in filtered_items
            ]

            dataset = {
                "ids": ids,
                "question": questions,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }

            all_results[question_type] = evaluate_dataset(
                dataset=dataset,
            )

        logging.info("Final Evaluation Summary:")

        for question_type, metrics in all_results.items():
            logging.info(
                "Question Type: %s",
                question_type,
            )

            logging.info(
                "  Context Relevance: %.4f",
                metrics["context_relevancy"],
            )

            logging.info(
                "  Context Recall: %.4f",
                metrics["context_recall"],
            )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        try:
            with open(report_path, "r", encoding="utf-8") as file:
                report = json.load(file)
            if not isinstance(report, dict):
                report = {}
        except (FileNotFoundError, json.JSONDecodeError):
            report = {}

        report.setdefault(prefix, {})
        report[prefix]["retrieval"] = all_results

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        logging.info(
            "Report saved for %s: %s",
            prefix,
            report_path,
        )


if __name__ == "__main__":
    main()
