from __future__ import annotations

import os
import json
import random
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from collections import defaultdict

from core.pipeline.graph_rag import GraphRAG
from core.utils import generate_rag_answer, source_filename, get_api_gateway

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DATA_DIR = Path(os.getenv("DATA_DIR", "evaluation/dataset"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", "evaluation/results"))
WORK_SPACE_DIR = Path(os.getenv("WORK_SPACE_DIR", "evaluation/workspace"))

DATASET_PREFIXES = os.getenv("DATASET_FILES", []).split(",")

if os.getenv("SAMPLE", "-1") == "-1":
    SAMPLE = float("inf")
else:
    SAMPLE = max(1, int(os.getenv("SAMPLE")))


def sample_question_type(
    grouped_questions: Dict[str, List[dict]],
    sample_size: int,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    random_generator = random.Random(seed)
    sampled_questions: Dict[str, List[dict]] = {}

    for corpus_name, questions in grouped_questions.items():
        questions_by_type: Dict[str, List[dict]] = defaultdict(list)

        for question in questions:
            question_type = str(question.get("question_type", "Unknown")).strip()
            questions_by_type[question_type].append(question)

        sampled_questions[corpus_name] = []

        for question_type, type_questions in questions_by_type.items():
            selected = random_generator.sample(
                type_questions,
                k=min(sample_size, len(type_questions)),
            )

            sampled_questions[corpus_name].extend(selected)

            logging.info(
                "Sampled questions | corpus=%s | type=%s | selected=%d/%d",
                corpus_name,
                question_type,
                len(selected),
                len(type_questions),
            )

    return sampled_questions


def process_corpus(
    corpus_name: str,
    prefix: str,
    context: str,
    work_space: Path,
    questions: List[dict],
) -> None:
    logging.info("Processing corpus: %s (Subset: %s)", corpus_name, prefix)

    output_dir = RESULT_DIR / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"predictions_{prefix}.json"
    work_space.mkdir(parents=True, exist_ok=True)

    rag = GraphRAG(persist_dir=str(work_space), api_gateway=get_api_gateway())

    if getattr(rag, "state_loaded", False):
        logging.info("Existing index loaded for: %s", prefix)
    else:
        logging.info("Indexing corpus: %s into %s", corpus_name, prefix)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / source_filename(title=corpus_name)
            file_path.write_text(context, encoding="utf-8")

            rag.ingest(file_path=str(file_path))

        logging.info(
            "Indexed corpus: %s (%d words)",
            corpus_name,
            len(context.split()),
        )

    if not questions:
        logging.warning("No questions found for corpus: %s", corpus_name)
        return

    logging.info("Found %d questions for %s", len(questions), corpus_name)

    results: List[dict] = []

    for question_data in questions:
        question_id = str(question_data.get("id", "")).strip()
        question = str(question_data.get("question", "")).strip()

        if not question:
            logging.warning("Question %s is empty and was skipped.", question_id)
            continue

        logging.info("Answering question: %s", question_id)

        try:
            retrieval_result = rag.query(question)
            contexts = rag.format_context(retrieval_result).strip()

            answer = generate_rag_answer(
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                context=contexts,
            )

            generated_answer = str(answer or "").strip()

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "source": corpus_name,
                    "context": contexts,
                    "evidence": question_data.get("evidence", ""),
                    "question_type": question_data.get("question_type", ""),
                    "generated_answer": generated_answer,
                    "gold_answer": question_data.get("answer", ""),
                }
            )

        except Exception as exc:
            logging.exception("Error processing question %s", question_id)

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "source": corpus_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    logging.info("Saved %d predictions to: %s", len(results), output_path)


def load_data() -> Tuple[Dict[str, str], Dict[str, List[dict]], Dict[str, str]]:
    corpus_contexts: Dict[str, str] = {}
    corpus_questions: Dict[str, List[dict]] = defaultdict(list)
    corpus_prefixes: Dict[str, str] = {}

    for prefix in DATASET_PREFIXES:
        filename = f"{prefix}_data.json"
        file_path = DATA_DIR / filename

        if not file_path.is_file():
            logging.error("File not found: %s", file_path)
            continue

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                logging.error(
                    "File %s must contain a single JSON object (dict).", file_path.name
                )
                continue

            corpus_name = str(data.get("corpus_name", "")).strip()
            context = str(data.get("context", "")).strip()
            questions = data.get("questions", [])

            if not corpus_name or not context:
                logging.warning(
                    "Skipping %s: missing corpus_name or context.", file_path.name
                )
                continue

            corpus_contexts[corpus_name] = context
            corpus_prefixes[corpus_name] = prefix

            if isinstance(questions, list):
                corpus_questions[corpus_name].extend(questions)

            logging.info(
                "Loaded file: %s | corpus=%s | words=%d | questions=%d",
                file_path.name,
                corpus_name,
                len(context.split()),
                len(questions),
            )

        except Exception:
            logging.exception("Failed to load file: %s", file_path)

    return corpus_contexts, dict(corpus_questions), corpus_prefixes


def main() -> None:
    if not DATASET_PREFIXES:
        raise RuntimeError("DATASET_FILES environment variable cannot be empty.")

    WORK_SPACE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_contexts, corpus_questions, corpus_prefixes = load_data()

    if not corpus_contexts:
        logging.error("No valid corpus data was loaded.")
        return

    sampled_questions = sample_question_type(
        grouped_questions=corpus_questions,
        sample_size=SAMPLE,
        seed=42,
    )

    for corpus_name, context in corpus_contexts.items():
        questions = sampled_questions.get(corpus_name, [])
        prefix = corpus_prefixes[corpus_name]

        if not questions:
            logging.warning(
                "No sampled questions found for corpus: %s",
                corpus_name,
            )
            continue

        process_corpus(
            corpus_name=corpus_name,
            prefix=prefix,
            context=context,
            work_space=WORK_SPACE_DIR / prefix,
            questions=questions,
        )


if __name__ == "__main__":
    main()
