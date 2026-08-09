from rouge_score import rouge_scorer


def compute_rouge_score(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rouge1",
    mode: str = "recall",
) -> float:
    """
    Compute ROUGE score between generated answer and ground truth.
    """
    answer = str(answer or "").strip()
    ground_truth = str(ground_truth or "").strip()

    if not answer or not ground_truth:
        return 0.0

    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)

    if rouge_type not in scores:
        raise ValueError(f"Unsupported ROUGE type: {rouge_type}")

    rouge_result = scores[rouge_type]

    if not hasattr(rouge_result, mode):
        raise ValueError(f"Unsupported ROUGE mode: {mode}")

    return float(getattr(rouge_result, mode))
