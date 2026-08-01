from __future__ import annotations

import os
import re
import logging
import unicodedata
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import Callable, Dict, Hashable, List, Optional, Tuple

from .models import TextChunk

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

RERANKER_MODEL = os.getenv("RERANKER_MODEL")

SearchResult = Tuple[str, Dict[str, object], float]
Tokenizer = Callable[[str], List[str]]

_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[-'][^\W_]+)*",
    flags=re.UNICODE,
)


def default_tokenizer(text: str) -> List[str]:
    """
    General-purpose tokenizer for BM25.
    """
    if not text:
        return []

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    return _TOKEN_PATTERN.findall(normalized)


def document_key(
    text: str,
    metadata: Dict[str, object],
) -> Hashable:
    """
    Return a stable identifier for deduplication and rank fusion.
    """
    source = metadata.get("source")
    chunk_id = metadata.get("chunk_id")

    if source is not None and chunk_id is not None:
        return (
            "chunk",
            str(source),
            str(chunk_id),
        )

    return ("text", text)


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.tokenizer = default_tokenizer

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[TextChunk] = []

    def clear(self) -> None:
        """
        Clear the current BM25 index.
        """
        self.bm25 = None
        self.chunks = []

    def build(
        self,
        chunks: List[TextChunk],
    ) -> None:
        """
        Build the BM25 index from TextChunk objects.
        """
        self.clear()

        indexed_chunks: List[TextChunk] = []
        tokenized_corpus: List[List[str]] = []

        for chunk in chunks:
            text = getattr(chunk, "text", None)

            if not text or not text.strip():
                continue

            tokens = self.tokenizer(text)

            if not tokens:
                continue

            indexed_chunks.append(chunk)
            tokenized_corpus.append(tokens)

        if not indexed_chunks:
            logging.warning(
                "BM25 index was not built because no valid chunks " "were provided."
            )
            return

        self.chunks = indexed_chunks

        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )

        logging.info(
            "BM25 index built from %d chunks.",
            len(self.chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: Optional[float] = 0.0,
    ) -> List[SearchResult]:
        """
        Search the BM25 index.
        """
        if (
            self.bm25 is None
            or not self.chunks
            or top_k <= 0
            or not query
            or not query.strip()
        ):
            return []

        query_tokens = self.tokenizer(query)

        if not query_tokens:
            return []

        scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=np.float64,
        )

        if scores.size == 0:
            return []

        number_of_results = min(
            top_k,
            len(self.chunks),
        )

        # Faster than sorting the complete score array for large datasets.
        if number_of_results < len(scores):
            candidate_indices = np.argpartition(
                scores,
                -number_of_results,
            )[-number_of_results:]

            sorted_indices = candidate_indices[
                np.argsort(
                    scores[candidate_indices],
                    kind="stable",
                )[::-1]
            ]
        else:
            sorted_indices = np.argsort(
                scores,
                kind="stable",
            )[::-1]

        results: List[SearchResult] = []

        for index in sorted_indices:
            score = float(scores[index])

            if not np.isfinite(score):
                continue

            if min_score is not None and score <= min_score:
                continue

            chunk = self.chunks[index]

            metadata: Dict[str, object] = {
                "source": str(chunk.source),
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                "retriever": "bm25",
            }

            results.append(
                (
                    chunk.text,
                    metadata,
                    score,
                )
            )

        return results


def reciprocal_rank_fusion(
    dense: List[SearchResult],
    sparse: List[SearchResult],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """
    Combine dense and sparse retrieval results using weighted RRF.
    """
    if k < 0:
        raise ValueError("RRF parameter 'k' must be non-negative.")

    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("RRF weights must be non-negative.")

    if dense_weight + sparse_weight == 0:
        raise ValueError("At least one RRF weight must be greater than zero.")

    fused_scores: Dict[Hashable, float] = {}
    documents: Dict[
        Hashable,
        Tuple[str, Dict[str, object]],
    ] = {}

    retrieval_details: Dict[
        Hashable,
        Dict[str, object],
    ] = {}

    def add_results(
        results: List[SearchResult],
        retriever_name: str,
        weight: float,
    ) -> None:
        seen_in_retriever = set()
        unique_rank = 0

        for text, metadata, original_score in results:
            key = document_key(text, metadata)

            # A retriever must not contribute more than once
            # for the same chunk.
            if key in seen_in_retriever:
                continue

            seen_in_retriever.add(key)
            unique_rank += 1

            contribution = weight / (k + unique_rank)

            fused_scores[key] = fused_scores.get(key, 0.0) + contribution

            if key not in documents:
                documents[key] = (
                    text,
                    dict(metadata),
                )

            details = retrieval_details.setdefault(
                key,
                {},
            )

            details[f"{retriever_name}_rank"] = unique_rank
            details[f"{retriever_name}_score"] = float(original_score)

    add_results(
        dense,
        retriever_name="dense",
        weight=dense_weight,
    )

    add_results(
        sparse,
        retriever_name="sparse",
        weight=sparse_weight,
    )

    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    if top_k is not None:
        if top_k <= 0:
            return []

        ranked_keys = ranked_keys[:top_k]

    fused_results: List[SearchResult] = []

    for key in ranked_keys:
        text, original_metadata = documents[key]

        metadata = dict(original_metadata)
        metadata.update(retrieval_details[key])
        metadata["retriever"] = "rrf"

        fused_results.append(
            (
                text,
                metadata,
                float(fused_scores[key]),
            )
        )

    return fused_results


class Reranker:
    def __init__(
        self,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = RERANKER_MODEL

        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length

        self._model: Optional[CrossEncoder] = None

    def _load(self) -> CrossEncoder:
        """
        Lazily load the CrossEncoder model.
        """
        if self._model is None:
            logging.info(
                "Loading reranker model '%s'.",
                self.model_name,
            )

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )

        return self._model

    @staticmethod
    def _deduplicate_candidates(
        candidates: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Remove duplicate candidates while preserving ranking order.
        """
        unique_candidates: List[SearchResult] = []
        seen = set()

        for candidate in candidates:
            text, metadata, _ = candidate
            key = document_key(text, metadata)

            if key in seen:
                continue

            seen.add(key)
            unique_candidates.append(candidate)

        return unique_candidates

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: int = 5,
    ) -> List[SearchResult]:
        """
        Rerank retrieval candidates using a CrossEncoder.
        """
        if not candidates or top_n <= 0 or not query or not query.strip():
            return []

        unique_candidates = self._deduplicate_candidates(candidates)

        if not unique_candidates:
            return []

        model = self._load()

        query_document_pairs = [(query, text) for text, _, _ in unique_candidates]

        raw_scores = model.predict(
            query_document_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(
            raw_scores,
            dtype=np.float64,
        )

        # Convert (N, 1) outputs to (N,).
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        if scores.ndim != 1:
            raise RuntimeError(
                "The selected CrossEncoder does not return one "
                "scalar relevance score per query-document pair. "
                f"Received output shape: {scores.shape}"
            )

        if len(scores) != len(unique_candidates):
            raise RuntimeError(
                "The number of reranker scores does not match "
                "the number of candidates."
            )

        ranked_indices = np.argsort(
            scores,
            kind="stable",
        )[::-1]

        number_of_results = min(
            top_n,
            len(unique_candidates),
        )

        reranked_results: List[SearchResult] = []

        for index in ranked_indices[:number_of_results]:
            text, original_metadata, retrieval_score = unique_candidates[index]

            metadata = dict(original_metadata)
            metadata["retrieval_score"] = float(retrieval_score)
            metadata["reranker_model"] = self.model_name
            metadata["retriever"] = "cross_encoder"

            reranked_results.append(
                (
                    text,
                    metadata,
                    float(scores[index]),
                )
            )

        return reranked_results
