"""
Retrieval Evaluation Module.
FAANG Pattern: Comprehensive metrics for retrieval quality assessment.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    hit_rate: float
    mrr: float  # Mean Reciprocal Rank
    recall_at_k: float
    precision_at_k: float
    ndcg: float  # Normalized Discounted Cumulative Gain
    avg_score: float


class RetrievalEvaluator:
    """
    Comprehensive retrieval quality evaluator.

    FAANG Patterns:
    - Multiple metrics for holistic evaluation
    - Configurable relevance criteria
    - Support for batch evaluation
    """

    @staticmethod
    def calculate_hit_rate(retrieved_docs: List[Dict], relevant_pages: List[int]) -> float:
        """
        Calculate hit rate (whether any relevant doc was retrieved).

        Args:
            retrieved_docs: List of retrieved document dicts
            relevant_pages: List of relevant page numbers

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        for doc in retrieved_docs:
            if doc.get('metadata', {}).get('page') in relevant_pages:
                return 1.0
        return 0.0

    @staticmethod
    def calculate_mrr(retrieved_docs: List[Dict], relevant_pages: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved_docs: List of retrieved document dicts
            relevant_pages: List of relevant page numbers

        Returns:
            MRR score (1/rank of first relevant document)
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc.get('metadata', {}).get('page') in relevant_pages:
                return 1.0 / i
        return 0.0

    @staticmethod
    def calculate_recall_at_k(
        retrieved_docs: List[Dict],
        relevant_pages: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Args:
            retrieved_docs: List of retrieved document dicts
            relevant_pages: List of relevant page numbers
            k: Cutoff for evaluation

        Returns:
            Recall score
        """
        if not relevant_pages:
            return 0.0

        retrieved_pages = set()
        for doc in retrieved_docs[:k]:
            page = doc.get('metadata', {}).get('page')
            if page is not None:
                retrieved_pages.add(page)

        relevant_set = set(relevant_pages)
        hits = len(retrieved_pages.intersection(relevant_set))

        return hits / len(relevant_set)

    @staticmethod
    def calculate_precision_at_k(
        retrieved_docs: List[Dict],
        relevant_pages: List[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            retrieved_docs: List of retrieved document dicts
            relevant_pages: List of relevant page numbers
            k: Cutoff for evaluation

        Returns:
            Precision score
        """
        if not retrieved_docs or k == 0:
            return 0.0

        relevant_set = set(relevant_pages)
        hits = 0

        for doc in retrieved_docs[:k]:
            page = doc.get('metadata', {}).get('page')
            if page in relevant_set:
                hits += 1

        return hits / min(k, len(retrieved_docs))

    @classmethod
    def evaluate(
        cls,
        retrieved_docs: List[Dict],
        relevant_pages: Optional[List[int]] = None,
        k: int = 3
    ) -> RetrievalMetrics:
        """
        Compute all retrieval metrics.

        Args:
            retrieved_docs: List of retrieved document dicts
            relevant_pages: List of relevant page numbers (optional)
            k: Cutoff for evaluation

        Returns:
            RetrievalMetrics with all computed metrics
        """
        relevant_pages = relevant_pages or []

        # Calculate average score
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        avg_score = np.mean(scores) if scores else 0.0

        # If no ground truth, return only score metrics
        if not relevant_pages:
            return RetrievalMetrics(
                hit_rate=0.0,
                mrr=0.0,
                recall_at_k=0.0,
                precision_at_k=0.0,
                ndcg=0.0,
                avg_score=float(avg_score)
            )

        return RetrievalMetrics(
            hit_rate=cls.calculate_hit_rate(retrieved_docs, relevant_pages),
            mrr=cls.calculate_mrr(retrieved_docs, relevant_pages),
            recall_at_k=cls.calculate_recall_at_k(retrieved_docs, relevant_pages, k),
            precision_at_k=cls.calculate_precision_at_k(retrieved_docs, relevant_pages, k),
            ndcg=0.0,  # Simplified - requires full relevance judgments
            avg_score=float(avg_score)
        )
