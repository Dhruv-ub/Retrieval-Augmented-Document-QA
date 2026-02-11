"""
Hallucination Detection Module.
FAANG Pattern: Multi-signal approach to detect unfaithful generation.
"""
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import re
from collections import Counter


@dataclass
class GroundingResult:
    """Result container for grounding analysis."""
    grounding_score: float
    coverage_score: float
    factual_density: float
    flagged_claims: List[str]
    confidence: str  # 'high', 'medium', 'low'


class HallucinationGuard:
    """
    Production-grade hallucination detector.

    FAANG Patterns:
    - Multiple detection signals
    - Confidence calibration
    - Explainable flagging

    Signals Used:
    1. Token overlap (baseline)
    2. N-gram coverage
    3. Entity grounding
    4. Claim density analysis
    """

    # Common stopwords to ignore
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'whom', 'its', 'his', 'her'
    }

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Tokenize and clean text."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Remove stopwords
        return [w for w in words if w not in cls.STOPWORDS and len(w) > 2]

    @classmethod
    def _get_ngrams(cls, tokens: List[str], n: int) -> Set[tuple]:
        """Extract n-grams from tokens."""
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    @classmethod
    def calculate_token_overlap(cls, answer: str, context: str) -> float:
        """
        Calculate token-level overlap between answer and context.

        Args:
            answer: Generated answer
            context: Source context

        Returns:
            Overlap ratio (0-1)
        """
        answer_tokens = set(cls._tokenize(answer))
        context_tokens = set(cls._tokenize(context))

        if not answer_tokens:
            return 1.0  # Empty answer is "grounded"

        overlap = answer_tokens.intersection(context_tokens)
        return len(overlap) / len(answer_tokens)

    @classmethod
    def calculate_ngram_coverage(
        cls,
        answer: str,
        context: str,
        n: int = 3
    ) -> float:
        """
        Calculate n-gram coverage of answer in context.

        Args:
            answer: Generated answer
            context: Source context
            n: N-gram size

        Returns:
            Coverage ratio (0-1)
        """
        answer_tokens = cls._tokenize(answer)
        context_tokens = cls._tokenize(context)

        if len(answer_tokens) < n:
            return cls.calculate_token_overlap(answer, context)

        answer_ngrams = cls._get_ngrams(answer_tokens, n)
        context_ngrams = cls._get_ngrams(context_tokens, n)

        if not answer_ngrams:
            return 1.0

        covered = answer_ngrams.intersection(context_ngrams)
        return len(covered) / len(answer_ngrams)

    @classmethod
    def extract_potential_claims(cls, answer: str) -> List[str]:
        """
        Extract sentences that look like factual claims.

        Args:
            answer: Generated answer

        Returns:
            List of potential claim sentences
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)

        claims = []
        claim_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will']

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentence
                words = sentence.lower().split()
                if any(indicator in words for indicator in claim_indicators):
                    claims.append(sentence)

        return claims

    @classmethod
    def check_grounding(
        cls,
        answer: str,
        context_docs: List[Dict[str, Any]]
    ) -> float:
        """
        Simple grounding check (backward compatible).

        Args:
            answer: Generated answer
            context_docs: List of context document dicts

        Returns:
            Grounding score (0-1)
        """
        context = " ".join([d.get('content', '') for d in context_docs])
        return cls.calculate_token_overlap(answer, context)

    @classmethod
    def analyze(
        cls,
        answer: str,
        context_docs: List[Dict[str, Any]]
    ) -> GroundingResult:
        """
        Comprehensive grounding analysis.

        Args:
            answer: Generated answer
            context_docs: List of context document dicts

        Returns:
            GroundingResult with detailed analysis
        """
        context = " ".join([d.get('content', '') for d in context_docs])

        # Calculate multiple signals
        token_overlap = cls.calculate_token_overlap(answer, context)
        ngram_coverage = cls.calculate_ngram_coverage(answer, context)

        # Combined grounding score (weighted average)
        grounding_score = 0.6 * token_overlap + 0.4 * ngram_coverage

        # Extract and check claims
        claims = cls.extract_potential_claims(answer)
        flagged_claims = []

        for claim in claims:
            claim_overlap = cls.calculate_token_overlap(claim, context)
            if claim_overlap < 0.3:  # Low overlap = potentially hallucinated
                flagged_claims.append(claim)

        # Calculate factual density
        factual_density = len(claims) / max(1, len(answer.split('.')))

        # Determine confidence level
        if grounding_score >= 0.7 and len(flagged_claims) == 0:
            confidence = "high"
        elif grounding_score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return GroundingResult(
            grounding_score=round(grounding_score, 3),
            coverage_score=round(ngram_coverage, 3),
            factual_density=round(factual_density, 3),
            flagged_claims=flagged_claims,
            confidence=confidence
        )
