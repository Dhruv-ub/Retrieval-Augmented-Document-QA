"""
Unit tests for Evaluation Layer.
"""
import sys
import os

# Handle both direct execution and exec() context
if '__file__' in dir():
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
else:
    sys.path.insert(0, os.getcwd())

from evaluation.hallucination_checks import HallucinationGuard


def test_grounding_high_overlap():
    """Test grounding with high overlap."""
    answer = "The cat sat on the mat."
    context_docs = [{"content": "The cat sat on the mat in the house."}]

    score = HallucinationGuard.check_grounding(answer, context_docs)
    assert score > 0.5, f"Expected high score, got {score}"
    print("test_grounding_high_overlap passed")


def test_grounding_low_overlap():
    """Test grounding with low overlap (hallucination)."""
    answer = "The elephant flew to the moon yesterday."
    context_docs = [{"content": "The cat sat on the mat in the house."}]

    score = HallucinationGuard.check_grounding(answer, context_docs)
    assert score < 0.5, f"Expected low score, got {score}"
    print("test_grounding_low_overlap passed")


def test_grounding_empty_answer():
    """Test grounding with empty answer."""
    answer = ""
    context_docs = [{"content": "Some context here."}]

    score = HallucinationGuard.check_grounding(answer, context_docs)
    assert score == 1.0  # Empty answer is considered grounded
    print("test_grounding_empty_answer passed")


def test_analyze_comprehensive():
    """Test comprehensive analysis."""
    answer = "Machine learning is a subset of artificial intelligence."
    context_docs = [{"content": "Machine learning is a subset of artificial intelligence that enables systems to learn."}]

    result = HallucinationGuard.analyze(answer, context_docs)
    assert result.confidence in ['high', 'medium', 'low']
    assert 0 <= result.grounding_score <= 1
    print("test_analyze_comprehensive passed")


def run_all_tests():
    """Run all evaluation tests."""
    test_grounding_high_overlap()
    test_grounding_low_overlap()
    test_grounding_empty_answer()
    test_analyze_comprehensive()
    print("\nAll evaluation tests passed!")


if __name__ == "__main__":
    run_all_tests()
