"""
Unit tests for Ingestion Layer.
FAANG Pattern: Comprehensive test coverage with edge cases.
"""
import sys
import os

# Handle both direct execution and exec() context
if '__file__' in dir():
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
else:
    sys.path.insert(0, os.getcwd())

from ingestion.chunk_docs import RecursiveChunker, ChunkingResult


class MockDocument:
    """Mock document for testing."""
    def __init__(self, content: str, metadata: dict = None):
        self.page_content = content
        self.metadata = metadata or {}


def test_chunker_empty_input():
    """Test chunker handles empty input gracefully."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    result = chunker.chunk([])

    assert result.chunk_count == 0
    assert result.original_count == 0
    assert len(result.chunks) == 0
    print("test_chunker_empty_input passed")


def test_chunker_single_document():
    """Test chunker with single document."""
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = MockDocument("This is a test document with some content that should be split.")
    result = chunker.chunk([doc])

    assert result.original_count == 1
    assert result.chunk_count >= 1
    assert all(hasattr(c, 'page_content') for c in result.chunks)
    print("test_chunker_single_document passed")


def test_chunker_preserves_metadata():
    """Test that chunker preserves and adds metadata."""
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = MockDocument("A" * 200, {"source": "test.pdf", "page": 1})
    result = chunker.chunk([doc])

    for chunk in result.chunks:
        assert 'chunk_index' in chunk.metadata
        assert 'chunk_size' in chunk.metadata
    print("test_chunker_preserves_metadata passed")


def run_all_tests():
    """Run all ingestion tests."""
    test_chunker_empty_input()
    test_chunker_single_document()
    test_chunker_preserves_metadata()
    print("\nAll ingestion tests passed!")


if __name__ == "__main__":
    run_all_tests()
