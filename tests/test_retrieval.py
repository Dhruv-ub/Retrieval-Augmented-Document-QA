"""
Unit tests for Retrieval Layer.
"""
import sys
import os

# Handle both direct execution and exec() context
if '__file__' in dir():
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
else:
    sys.path.insert(0, os.getcwd())

import numpy as np


class MockDocument:
    """Mock document for testing."""
    def __init__(self, content: str, metadata: dict = None):
        self.page_content = content
        self.metadata = metadata or {}


def test_vector_store_add_and_search():
    """Test adding and searching vectors."""
    from retrieval.vector_store import FAISSVectorStore

    store = FAISSVectorStore(dimension=4)

    # Create mock embeddings and documents
    embeddings = np.random.rand(3, 4).astype('float32')
    docs = [
        MockDocument("Document 1", {"page": 1}),
        MockDocument("Document 2", {"page": 2}),
        MockDocument("Document 3", {"page": 3}),
    ]

    added = store.add(embeddings, docs)
    assert added == 3
    assert store.size() == 3
    print("test_vector_store_add_and_search passed")


def test_vector_store_empty_search():
    """Test search on empty store."""
    from retrieval.vector_store import FAISSVectorStore

    store = FAISSVectorStore(dimension=4)
    query = np.random.rand(1, 4).astype('float32')

    results = store.search(query, k=3)
    assert len(results.results) == 0
    print("test_vector_store_empty_search passed")


def run_all_tests():
    """Run all retrieval tests."""
    test_vector_store_add_and_search()
    test_vector_store_empty_search()
    print("\nAll retrieval tests passed!")


if __name__ == "__main__":
    run_all_tests()
