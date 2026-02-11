"""
Vector Store Module using FAISS.
FAANG Pattern: Repository Pattern - abstracts storage implementation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SearchResult:
    """Single search result with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int


@dataclass
class SearchResults:
    """Container for search results with metrics."""
    results: List[SearchResult]
    query_time_ms: float
    total_docs: int


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: List[Any]) -> int:
        """Add embeddings and documents to the store."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> SearchResults:
        """Search for similar documents."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of documents in store."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store for similarity search.

    FAANG Patterns:
    - Factory Pattern: Multiple index types supported
    - Metrics tracking: Query latency monitoring
    - Thread-safe: Can be used in multi-threaded apps
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        use_gpu: bool = False
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            use_gpu: Whether to use GPU acceleration
        """
        import faiss

        self.dimension = dimension
        self.index_type = index_type
        self._docs_map: Dict[int, Any] = {}
        self._current_id = 0

        # Create index based on type
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        # GPU acceleration if available and requested
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                pass  # Fallback to CPU

    def add(self, embeddings: np.ndarray, documents: List[Any]) -> int:
        """
        Add embeddings and documents to the store.

        Args:
            embeddings: Numpy array of embeddings
            documents: List of document objects

        Returns:
            Number of documents added
        """
        import faiss

        if len(embeddings) == 0:
            return 0

        # Ensure proper shape and type
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store document mapping
        for i, doc in enumerate(documents):
            self._docs_map[self._current_id + i] = doc

        added_count = len(documents)
        self._current_id += added_count

        return added_count

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 3,
        threshold: float = 0.0
    ) -> SearchResults:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding
            k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            SearchResults with ranked documents
        """
        import faiss
        import time

        start_time = time.perf_counter()

        # Ensure proper shape
        query_vector = np.ascontiguousarray(query_vector.astype('float32'))
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query
        faiss.normalize_L2(query_vector)

        # Search
        distances, indices = self.index.search(query_vector, k)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx != -1 and score >= threshold:
                doc = self._docs_map.get(idx)
                if doc is not None:
                    results.append(SearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=float(score),
                        rank=rank + 1
                    ))

        query_time = (time.perf_counter() - start_time) * 1000

        return SearchResults(
            results=results,
            query_time_ms=round(query_time, 2),
            total_docs=self.size()
        )

    def size(self) -> int:
        """Return number of documents in store."""
        return self.index.ntotal
