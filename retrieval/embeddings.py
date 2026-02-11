"""
Embedding Model Module.
FAANG Pattern: Adapter Pattern - abstracts embedding provider.
"""
from typing import List, Union, Optional
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EmbeddingResult:
    """Result container for embedding operations."""
    embeddings: np.ndarray
    dimension: int
    count: int
    model_name: str


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed multiple documents."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Production embedding model using Sentence Transformers.

    FAANG Patterns:
    - Lazy loading: Model loaded on first use
    - Device detection: Auto GPU/CPU selection
    - Batch processing: Efficient for large document sets
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Compute device (auto-detected if None)
            batch_size: Batch size for document encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device
        self._dimension = None

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = SentenceTransformer(self.model_name, device=self._device)
            # Get dimension from a test embedding
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_embedding.shape[1]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._dimension

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            text: Query string to embed

        Returns:
            Numpy array of shape (1, dimension)
        """
        self._load_model()
        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed multiple documents with batching.

        Args:
            texts: List of document strings

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        self._load_model()

        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                dimension=self.dimension,
                count=0,
                model_name=self.model_name
            )

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100
        )

        return EmbeddingResult(
            embeddings=embeddings,
            dimension=self.dimension,
            count=len(texts),
            model_name=self.model_name
        )
