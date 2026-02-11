"""
Document Chunking Module.
FAANG Pattern: Strategy Pattern for different chunking approaches.
"""
from typing import List, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ChunkingStrategy(Protocol):
    """Protocol defining the chunking interface."""
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        ...


@dataclass
class ChunkingResult:
    """Result container for chunking operations."""
    chunks: List[Document]
    original_count: int
    chunk_count: int
    avg_chunk_size: float


class RecursiveChunker:
    """
    Recursive character-based text splitter.

    FAANG Pattern:
    - Configurable via constructor injection
    - Immutable configuration after initialization
    - Metrics tracking for observability
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Initialize the chunker with configuration.

        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between consecutive chunks
            separators: Priority list of separators
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )

    def chunk(self, documents: List[Document]) -> ChunkingResult:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            ChunkingResult with chunks and metrics
        """
        if not documents:
            return ChunkingResult(
                chunks=[],
                original_count=0,
                chunk_count=0,
                avg_chunk_size=0.0
            )

        chunks = self._splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        # Calculate metrics
        total_size = sum(len(c.page_content) for c in chunks)
        avg_size = total_size / len(chunks) if chunks else 0

        return ChunkingResult(
            chunks=chunks,
            original_count=len(documents),
            chunk_count=len(chunks),
            avg_chunk_size=round(avg_size, 2)
        )
