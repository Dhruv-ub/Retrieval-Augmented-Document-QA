"""
Document Loading Module.
FAANG Pattern: Single Responsibility Principle - handles ONLY file I/O.
"""
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import os
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


@dataclass
class LoadResult:
    """Result container for document loading operations."""
    documents: List[Document]
    success: bool
    error_message: Optional[str] = None
    pages_loaded: int = 0


class DocumentLoader:
    """
    Production-grade document loader with error handling.

    FAANG Patterns:
    - Single Responsibility: Only handles loading
    - Result Pattern: Returns structured results, not raw exceptions
    - Validation: Input validation before processing
    """

    SUPPORTED_EXTENSIONS = {'.pdf'}
    MAX_FILE_SIZE_MB = 50

    def __init__(self):
        """Initialize the document loader."""
        pass

    def _validate_file(self, file_path: str) -> Optional[str]:
        """
        Validate file before loading.

        Args:
            file_path: Path to the file

        Returns:
            Error message if validation fails, None otherwise
        """
        path = Path(file_path)

        if not path.exists():
            return f"File not found: {file_path}"

        if not path.is_file():
            return f"Path is not a file: {file_path}"

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return f"Unsupported file type: {path.suffix}. Supported: {self.SUPPORTED_EXTENSIONS}"

        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            return f"File too large: {file_size_mb:.2f}MB > {self.MAX_FILE_SIZE_MB}MB limit"

        return None

    def load_pdf(self, file_path: str) -> LoadResult:
        """
        Load PDF document with comprehensive error handling.

        Args:
            file_path: Path to the PDF file

        Returns:
            LoadResult containing documents and status
        """
        # Validate input
        validation_error = self._validate_file(file_path)
        if validation_error:
            return LoadResult(
                documents=[],
                success=False,
                error_message=validation_error
            )

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Add source metadata
            for doc in documents:
                doc.metadata['source_file'] = str(Path(file_path).name)
                doc.metadata['loader'] = 'PyPDFLoader'

            return LoadResult(
                documents=documents,
                success=True,
                pages_loaded=len(documents)
            )

        except Exception as e:
            return LoadResult(
                documents=[],
                success=False,
                error_message=f"Failed to load PDF: {str(e)}"
            )
