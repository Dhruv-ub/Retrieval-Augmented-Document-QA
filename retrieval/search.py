"""
Retrieve top-K relevant chunks for a query.
INPUT: User query (string), FAISS index
OUTPUT: List[{text, source, page, score}]
Embeds query, performs FAISS search. Does NOT call LLM or format final answer.
"""
import os
import sys
import numpy as np
import faiss
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer


def _get_base_dir() -> str:
    """Resolve base directory dynamically."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SearchEngine:
    """Embed query and search FAISS for top-K chunks."""

    _model: SentenceTransformer = None
    _model_name: str = "all-MiniLM-L6-v2"

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(cls._model_name)
        return cls._model

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = _get_base_dir()
        processed_dir = os.path.join(base_dir, "data", "processed_chunks")
        index_path = os.path.join(processed_dir, "faiss.index")
        metadata_path = os.path.join(processed_dir, "metadata.pkl")

        from retrieval.vector_store import VectorStore
        self.vector_store = VectorStore(index_path, metadata_path)
        self.vector_store.load_index()

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Embed query and return top-K chunks."""
        if self.vector_store.index is None:
            return []
        model = self._get_model()
        query_embedding = model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        return self.vector_store.search(query_embedding, k=k)


if __name__ == "__main__":
    engine = SearchEngine()
    if engine.vector_store.index:
        res = engine.search("machine learning attention", k=3)
        for r in res:
            print(f"[{r['score']:.4f}] {r['text'][:80]}... (Source: {r['source']})")
    else:
        print("Index not loaded. Run ingestion pipeline first.")
