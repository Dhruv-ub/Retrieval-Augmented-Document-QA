"""
Convert chunks into vector embeddings and build FAISS index.
INPUT: data/processed_chunks/chunks.json
OUTPUT: data/processed_chunks/embeddings.npy, faiss.index, metadata.pkl
Uses sentence-transformers, normalizes embeddings, FAISS cosine similarity.
"""
import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict


def _get_base_dir() -> str:
    """Resolve base directory dynamically."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings and normalize for cosine similarity."""
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        embeddings = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity (FAISS IndexFlatIP with normalized vectors)
        faiss.normalize_L2(embeddings)
        return embeddings


def build_and_save_index(chunks: List[Dict], output_dir: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Load chunks, generate embeddings, build FAISS index, save everything.
    """
    os.makedirs(output_dir, exist_ok=True)
    texts = [c["text"] for c in chunks]

    generator = EmbeddingGenerator(model_name)
    embeddings = generator.generate_embeddings(texts)

    # Build FAISS index (cosine similarity via Inner Product on normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Metadata: store chunk info for retrieval (text, source, page, chunk_id)
    metadata = [{"text": c["text"], "source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]

    # Save outputs
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved embeddings.npy, faiss.index, metadata.pkl to {output_dir}")
    print(f"Index has {index.ntotal} vectors.")


if __name__ == "__main__":
    base_dir = _get_base_dir()
    chunks_path = os.path.join(base_dir, "data", "processed_chunks", "chunks.json")
    output_dir = os.path.join(base_dir, "data", "processed_chunks")

    if not os.path.exists(chunks_path):
        print("chunks.json not found. Run chunk_docs.py first.")
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        build_and_save_index(chunks, output_dir)
