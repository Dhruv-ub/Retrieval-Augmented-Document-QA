"""
Run the full ingestion pipeline: load -> chunk -> embed.
Execute from project root: python run_ingestion.py
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.load_docs import load_documents, save_raw_docs
from ingestion.chunk_docs import chunk_documents, save_chunks
from ingestion.embed_docs import build_and_save_index

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOCS = os.path.join(BASE_DIR, "data", "raw_docs")
PROCESSED = os.path.join(BASE_DIR, "data", "processed_chunks")


def main():
    print("=== Step 1: Load documents ===")
    docs = load_documents(RAW_DOCS)
    if not docs:
        print("No documents found. Add PDF/TXT/MD files to data/raw_docs/")
        return
    save_raw_docs(docs, os.path.join(PROCESSED, "raw_docs.json"))

    print("\n=== Step 2: Chunk documents ===")
    chunks = chunk_documents(docs)
    save_chunks(chunks, os.path.join(PROCESSED, "chunks.json"))

    print("\n=== Step 3: Embed and build index ===")
    build_and_save_index(chunks, PROCESSED)
    print("\nIngestion complete. Run: uvicorn app.api:app --reload")


if __name__ == "__main__":
    main()
