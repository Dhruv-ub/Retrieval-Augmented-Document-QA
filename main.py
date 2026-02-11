"""
Main Application Entry Point.
FAANG Pattern: Clean initialization with proper error handling and logging.

Usage:
    python main.py --pdf path/to/your/document.pdf
"""
import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.load_docs import DocumentLoader
from ingestion.chunk_docs import RecursiveChunker
from retrieval.embeddings import SentenceTransformerEmbedder
from retrieval.vector_store import FAISSVectorStore
from generation.llm import LLMEngine
from generation.prompt import PromptTemplate
from evaluation.hallucination_checks import HallucinationGuard
from evaluation.retrieval_metrics import RetrievalEvaluator


@dataclass
class RAGResponse:
    """Structured response from RAG pipeline."""
    answer: str
    source_pages: List[int]
    grounding_score: float
    retrieval_time_ms: float
    generation_time_ms: float
    retrieved_docs: List[Dict[str, Any]]


def initialize_system(pdf_path: str):
    """
    Initialize the full RAG system.

    Args:
        pdf_path: Path to the PDF document to ingest

    Returns:
        Tuple of (rag_pipeline function, components dict)
    """
    print("=" * 60)
    print("FAANG-Level RAG System - Initialization")
    print("=" * 60)

    # Step 1: Validate Input
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found at {pdf_path}\n"
            f"Please provide a valid PDF path."
        )

    print(f"Found PDF: {pdf_path}")
    print(f"   Size: {os.path.getsize(pdf_path) / 1024:.2f} KB")

    # Step 2: Document Ingestion
    print("\nStage 1: Document Ingestion")
    print("-" * 40)

    loader = DocumentLoader()
    load_result = loader.load_pdf(pdf_path)

    if not load_result.success:
        raise RuntimeError(f"Failed to load PDF: {load_result.error_message}")

    print(f"   Loaded {load_result.pages_loaded} pages")

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunk_result = chunker.chunk(load_result.documents)

    print(f"   Created {chunk_result.chunk_count} chunks")
    print(f"   Avg chunk size: {chunk_result.avg_chunk_size:.0f} chars")

    # Step 3: Build Vector Index
    print("\nStage 2: Vector Indexing")
    print("-" * 40)

    embed_model = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    chunk_texts = [c.page_content for c in chunk_result.chunks]
    embedding_result = embed_model.embed_documents(chunk_texts)

    print(f"   Embedded {embedding_result.count} chunks")
    print(f"   Embedding dimension: {embedding_result.dimension}")

    vector_store = FAISSVectorStore(dimension=embedding_result.dimension)
    added = vector_store.add(embedding_result.embeddings, chunk_result.chunks)

    print(f"   FAISS index built with {added} vectors")

    # Step 4: Load LLM
    print("\nStage 3: LLM Initialization")
    print("-" * 40)

    llm = LLMEngine()
    print("   TinyLlama-1.1B ready")

    # Step 5: Define RAG Pipeline
    print("\nStage 4: Pipeline Configuration")
    print("-" * 40)

    def rag_pipeline(query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Production RAG pipeline with comprehensive metrics.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer and metrics
        """
        # Retrieve relevant documents
        retrieval_start = time.perf_counter()

        query_embedding = embed_model.embed_query(query)
        search_results = vector_store.search(query_embedding, k=top_k)

        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        # Build context from retrieved documents
        retrieved_docs = []
        for result in search_results.results:
            retrieved_docs.append({
                'content': result.content,
                'metadata': result.metadata,
                'score': result.score
            })

        context_str = "\n\n---\n\n".join([
            f"[Source: Page {d['metadata'].get('page', 'N/A')}]\n{d['content']}"
            for d in retrieved_docs
        ])

        source_pages = list(set([
            d['metadata'].get('page', 0)
            for d in retrieved_docs
        ]))

        # Generate answer
        generation_start = time.perf_counter()

        prompt = PromptTemplate.get_rag_prompt(context_str, query)
        answer = llm.generate(prompt, max_new_tokens=256)

        generation_time = (time.perf_counter() - generation_start) * 1000

        # Evaluate grounding (hallucination check)
        grounding_result = HallucinationGuard.analyze(answer, retrieved_docs)

        return {
            'answer': answer,
            'source_pages': source_pages,
            'grounding_score': grounding_result.grounding_score,
            'grounding_confidence': grounding_result.confidence,
            'retrieval_time_ms': round(retrieval_time, 2),
            'generation_time_ms': round(generation_time, 2),
            'num_sources': len(retrieved_docs),
            'avg_retrieval_score': sum(d['score'] for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        }

    print("   RAG pipeline configured")

    return rag_pipeline


def main():
    parser = argparse.ArgumentParser(description="FAANG-Level RAG System")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF document")
    parser.add_argument("--no-ui", action="store_true", help="Skip Gradio UI launch")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    rag_pipeline = initialize_system(args.pdf)

    if args.no_ui:
        # Interactive CLI mode
        print("\n" + "=" * 60)
        print("System Ready! Enter queries (type 'quit' to exit):")
        print("=" * 60)
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query:
                continue
            result = rag_pipeline(query)
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {result['source_pages']}")
            print(f"Grounding: {result['grounding_score']:.2f} ({result['grounding_confidence']})")
    else:
        # Launch Gradio UI
        print("\n" + "=" * 60)
        print("System Ready! Launching UI...")
        print("=" * 60)

        from app.ui import create_ui
        app = create_ui(rag_pipeline)
        app.launch(share=args.share, debug=False)


if __name__ == "__main__":
    main()
