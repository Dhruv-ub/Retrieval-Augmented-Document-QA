import os
import sys

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.load_docs import DocumentLoader
from ingestion.chunk_docs import DocumentChunker
from ingestion.embed_docs import EmebeddingGenerator
from retrieval.vector_store import VectorStore
from retrieval.search import SearchEngine
from generation.prompt import PromptEngineering
from generation.llm import LLMService

def run_experiment():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=== PHASE 1: DATA PREP ===")
    # 1. Load
    raw_path = os.path.join(base_dir, "data", "raw_docs")
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print(f"Created {raw_path}. Please add PDFs/TXTs there.")
        return

    loader = DocumentLoader(raw_path)
    docs = loader.load_documents()
    
    if not docs:
        print("No documents found. Please add files to data/raw_docs/")
        # create valid dummy doc for test
        docs = [{"text": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn' from data.", "source": "dummy.txt", "page": 1}]
        print("Using dummy data for test.")

    # 2. Chunk
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)
    
    # 3. Embed & Index
    print("\n=== PHASE 2: INDEXING ===")
    embed_dir = os.path.join(base_dir, "embeddings")
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
        
    gen = EmebeddingGenerator()
    embeddings = gen.generate_embeddings(chunks)
    
    index_path = os.path.join(embed_dir, "faiss_index.index")
    metadata_path = os.path.join(embed_dir, "metadata.pkl")
    store = VectorStore(index_path, metadata_path)
    store.build_index(embeddings, chunks)
    store.save_index()
    
    # 4. Search & Generate
    print("\n=== PHASE 3: RAG PIPELINE ===")
    search_engine = SearchEngine()
    llm_service = LLMService()
    
    queries = [
        "What is machine learning?",
        "Explain the transformer architecture."
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        retrieved = search_engine.search(q, k=3)
        print(f"Retrieved {len(retrieved)} chunks.")
        
        prompt = PromptEngineering.build_prompt(q, retrieved)
        response = llm_service.generate_response(prompt)
        
        print(f"Answer: {response['answer']}")
        print(f"Latency: {response['latency_ms']:.2f}ms")

if __name__ == "__main__":
    run_experiment()
