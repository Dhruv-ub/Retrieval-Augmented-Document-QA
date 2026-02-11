"""
RAG API: Query -> search -> prompt -> LLM -> response.
Load index once. Handle queries. Return structured response.
Does NOT ingest documents, rebuild FAISS, or re-embed corpus.
"""
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Lazy-loaded globals
_search_engine = None
_llm_service = None


def get_search_engine():
    global _search_engine
    if _search_engine is None:
        from retrieval.search import SearchEngine
        _search_engine = SearchEngine()
    return _search_engine


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        from generation.llm import LLMService
        _llm_service = LLMService()
    return _llm_service


app = FastAPI(title="RAG System API")


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        search_engine = get_search_engine()
        if search_engine.vector_store.index is None:
            raise HTTPException(
                status_code=503,
                detail="Index not loaded. Run ingestion pipeline first: python -m ingestion.load_docs, chunk_docs, embed_docs"
            )

        chunks = search_engine.search(request.query, k=request.k)

        from generation.prompt import PromptEngineering
        prompt = PromptEngineering.build_prompt(request.query, chunks)

        llm = get_llm_service()
        response = llm.generate_response(prompt)

        sources = list({c["source"] for c in chunks})

        return QueryResponse(
            answer=response["answer"],
            sources=sources,
            latency_ms=response["latency_ms"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
