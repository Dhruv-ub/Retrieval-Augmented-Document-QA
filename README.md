<h1 align="center">Production-Grade RAG System</h1>

<p align="center">
  <strong>Retrieval-Augmented Generation for ML Research Papers</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg" alt="PyTorch"></a>
  <a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-Vector%20Search-green.svg" alt="FAISS"></a>
  <a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"><img src="https://img.shields.io/badge/LLM-TinyLlama--1.1B-yellow.svg" alt="TinyLlama"></a>
  <a href="https://colab.research.google.com/drive/1sokY8WxhcR_jIjmJSaG3vfgJp8l_k_wp?usp=sharing"><img src="https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?logo=googlecolab" alt="Open In Colab"></a>
</p>

---

## Overview

A modular, end-to-end **Retrieval-Augmented Generation (RAG)** system built with production engineering principles. The system ingests PDF research papers, builds a semantic vector index, retrieves context-relevant passages, and generates grounded answers with hallucination detection — all exposed through both a **Gradio chat UI** and a **FastAPI REST endpoint**.

> **Try it instantly:** [Open the interactive Colab notebook](https://colab.research.google.com/drive/1sokY8WxhcR_jIjmJSaG3vfgJp8l_k_wp?usp=sharing) — no local setup required.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Modular Architecture** | Clean separation into config, ingestion, retrieval, generation, evaluation, and app layers following SOLID principles |
| **Semantic Search** | FAISS-powered vector similarity search with normalized cosine similarity over sentence-transformer embeddings |
| **Grounded Generation** | TinyLlama-1.1B with explicit context-only instruction prompts to minimize hallucination |
| **Hallucination Detection** | Multi-signal grounding analysis: token overlap, n-gram coverage, claim extraction, and confidence calibration |
| **Retrieval Evaluation** | Comprehensive metrics suite — Hit Rate, MRR, Recall@K, Precision@K, NDCG |
| **Structured Logging** | Production-grade observability with request tracing, performance decorators, and component-level isolation |
| **Type-Safe Configuration** | Centralized, immutable dataclass configuration with environment variable overrides |
| **Dual Interface** | Gradio chat UI for interactive use + FastAPI REST API for programmatic access |
| **Comprehensive Tests** | Unit tests for ingestion, retrieval, and evaluation layers with edge case coverage |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│              Gradio Chat UI  ·  FastAPI REST API                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      RAG Pipeline                               │
│                                                                 │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌─────────┐ │
│  │  Query   │──▶│  Retrieve  │──▶│  Generate  │──▶│ Evaluate │ │
│  │ Embedding│   │ Top-K Docs │   │  Answer    │   │ Grounding│ │
│  └──────────┘   └────────────┘   └────────────┘   └─────────┘ │
│       │               │               │               │        │
│       ▼               ▼               ▼               ▼        │
│  SentenceTransf.  FAISS Index   TinyLlama-1.1B  Hallucination │
│  all-MiniLM-L6   (Cosine Sim)   (Float16)        Guard       │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  Offline Ingestion Pipeline                     │
│        Load PDFs  →  Chunk (500 tok)  →  Embed  →  Index       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag-system/
│
├── config/                          # Configuration & Observability
│   ├── __init__.py
│   ├── settings.py                  # Type-safe dataclass configuration (Singleton)
│   └── logger.py                    # Structured logging with performance decorators
│
├── ingestion/                       # Document Ingestion Pipeline
│   ├── __init__.py
│   ├── load_docs.py                 # PDF loading with validation (Single Responsibility)
│   ├── chunk_docs.py                # Recursive text chunking (Strategy Pattern)
│   └── embed_docs.py                # Embedding generation + FAISS index builder
│
├── retrieval/                       # Semantic Retrieval Layer
│   ├── __init__.py
│   ├── embeddings.py                # Sentence-transformer adapter (Adapter Pattern)
│   ├── vector_store.py              # FAISS vector store (Repository Pattern)
│   └── search.py                    # Search engine orchestrator
│
├── generation/                      # LLM Generation Layer
│   ├── __init__.py
│   ├── prompt.py                    # RAG prompt templates (Template Method Pattern)
│   └── llm.py                       # TinyLlama engine — Float16 (Facade Pattern)
│
├── evaluation/                      # Evaluation & Safety
│   ├── __init__.py
│   ├── retrieval_metrics.py         # Hit Rate, MRR, Recall@K, Precision@K, NDCG
│   └── hallucination_checks.py      # Multi-signal grounding & claim analysis
│
├── app/                             # Application Layer
│   ├── __init__.py
│   ├── ui.py                        # Gradio 5.x chat interface
│   └── api.py                       # FastAPI REST endpoint
│
├── tests/                           # Test Suite
│   ├── __init__.py
│   ├── test_ingestion.py            # Chunking tests with edge cases
│   ├── test_retrieval.py            # Vector store add/search tests
│   └── test_evaluation.py           # Grounding & hallucination tests
│
├── notebooks/                       # Experimentation
│   ├── colab_experiment.ipynb
│   └── experiments.ipynb
│
├── data/
│   ├── raw_docs/                    # Input: place PDF files here
│   └── processed_chunks/            # Output: chunks, embeddings, FAISS index
│
├── main.py                          # Main entry point (full pipeline + UI)
├── run_ingestion.py                 # Standalone offline ingestion script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | 384-dim dense embeddings, cosine similarity |
| Vector Store | `FAISS` (IndexFlatIP) | Approximate nearest neighbor search |
| LLM | `TinyLlama-1.1B-Chat` (Float16) | Lightweight instruction-tuned generation |
| Chunking | `LangChain` RecursiveCharacterTextSplitter | Semantic-aware document splitting |
| PDF Parsing | `pypdf` / `LangChain` PyPDFLoader | Robust PDF text extraction |
| UI | `Gradio` 5.x | Interactive chat with metrics dashboard |
| API | `FastAPI` + `Uvicorn` | Production REST endpoint |
| Config | Python `dataclasses` (frozen) | Immutable, type-safe configuration |

---

## Quick Start

### Option 1: Google Colab (Zero Setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sokY8WxhcR_jIjmJSaG3vfgJp8l_k_wp?usp=sharing)

1. Click the badge above to open the notebook
2. Upload a PDF when prompted
3. Run all cells — the Gradio UI will launch with a public share link

### Option 2: Local Installation

#### Prerequisites

- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for accelerated inference

#### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-system.git
cd rag-system
```

#### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Add Your Documents

Place one or more PDF files into the `data/raw_docs/` directory:

```bash
cp /path/to/your/paper.pdf data/raw_docs/
```

#### 5. Launch the System

**Full pipeline with Gradio UI:**

```bash
python main.py --pdf data/raw_docs/your_paper.pdf
```

This will:
1. Load and validate the PDF
2. Chunk the document (500 chars, 50 overlap)
3. Generate embeddings with all-MiniLM-L6-v2
4. Build a FAISS index
5. Load TinyLlama-1.1B
6. Launch an interactive Gradio chat interface

**CLI-only mode (no browser UI):**

```bash
python main.py --pdf data/raw_docs/your_paper.pdf --no-ui
```

**With a public share link (useful for demos):**

```bash
python main.py --pdf data/raw_docs/your_paper.pdf --share
```

#### 6. Alternative: Offline Ingestion + REST API

```bash
# Step 1: Run ingestion pipeline (once)
python run_ingestion.py

# Step 2: Start the FastAPI server
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Query the API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the attention mechanism?", "k": 3}'
```

---

## Running Tests

```bash
# Run all test suites
python -m tests.test_ingestion
python -m tests.test_retrieval
python -m tests.test_evaluation
```

---

## Design Patterns & Engineering Principles

This project deliberately applies patterns commonly evaluated in FAANG system design and coding interviews:

| Pattern | Where Applied | Why |
|---------|--------------|-----|
| **Single Responsibility** | Each module owns exactly one concern | Maintainability, testability |
| **Strategy Pattern** | `ChunkingStrategy` protocol | Swappable chunking algorithms without modifying callers |
| **Adapter Pattern** | `BaseEmbedder` → `SentenceTransformerEmbedder` | Decouple embedding provider from retrieval logic |
| **Repository Pattern** | `BaseVectorStore` → `FAISSVectorStore` | Abstract storage; swap FAISS for Pinecone/Weaviate trivially |
| **Facade Pattern** | `LLMEngine` wraps tokenizer + model + generation | Simple `generate()` interface hides HuggingFace complexity |
| **Template Method** | `BasePromptTemplate` → `RAGPromptTemplate` | Consistent prompt structure with customizable components |
| **Singleton** | `get_config()` in settings | Single source of truth for configuration |
| **Result Pattern** | `LoadResult`, `ChunkingResult`, etc. | Structured error handling without exceptions in business logic |
| **Lazy Loading** | Embedding model + LLM loaded on first call | Fast startup, memory-efficient |

---

## Evaluation Metrics

The system provides two categories of evaluation:

### Retrieval Quality
- **Hit Rate** — Did at least one relevant document appear in top-K?
- **MRR (Mean Reciprocal Rank)** — How early does the first relevant result appear?
- **Recall@K** — What fraction of relevant documents were retrieved?
- **Precision@K** — What fraction of retrieved documents are relevant?

### Generation Faithfulness
- **Token Overlap Score** — Word-level grounding between answer and context
- **N-gram Coverage** — Trigram overlap to detect paraphrased hallucinations
- **Claim Extraction** — Identifies factual statements and cross-checks against context
- **Confidence Calibration** — Classifies answer reliability as High / Medium / Low

---

## Configuration

All settings are centralized in `config/settings.py` using frozen dataclasses:

```python
from config.settings import get_config

config = get_config()
config.embedding.model_name   # "all-MiniLM-L6-v2"
config.chunking.chunk_size     # 500
config.llm.model_id            # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
config.retriever.top_k         # 3
```

Override via environment variables:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

---

## Sample Output

```
Query: "What is the transformer architecture?"

Answer: The transformer architecture relies entirely on self-attention mechanisms,
dispensing with recurrence and convolution. It consists of an encoder-decoder
structure where both components use stacked self-attention and point-wise
fully connected layers.

──────────────────────────────
System Metrics
| Metric            | Value          |
|-------------------|----------------|
| Retrieval Latency | 2.34 ms        |
| Generation Time   | 1847.12 ms     |
| Grounding Score   | 0.82 (High)    |
| Source Pages       | [3, 5, 7]     |
| Flagged Claims     | 0             |
```

---

## Roadmap

- [ ] Hybrid search (dense + BM25 sparse retrieval)
- [ ] Multi-document cross-referencing
- [ ] Streaming token generation in Gradio UI
- [ ] ONNX/TensorRT optimized inference
- [ ] Pinecone / Weaviate cloud vector store adapter
- [ ] Docker containerization + Kubernetes deployment config
- [ ] CI/CD pipeline with automated test + lint gates
- [ ] RAG evaluation benchmarks (RAGAS framework integration)

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built as a production-grade ML engineering portfolio project.</sub>
</p>
