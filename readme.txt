# Multi-Agent Code Debugging System (Production RAG Refactor)

## Overview
This system is a production-grade multi-agent orchestration framework designed for repository analysis and autonomous code debugging. It features a sophisticated RAG pipeline that understands code structure and documentation.

## Key Features
- **Semantic Code Chunking:** Uses AST parsing to slice code into logical method/class boundaries.
- **Metadata Enrichment:** Automatically extracts internal call graphs, keywords, and local imports for every code chunk.
- **Hybrid Retrieval:** Combines FAISS semantic search with exact-match keyword searching.
- **Context Ranking:** Heuristic-based scoring to prioritize the most relevant code and documentation.
- **Validation Layer:** Integrated static analysis and syntax verification for AI-generated fixes.
- **Multi-Agent Consensus:** Uses an Arbiter pattern to evaluate and merge suggestions from multiple LLM models.

## Current Architecture
- `src/rag/`: The core retrieval pipeline components.
- `src/agents/`: Agent logic for analysis and fix generation.
- `src/validation/`: Code integrity and linting gatekeepers.
- `src/parsers/`: Repository and documentation ingestion.
- `tests/unit/`: Comprehensive suite of 15+ unit tests.

## Tech Stack
- Python 3.12
- LangChain / LangGraph
- Groq (LLM Inference)
- FAISS (Vector DB)
- HuggingFace (Local Embeddings)
- Pytest (Testing)
