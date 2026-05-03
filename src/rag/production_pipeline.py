"""
Module for the main RAG (Retrieval Augmented Generation) pipeline.
Handles repository ingestion, metadata enrichment, and multi-stage retrieval.
"""
from typing import List, Dict, Any
from src.rag.indexing.semantic_chunker import SemanticChunker
from src.rag.indexing.metadata_enricher import MetadataEnricher
from src.rag.engine.vector_store import CodeVectorStore
from src.rag.parsing.query_parser import QueryParser
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.context_ranker import ContextRanker
from src.rag.retrieval.doc_retriever import GlobalRetriever
from src.rag.engine.doc_embeddings import DocVectorStore
from src.rag.engine.embedding import Embedder

class ProductionRAG:
    """
    The central hub for code and documentation retrieval.
    Orchestrates chunking, indexing, and the hybrid retrieval flow.
    """
    def __init__(self, llm):
        self.embedder = Embedder()
        self.chunker = SemanticChunker()
        self.enricher = MetadataEnricher()
        self.code_store = CodeVectorStore(self.embedder.get_embeddings())
        self.doc_store = DocVectorStore(self.embedder)
        self.query_parser = QueryParser(llm)
        self.ranker = ContextRanker()
        
        self.all_code_chunks = []
        self.global_retriever = None

    def initialize_repo(self, repo_files: List[Dict[str, str]], doc_files: List[Dict[str, str]]):
        """Processes and indexes a repository for subsequent querying."""
        all_chunks = []
        for file_data in repo_files:
            chunks = self.chunker.chunk_code(file_data["path"], file_data["content"])
            all_chunks.extend(chunks)
        
        self.all_code_chunks = self.enricher.enrich_chunks(all_chunks)
        self.code_store.add_chunks(self.all_code_chunks)
        self.doc_store.create_from_docs(doc_files)
        
        hybrid = HybridRetriever(self.code_store, self.all_code_chunks)
        self.global_retriever = GlobalRetriever(hybrid, self.doc_store)

    def query(self, raw_input: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes a natural language query through the full RAG stack."""
        if not self.global_retriever:
            raise ValueError("RAG Pipeline not initialized. Call initialize_repo first.")

        analysis = self.query_parser.parse(raw_input)
        raw_results = self.global_retriever.retrieve_all(analysis, top_k=top_k * 2)
        final_context = self.ranker.rank_and_filter(raw_results, analysis)
        
        return final_context[:top_k]
