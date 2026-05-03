from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.schemas.rag_schemas import QueryAnalysis

class HybridRetriever:
    def __init__(self, vector_store, all_chunks: List[Dict[str, Any]]):
        """
        Initializes the HybridRetriever.
        vector_store: An instance of CodeVectorStore.
        all_chunks: The complete list of enriched chunks for hybrid matching.
        """
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        
        # Initialize BM25 with safety check
        if all_chunks:
            self.tokenized_corpus = [self._tokenize(c["content"]) for c in all_chunks]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.tokenized_corpus = []
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for code and text."""
        processed = text.replace("(", " ").replace(")", " ").replace(".", " ").replace("_", " ").replace(":", " ")
        return processed.lower().split()

    def retrieve(self, analysis: QueryAnalysis, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval: Semantic (FAISS) + BM25 Keyword Search.
        """
        # 1. Semantic Retrieval (FAISS)
        semantic_results = self.vector_store.search(analysis.raw_query, top_k=top_k * 2)
        
        # 2. BM25 Retrieval (Keyword) - only if index exists
        bm25_results = []
        if self.bm25:
            tokenized_query = self._tokenize(analysis.raw_query)
            bm25_results = self.bm25.get_top_n(tokenized_query, self.all_chunks, n=top_k * 2)
        
        # 3. Metadata Heuristic Search (High Priority matches)
        heuristic_results = self._heuristic_search(analysis)
        
        # 4. Merge and Deduplicate
        seen_ids = set()
        combined = []
        
        for results_list in [heuristic_results, bm25_results, semantic_results]:
            for res in results_list:
                chunk_id = self._get_chunk_id(res)
                if chunk_id not in seen_ids:
                    combined.append(res)
                    seen_ids.add(chunk_id)
        
        return combined[:top_k]

    def _heuristic_search(self, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        matches = []
        for chunk in self.all_chunks:
            metadata = chunk["metadata"]
            if analysis.function_hint and metadata.get("name") == analysis.function_hint:
                matches.append(chunk)
                continue
            if analysis.file_hint and analysis.file_hint in metadata.get("file_path", ""):
                matches.append(chunk)
        return matches

    def _get_chunk_id(self, chunk: Dict[str, Any]) -> str:
        meta = chunk["metadata"]
        return f"{meta.get('file_path')}:{meta.get('name')}:{meta.get('line_start')}"
