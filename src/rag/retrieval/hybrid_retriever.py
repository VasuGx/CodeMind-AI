from typing import List, Dict, Any
from src.schemas.rag_schemas import QueryAnalysis

class HybridRetriever:
    def __init__(self, vector_store, all_chunks: List[Dict[str, Any]]):
        """
        Initializes the HybridRetriever.
        vector_store: An instance of CodeVectorStore.
        all_chunks: The complete list of enriched chunks for keyword matching.
        """
        self.vector_store = vector_store
        self.all_chunks = all_chunks

    def retrieve(self, analysis: QueryAnalysis, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval: Semantic (FAISS) + Keyword Search.
        """
        # 1. Semantic Retrieval
        semantic_results = self.vector_store.search(analysis.raw_query, top_k=top_k * 2)
        
        # 2. Keyword Retrieval
        keyword_results = self._keyword_search(analysis)
        
        # 3. Merge and Deduplicate
        # We prioritize keyword matches if they exist, then fill with semantic results
        seen_ids = set()
        combined = []
        
        # Add keyword results first (often higher precision for specific errors)
        for res in keyword_results:
            chunk_id = self._get_chunk_id(res)
            if chunk_id not in seen_ids:
                combined.append(res)
                seen_ids.add(chunk_id)
        
        # Add semantic results
        for res in semantic_results:
            chunk_id = self._get_chunk_id(res)
            if chunk_id not in seen_ids:
                combined.append(res)
                seen_ids.add(chunk_id)
                
        return combined[:top_k]

    def _keyword_search(self, analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """
        Scans all chunks for exact matches in name, keywords, or file hints.
        """
        matches = []
        
        search_terms = set(analysis.keywords)
        if analysis.file_hint:
            search_terms.add(analysis.file_hint)
        if analysis.function_hint:
            search_terms.add(analysis.function_hint)
            
        if not search_terms:
            return []

        for chunk in self.all_chunks:
            metadata = chunk["metadata"]
            
            # Check for name match (High priority)
            if analysis.function_hint and metadata.get("name") == analysis.function_hint:
                matches.append(chunk)
                continue
                
            # Check for file match
            if analysis.file_hint and analysis.file_hint in metadata.get("file_path", ""):
                matches.append(chunk)
                continue
                
            # Check for keyword overlap
            chunk_keywords = set(metadata.get("keywords", []))
            if search_terms.intersection(chunk_keywords):
                matches.append(chunk)
                
        return matches

    def _get_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """Helper to create a unique ID for a chunk to prevent duplicates."""
        meta = chunk["metadata"]
        return f"{meta.get('file_path')}:{meta.get('name')}:{meta.get('line_start')}"
