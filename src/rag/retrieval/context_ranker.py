from typing import List, Dict, Any
from src.schemas.rag_schemas import QueryAnalysis

class ContextRanker:
    def __init__(self):
        pass

    def rank_and_filter(self, chunks: List[Dict[str, Any]], analysis: QueryAnalysis, max_tokens: int = 4000) -> List[Dict[str, Any]]:
        """
        Ranks chunks based on metadata signals and filters them to fit within a token limit.
        """
        if not chunks:
            return []

        # 1. Scoring
        scored_chunks = []
        for chunk in chunks:
            score = 0
            metadata = chunk["metadata"]
            
            # Boost for function/name match (Highest Priority)
            if analysis.function_hint and metadata.get("name") == analysis.function_hint:
                score += 50
            
            # Boost for file match
            if analysis.file_hint and analysis.file_hint in metadata.get("file_path", ""):
                score += 30
                
            # Boost for keyword overlap
            chunk_keywords = set(metadata.get("keywords", []))
            overlap = set(analysis.keywords).intersection(chunk_keywords)
            score += len(overlap) * 5
            
            scored_chunks.append((score, chunk))

        # 2. Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Pruning / Token Filtering
        # Approximation: 1 token ~= 4 characters
        total_chars = 0
        max_chars = max_tokens * 4
        
        filtered = []
        for score, chunk in scored_chunks:
            chunk_len = len(chunk["content"])
            if total_chars + chunk_len <= max_chars:
                filtered.append(chunk)
                total_chars += chunk_len
            else:
                # If a chunk is too big, we just skip it and keep looking for smaller ones 
                # (or we could stop here, but skipping allows better density)
                continue
                
        return filtered
