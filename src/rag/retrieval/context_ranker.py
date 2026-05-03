import re
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
            content = chunk["content"].lower()
            
            # A. Boost for function/name match (Highest Priority)
            if analysis.function_hint and metadata.get("name") == analysis.function_hint:
                score += 100
            
            # B. Boost for file match
            if analysis.file_hint and analysis.file_hint in metadata.get("file_path", ""):
                score += 60
                
            # C. Boost for keyword overlap in metadata
            chunk_keywords = set(metadata.get("keywords", []))
            overlap = set(analysis.keywords).intersection(chunk_keywords)
            score += len(overlap) * 10
            
            # D. Proximity Scoring (Exact error phrases in content)
            # If the raw query (e.g. "NoneType has no attribute resources") is in content
            if analysis.error_type and analysis.error_type.lower() in content:
                score += 40
                
            # E. Line Number Match (If the query contains "line 42")
            line_match = re.search(r"line (\d+)", analysis.raw_query.lower())
            if line_match:
                line_num = int(line_match.group(1))
                if metadata.get("line_start", 0) <= line_num <= metadata.get("line_end", 0):
                    score += 80
            
            scored_chunks.append((score, chunk))

        # 2. Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Pruning / Token Filtering
        total_chars = 0
        max_chars = max_tokens * 4
        
        filtered = []
        for score, chunk in scored_chunks:
            chunk_len = len(chunk["content"])
            if total_chars + chunk_len <= max_chars:
                filtered.append(chunk)
                total_chars += chunk_len
            else:
                continue
                
        return filtered
