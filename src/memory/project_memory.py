import json
import os
import hashlib
import numpy as np
from typing import List, Optional, Tuple
from src.schemas.memory_schemas import MemoryItem

class ProjectMemory:
    def __init__(self, embedder, storage_path: str = "project_memory.json"):
        self.embedder = embedder
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                self.memories = [MemoryItem(**item) for item in data]

    def _save(self):
        with open(self.storage_path, "w") as f:
            json.dump([item.model_dump() for item in self.memories], f, indent=2)

    def add_fix(self, error_description: str, fix: str, context: str, reasoning: str):
        """Stores a new fix in memory with its embedding."""
        error_id = hashlib.md5(error_description.encode()).hexdigest()
        
        # Avoid duplicates
        if any(m.error_id == error_id for m in self.memories):
            return

        embedding = self.embedder.embed_query(error_description)
        
        item = MemoryItem(
            error_id=error_id,
            error_description=error_description,
            retrieved_context=context,
            final_fix=fix,
            reasoning=reasoning,
            embedding=embedding
        )
        
        self.memories.append(item)
        self._save()

    def find_similar_fix(self, error_description: str, threshold: float = 0.85) -> Optional[MemoryItem]:
        """Searches memory for a similar past error."""
        if not self.memories:
            return None

        query_embedding = np.array(self.embedder.embed_query(error_description))
        
        best_match = None
        best_score = -1.0
        
        for item in self.memories:
            if item.embedding is None:
                continue
                
            item_embedding = np.array(item.embedding)
            # Cosine similarity
            score = np.dot(query_embedding, item_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
            )
            
            if score > best_score:
                best_score = score
                best_match = item
                
        if best_score >= threshold:
            print(f"   -> [MEMORY HIT] Similarity: {best_score:.4f}")
            return best_match
            
        return None
