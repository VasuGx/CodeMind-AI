from typing import List, Dict, Any
from src.schemas.rag_schemas import QueryAnalysis

class GlobalRetriever:
    def __init__(self, code_retriever, doc_store):
        """
        GlobalRetriever manages both code and documentation searches.
        code_retriever: Instance of HybridRetriever (handles code).
        doc_store: Instance of DocVectorStore (handles docs).
        """
        self.code_retriever = code_retriever
        self.doc_store = doc_store

    def retrieve_all(self, 
                     analysis: QueryAnalysis, 
                     top_k: int = 5, 
                     code_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieves results from both code and documentation, merged by relevance.
        """
        # 1. Retrieve from code (Semantic + Keyword)
        code_results = self.code_retriever.retrieve(analysis, top_k=top_k)
        for res in code_results:
            res["metadata"]["origin"] = "code"
            res["final_score"] = code_weight # Simple weighting
            
        # 2. Retrieve from documentation
        doc_results = []
        if self.doc_store and self.doc_store.vector_store:
            # We use the raw search method from FAISS
            raw_docs = self.doc_store.vector_store.similarity_search(analysis.raw_query, k=top_k)
            for d in raw_docs:
                doc_results.append({
                    "content": d.page_content,
                    "metadata": {**d.metadata, "origin": "documentation"},
                    "final_score": 1.0 - code_weight
                })

        # 3. Merge and sort
        combined = code_results + doc_results
        # In a production system, we would re-rank here. 
        # For now, we prioritize based on weight and origin.
        combined.sort(key=lambda x: x["final_score"], reverse=True)
        
        return combined[:top_k]
