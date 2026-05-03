from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.rag.engine.embedding import Embedder

class DocVectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder.get_embeddings()
        self.vector_store = None

    def create_from_docs(self, docs: List[Dict[str, str]]):
        """
        Creates a FAISS vector store specifically for documentation.
        """
        documents = []
        chunk_size = 1000
        
        for doc_data in docs:
            content = doc_data["content"]
            source = doc_data["source"]
            
            # Simple chunking
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                doc = Document(
                    page_content=chunk, 
                    metadata={"source": source, "type": "documentation"}
                )
                documents.append(doc)

        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embedder)
        else:
            # We don't raise an error because a repo might legitimately have no docs
            self.vector_store = None

    def search(self, query: str, top_k: int = 2) -> str:
        """Searches the doc vector store and returns a formatted string context."""
        if not self.vector_store:
            return "No documentation available."
            
        results = self.vector_store.similarity_search(query, k=top_k)
        
        context = ""
        for res in results:
            context += f"[Doc Source: {res.metadata.get('source')}]\n{res.page_content.strip()}\n\n"
        return context.strip()
