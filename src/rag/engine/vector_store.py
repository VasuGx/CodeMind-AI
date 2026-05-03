from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class CodeVectorStore:
    def __init__(self, embeddings):
        """
        Takes a LangChain embeddings object and initializes a FAISS store.
        """
        self.embeddings = embeddings
        self.vector_store = None

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Takes a list of enriched chunks and adds them to the FAISS vector store.
        Each chunk is a dict with 'content' and 'metadata'.
        """
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)

        if not documents:
            return

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches the store and returns a list of results with content and metadata.
        """
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=top_k)
        
        output = []
        for res in results:
            output.append({
                "content": res.page_content,
                "metadata": res.metadata
            })
        return output

    def save(self, path: str):
        if self.vector_store:
            self.vector_store.save_local(path)

    def load(self, path: str):
        self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
