"""
Module for managing repository lifecycles including cloning, indexing, and documentation attachment.
"""
import os
import hashlib
from git import Repo
from typing import Dict, Optional, List
from src.rag.production_pipeline import ProductionRAG
from src.parsers.repo_loader import RepoLoader
from src.parsers.doc_ingestion import DocIngestor

class RepoManager:
    """
    Orchestrates the ingestion of codebases. Handles mapping between repo IDs and RAG instances.
    """
    def __init__(self, llm, storage_dir: str = "indexed_repos"):
        self.llm = llm
        self.storage_dir = storage_dir
        self.repos: Dict[str, ProductionRAG] = {}
        self.repo_metadata: Dict[str, int] = {}
        
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def load_repo(self, repo_url: Optional[str] = None, local_path: Optional[str] = None) -> str:
        """
        Clones (if URL) or loads a local repo, then initializes a ProductionRAG pipeline.
        Returns a deterministic repo_id based on the source path/URL.
        """
        if repo_url:
            path_to_index = self._clone_repo(repo_url)
            repo_id = hashlib.md5(repo_url.encode()).hexdigest()
        elif local_path:
            path_to_index = local_path
            repo_id = hashlib.md5(local_path.encode()).hexdigest()
        else:
            raise ValueError("Either repo_url or local_path must be provided.")

        if repo_id in self.repos:
            return repo_id

        loader = RepoLoader(path_to_index)
        file_paths = loader.get_files(extensions=['.py'])
        
        code_files = []
        for p in file_paths:
            content = loader.read_file(p)
            rel_path = os.path.relpath(p, path_to_index).replace("\\", "/")
            code_files.append({"path": rel_path, "content": content})
        
        doc_ingestor = DocIngestor(path_to_index)
        doc_files = doc_ingestor.get_docs()

        rag = ProductionRAG(llm=self.llm)
        rag.initialize_repo(code_files, doc_files)
        
        self.repos[repo_id] = rag
        self.repo_metadata[repo_id] = len(code_files)
        
        return repo_id

    def load_external_docs(self, repo_id: str, doc_name: str, content: str):
        """Injects independent documentation strings into an existing RAG index."""
        rag = self.get_rag(repo_id)
        if not rag:
            raise ValueError("Repo not loaded yet. Load repo before attaching docs.")
        
        rag.doc_store.create_from_docs([{"source": doc_name, "content": content}])

    def _clone_repo(self, url: str) -> str:
        """Internal helper to clone GitHub repositories."""
        repo_name = url.split("/")[-1].replace(".git", "")
        clone_path = os.path.join(self.storage_dir, repo_name)
        if os.path.exists(clone_path):
            return clone_path
        Repo.clone_from(url, clone_path)
        return clone_path

    def get_rag(self, repo_id: str) -> Optional[ProductionRAG]:
        """Retrieves the RAG instance for a specific repository."""
        return self.repos.get(repo_id)

    def get_file_count(self, repo_id: str) -> int:
        """Returns the number of indexed files for a repository."""
        return self.repo_metadata.get(repo_id, 0)
