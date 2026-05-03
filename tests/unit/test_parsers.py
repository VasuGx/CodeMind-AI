import os
from pathlib import Path
from src.parsers.repo_loader import RepoLoader
from src.parsers.parser import CodeParser
from src.parsers.doc_ingestion import DocIngestor

def test_repo_loader():
    loader = RepoLoader(".")
    files = loader.get_files(extensions=['.py'])
    assert len(files) > 0

def test_doc_ingestor():
    ingestor = DocIngestor(".")
    docs = ingestor.get_docs()
    # It shouldn't crash, and returns a list
    assert isinstance(docs, list)
