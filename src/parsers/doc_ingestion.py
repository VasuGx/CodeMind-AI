import os
from pathlib import Path
from typing import List, Dict

class DocIngestor:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.doc_extensions = ['.md', '.txt', '.rst']
        self.target_files = ['README.md', 'CONTRIBUTING.md', 'STYLEGUIDE.md']

    def get_docs(self) -> List[Dict[str, str]]:
        """
        Scans for primary documentation files and returns their content.
        """
        docs = []
        for root, _, files in os.walk(self.repo_path):
            # Skip common non-source directories
            if any(part.startswith('.') or part in ['venv', 'node_modules', '__pycache__'] for part in Path(root).parts):
                continue
                
            for file in files:
                if file.upper() in [t.upper() for t in self.target_files] or Path(file).suffix in self.doc_extensions:
                    filepath = Path(root) / file
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                docs.append({
                                    "source": str(filepath.relative_to(self.repo_path)),
                                    "content": content
                                })
                    except UnicodeDecodeError:
                        continue
        return docs
