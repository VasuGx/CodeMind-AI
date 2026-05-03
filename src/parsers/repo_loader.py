import os
from pathlib import Path
from typing import List

class RepoLoader:
    def __init__(self, repo_path: str, ignore_dirs: List[str] = None):
        self.repo_path = Path(repo_path)
        self.ignore_dirs = ignore_dirs or ['.git', 'venv', '__pycache__', 'node_modules', '.idea']

    def get_files(self, extensions: List[str] = None) -> List[Path]:
        """Recursively find all files matching given extensions."""
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Mutate dirs in-place to avoid walking into ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for filename in filenames:
                if extensions is None or any(filename.endswith(ext) for ext in extensions):
                    files.append(Path(root) / filename)
        return files
        
    def read_file(self, file_path: Path) -> str:
        """Read content of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
