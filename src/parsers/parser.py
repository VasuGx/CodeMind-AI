import ast
from typing import Dict, Any, List

class CodeParser:
    def parse_python_file(self, content: str, filepath: str) -> Dict[str, Any]:
        """Parses a Python file to extract classes and functions."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"file": filepath, "error": "Syntax Error"}
        except Exception as e:
            return {"file": filepath, "error": str(e)}

        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    "name": node.name,
                    "methods": methods
                })
            # Also capture top-level functions (we will filter below to avoid duplication of methods if needed, 
            # but ast.walk visits everything. Let's just collect all function names).

        # Better approach for just top-level functions:
        top_level_functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        
        return {
            "file": filepath,
            "classes": classes,
            "functions": top_level_functions
        }

    def parse_file(self, content: str, filepath: str) -> Dict[str, Any]:
        """Generic parse file routing based on extension."""
        if filepath.endswith('.py'):
            return self.parse_python_file(content, filepath)
        else:
            return {
                "file": filepath,
                "info": "File parsing not supported for this extension yet."
            }

def summarize_repo(repo_loader, extensions: List[str] = ['.py']) -> Dict[str, Any]:
    parser = CodeParser()
    files = repo_loader.get_files(extensions=extensions)
    
    summary = {
        "repo_path": str(repo_loader.repo_path),
        "total_files_analyzed": len(files),
        "files": []
    }
    
    for file_path in files:
        content = repo_loader.read_file(file_path)
        if content.startswith("Error"):
            summary["files"].append({"file": str(file_path), "error": content})
            continue
            
        file_summary = parser.parse_file(content, str(file_path))
        summary["files"].append(file_summary)
        
    return summary
