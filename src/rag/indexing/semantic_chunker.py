import ast
from typing import List, Dict, Any, Optional

class SemanticChunker:
    def __init__(self, max_lines: int = 200):
        self.max_lines = max_lines

    def chunk_code(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Parses python code and extracts non-overlapping hierarchical semantic chunks.
        """
        if not content.strip():
            return []

        try:
            tree = ast.parse(content)
        except (SyntaxError, IndentationError):
            return [{
                "content": content,
                "metadata": {
                    "file_path": file_path,
                    "type": "file",
                    "name": "unknown_due_to_syntax_error",
                    "parent": None,
                    "line_start": 1,
                    "line_end": len(content.splitlines()),
                    "docstring": None,
                    "imports": []
                }
            }]

        # 1. Extract all module-level imports
        module_imports = self._extract_imports(tree)

        # 2. Recursive extraction of chunks
        chunks = []
        self._recursive_process(tree, content, file_path, module_imports, chunks)
        
        return chunks

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        return imports

    def _recursive_process(self, 
                          node: ast.AST, 
                          full_content: str, 
                          file_path: str, 
                          imports: List[str], 
                          chunks: List[Dict[str, Any]], 
                          parent_name: Optional[str] = None):
        """
        Recursively processes nodes to extract chunks without duplication.
        """
        # Look for body (Module, ClassDef, FunctionDef all have a body)
        body = getattr(node, 'body', [])
        
        for child in body:
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract segment
                segment = ast.get_source_segment(full_content, child)
                if not segment:
                    continue

                line_count = len(segment.splitlines())
                
                # Check chunk size
                if line_count <= self.max_lines:
                    # Create the hollowed chunk
                    hollow_content = self._hollow_segment(child, segment, full_content)
                    node_type = "class" if isinstance(child, ast.ClassDef) else "function"
                    
                    chunks.append({
                        "content": hollow_content,
                        "metadata": {
                            "file_path": file_path,
                            "type": node_type,
                            "name": child.name,
                            "parent": parent_name,
                            "line_start": child.lineno,
                            "line_end": child.end_lineno,
                            "docstring": ast.get_docstring(child),
                            "imports": imports
                        }
                    })
                
                # Always recurse, even if the parent was too large to be its own chunk
                self._recursive_process(child, full_content, file_path, imports, chunks, child.name)

    def _hollow_segment(self, node: ast.AST, segment: str, full_content: str) -> str:
        """
        Removes the source code of child definitions from the parent's segment.
        """
        child_segments = []
        # Find all direct children that are chunkable
        for child in getattr(node, 'body', []):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                child_src = ast.get_source_segment(full_content, child)
                if child_src:
                    child_segments.append(child_src)
        
        result = segment
        for child_src in child_segments:
            # We use a placeholder that clearly indicates nesting
            placeholder = f"# ... [Nested implementation of {child_src.splitlines()[0].strip()}] ..."
            result = result.replace(child_src, placeholder)
        
        return result
