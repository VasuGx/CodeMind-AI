import ast
from typing import List, Dict, Any, Set

class MetadataEnricher:
    def __init__(self):
        pass

    @staticmethod
    def enrich_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of chunks and adds 'keywords', 'internal_calls', and 'relevant_imports'.
        """
        # 1. Collect all "known" internal names in this file (to identify internal calls)
        internal_names = {c["metadata"]["name"] for c in chunks}
        
        for chunk in chunks:
            content = chunk["content"]
            try:
                # We wrap in a dummy function if it's a method to ensure it's parsable 
                # (though usually chunks are already full FunctionDefs/ClassDefs)
                tree = ast.parse(content)
            except SyntaxError:
                # If we can't parse the chunk alone (e.g. it's hollowed out too much), 
                # we just skip enrichment for this one or use basic string matching.
                chunk["metadata"]["keywords"] = []
                chunk["metadata"]["internal_calls"] = []
                continue

            keywords = set()
            calls = set()
            
            for node in ast.walk(tree):
                # Extract Variables and Arguments as keywords
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, (ast.Store, ast.Param)):
                        keywords.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    keywords.add(node.name)
                    if hasattr(node, 'args'):
                        for arg in node.args.args:
                            keywords.add(arg.arg)

                # Extract Calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)

            # Filter calls to only those that exist in this file
            chunk["metadata"]["internal_calls"] = list(calls.intersection(internal_names))
            
            # Clean up keywords
            chunk["metadata"]["keywords"] = list(keywords)
            
            # Smart Imports: Which of the module imports are actually referenced here?
            all_imports = chunk["metadata"].get("imports", [])
            used_imports = []
            for imp in all_imports:
                # e.g. "import os" -> "os", "from x import y" -> "y"
                imp_parts = imp.split()
                if imp_parts:
                    last_part = imp_parts[-1]
                    if last_part in content:
                        used_imports.append(imp)
            
            chunk["metadata"]["relevant_imports"] = used_imports

        return chunks
