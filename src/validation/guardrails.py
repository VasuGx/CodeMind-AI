"""
Module for security, style, and confidence auditing of generated code fixes.
"""
import re
from typing import List, Dict, Any

class SecurityScanner:
    """Detects dangerous code patterns in AI-generated snippets."""
    DANGEROUS_PATTERNS = [
        r"eval\(", r"exec\(", r"os\.system\(", r"subprocess\.call\(", 
        r"shutil\.rmtree\(", r"os\.remove\(", r"os\.chmod\("
    ]

    @staticmethod
    def scan(code: str) -> Dict[str, Any]:
        """Scans code for potentially malicious or unsafe snippets."""
        found_patterns = []
        for pattern in SecurityScanner.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                found_patterns.append(pattern.replace("\\", ""))
        
        return {
            "is_safe": len(found_patterns) == 0,
            "blocked_patterns": found_patterns,
            "risk_level": "high" if found_patterns else "low"
        }

class StyleEnforcer:
    """Verifies that generated code follows project coding standards."""
    @staticmethod
    def check_conventions(code: str, doc_context: str) -> Dict[str, Any]:
        """Heuristic check for type hints and naming conventions."""
        has_type_hints = "->" in code or ":" in code
        is_snake_case = not re.search(r"[a-z]+[A-Z]+[a-z]+", code)
        
        return {
            "follows_type_hints": has_type_hints,
            "is_snake_case": is_snake_case,
            "score": 1.0 if (has_type_hints and is_snake_case) else 0.5
        }

class ConfidenceCalculator:
    """Calculates a trust score for the proposed solution."""
    @staticmethod
    def calculate(consensus: bool, 
                  retry_count: int, 
                  rag_top_k: int, 
                  validation_passed: bool) -> float:
        """
        Combines model agreement, RAG stability, and execution results into a single metric.
        """
        score = 0.0
        if consensus: score += 0.4
        
        if validation_passed and retry_count == 0:
            score += 0.3
        elif validation_passed and retry_count == 1:
            score += 0.15
            
        if rag_top_k <= 5:
            score += 0.3
        elif rag_top_k <= 10:
            score += 0.15
            
        return round(score, 2)
