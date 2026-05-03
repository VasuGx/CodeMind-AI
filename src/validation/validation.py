import ast
import tempfile
import subprocess
import os
import sys

class ValidationResult:
    def __init__(self, is_valid: bool, errors: str = ""):
        self.is_valid = is_valid
        self.errors = errors

class CodeValidator:
    @staticmethod
    def validate(code_snippet: str) -> ValidationResult:
        """
        Runs a syntax check and a basic lint check on the provided code snippet.
        """
        # 1. Syntax Check (AST)
        try:
            ast.parse(code_snippet)
        except SyntaxError as e:
            return ValidationResult(False, f"Syntax Error: {e}")
            
        # 2. Lint Check (Flake8 via temp file)
        # We only check for critical errors like undefined names (F821) and syntax (E999, F401 etc)
        # We ignore formatting rules (E501 etc) because LLM code doesn't need to be perfectly PEP8 formatted to work.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(code_snippet)
            temp_path = temp_file.name
            
        try:
            # Run flake8 selecting only critical rules via python module
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "--select=E9,F63,F7,F82", temp_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Flake8 found critical issues
                return ValidationResult(False, f"Static Validation Errors:\n{result.stdout}")
                
            return ValidationResult(True, "Code passed syntax and static validation.")
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
