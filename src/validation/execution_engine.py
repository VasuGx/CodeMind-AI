"""
Runtime Validation Module for verifying code fixes in a secure, isolated environment.
"""
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, Optional

class ExecutionEngine:
    """
    Handles the sandbox execution of Python code snippets.
    Used to verify that proposed fixes are syntactically correct and run without error.
    """
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def validate_fix_runtime(self, 
                            file_path: str, 
                            fixed_code: str, 
                            test_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes code in a temporary directory and captures stdout/stderr.
        
        Args:
            file_path: The virtual path of the file (used for metadata).
            fixed_code: The string content of the Python code to test.
            test_code: Optional additional code to trigger the bug/test.
            
        Returns:
            A dictionary containing success status and any error messages.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_fix.py")
            
            full_code = fixed_code
            if test_code:
                full_code += f"\n\n{test_code}"
            
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(full_code)

            try:
                # Execute using the host's Python interpreter
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    return {"is_valid": True, "output": result.stdout}
                else:
                    return {"is_valid": False, "error_message": result.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"is_valid": False, "error_message": "Execution timed out (Infinite loop?)"}
            except Exception as e:
                return {"is_valid": False, "error_message": str(e)}
