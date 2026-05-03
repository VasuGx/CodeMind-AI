import sys
import os
from pathlib import Path

# Add project root to path so pytest can find 'src'
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
