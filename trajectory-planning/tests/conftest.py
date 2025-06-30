import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'

# Add the src directory to the Python path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))