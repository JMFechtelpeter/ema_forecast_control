# Make "main.*" resolve to packages that actually live at the project root.
from pathlib import Path

# Point the package search path for 'main' to the project root (the parent of this folder)
__path__ = [str(Path(__file__).resolve().parent.parent)]