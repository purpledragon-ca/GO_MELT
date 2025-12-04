"""
GO-MELT Simulator Package

This package provides the GO-MELT thermal simulation environment for additive manufacturing.
Main components:
- GoMeltSimulator: High-level simulation class
- go_melt: Main simulation driver function
- Various utility functions for mesh, computation, and visualization
"""

# Import main simulator class
from .simulator import GoMeltSimulator


# Import gcode parsing functions
from .createPath import (
    parsingGcode,
    count_lines,
    format_fixed,
)

# Import visualization functions (gif_creator is in parent directory)
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from gif_creator import (
    find_latest_results_dir,
    create_animation,
    create_overview_animation,
)

# Import all compute functions (these are used extensively throughout)
# Note: This uses wildcard import to match the pattern in go_melt.py
from .computeFunctions import *

# Make main classes/functions available at package level
__all__ = [
    # Main classes
    'GoMeltSimulator'
    
    # Gcode parsing
    'parsingGcode',
    'count_lines',
    'format_fixed',
    
    # Visualization
    'find_latest_results_dir',
    'create_animation',
    'create_overview_animation',
    
    # Note: computeFunctions exports are available via wildcard import
    # but not explicitly listed in __all__ to avoid duplication
]

