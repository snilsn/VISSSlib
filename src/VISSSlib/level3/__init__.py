"""
Level 3 processing module for VISSS data.

This package contains various level 3 data processing products
including riming retrievals, microphysics calculations, and
other derived quantities from level 2 data.
"""

# Import aux data
from .aux import *

# Import all level3 products
from .combined_riming import *

# Define available level3 products
AVAILABLE_PRODUCTS = {
    "combined_riming": retrieveCombinedRiming,
}

__all__ = ["retrieveCombinedRiming", "AVAILABLE_PRODUCTS"]
