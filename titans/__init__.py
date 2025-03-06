"""
Titans: Learning to Memorize at Test Time

This package implements the Titans architecture as described in the research paper,
providing models that can learn to memorize during inference.

The package includes three architectural variants:
- MAC (Memory as Context): Uses memory as context for attention
- MAG (Memory as Gating): Uses memory with sliding window attention
- MAL (Memory as Layer): Uses memory as an additional transformation layer

Each variant represents a different approach to incorporating neural memory
into transformer-based architectures.
"""

from .models.titans_base import TitansBase
from .models.titans_mac import TitansMAC
from .models.titans_mag import TitansMAG
from .models.titans_mal import TitansMAL

__all__ = [
    'TitansBase',
    'TitansMAC',
    'TitansMAG',
    'TitansMAL',
] 