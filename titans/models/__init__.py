"""
Model implementations for the Titans architecture.

This package contains the three main architectural variants described in Section 3
of the research paper, each representing a different approach to incorporating
neural memory into transformer-based architectures:

- TitansBase: Base class that provides shared components for all variants.

- TitansMAC (Memory as Context): As described in Section 3.1, treats memory as
  context for the attention mechanism by concatenating persistent memory tokens,
  long-term memory output, and the current context.

- TitansMAG (Memory as Gating): As described in Section 3.2, uses sliding window
  attention as short-term memory and neural memory as long-term memory, combining
  them with a gating mechanism.

- TitansMAL (Memory as Layer): As described in Section 3.3, uses memory as an
  additional transformation layer in the transformer architecture.
"""

from .titans_base import TitansBase
from .titans_mac import TitansMAC
from .titans_mag import TitansMAG
from .titans_mal import TitansMAL

__all__ = [
    'TitansBase',
    'TitansMAC',
    'TitansMAG',
    'TitansMAL',
] 