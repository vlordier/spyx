"""
spyx_mlx — MLX-native spiking neural network library.
API mirrors spyx but runs natively on Apple Silicon via MLX.
"""

from . import axn, data, fn, nn
from ._version import __version__

try:
    from . import experimental, loaders, nir
except Exception:  # optional JAX-backed compatibility modules
    experimental = None
    loaders = None
    nir = None

__all__ = [
    "__version__",
    "axn",
    "data",
    "experimental",
    "fn",
    "loaders",
    "nir",
    "nn",
]
