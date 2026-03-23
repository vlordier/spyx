"""Canonical Spyx model template API.

This module re-exports reference model templates from focused implementation modules.
"""

from __future__ import annotations

from . import core as _core
from . import fusion as _fusion
from . import vision as _vision

__all__: list[str] = []


def _export_public(module) -> None:
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if name in globals():
            continue
        globals()[name] = value
        __all__.append(name)


_export_public(_core)
_export_public(_vision)
_export_public(_fusion)

# Keep export order deterministic.
__all__.sort()

del _core

del _vision

del _fusion

del _export_public
