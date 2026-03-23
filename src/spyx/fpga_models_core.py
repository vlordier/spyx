"""Legacy shim for Spyx model templates.

Prefer importing from ``spyx.models.core``.
"""

from .models import core as _core

__all__ = [name for name in vars(_core) if not name.startswith("_")]
globals().update({name: getattr(_core, name) for name in __all__})

del _core
