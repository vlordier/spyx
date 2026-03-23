"""Legacy shim for Spyx model templates.

Prefer importing from ``spyx.models.vision``.
"""

from .models import vision as _vision

__all__ = [name for name in vars(_vision) if not name.startswith("_")]
globals().update({name: getattr(_vision, name) for name in __all__})

del _vision
