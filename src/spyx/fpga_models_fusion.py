"""Legacy shim for Spyx model templates.

Prefer importing from ``spyx.models.fusion``.
"""

from .models import fusion as _fusion

__all__ = [name for name in vars(_fusion) if not name.startswith("_")]
globals().update({name: getattr(_fusion, name) for name in __all__})

del _fusion


