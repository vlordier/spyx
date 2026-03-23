"""Compatibility barrel for FPGA-friendly SNN model templates.

Model implementations are split across focused modules:
- fpga_models_core.py
- fpga_models_vision.py
- fpga_models_fusion.py
"""

from . import fpga_models_core as _core
from . import fpga_models_fusion as _fusion
from . import fpga_models_vision as _vision

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

del _core
del _vision
del _fusion
del _export_public
