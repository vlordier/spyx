from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPYX_NN = ROOT / "spyx" / "src" / "spyx" / "nn.py"
SPYX_FN = ROOT / "spyx" / "src" / "spyx" / "fn.py"
SPYX_INIT = ROOT / "spyx" / "src" / "spyx" / "__init__.py"

MLX_NN = ROOT / "spyx_mlx" / "nn.py"
MLX_FN = ROOT / "spyx_mlx" / "fn.py"
MLX_INIT = ROOT / "spyx_mlx" / "__init__.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _init_public_bindings(path: Path) -> set[str]:
    mod = _module_ast(path)
    names: set[str] = set()
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _public_classes(path: Path) -> set[str]:
    mod = _module_ast(path)
    return {
        node.name
        for node in mod.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    }


def _public_functions(path: Path) -> set[str]:
    mod = _module_ast(path)
    return {
        node.name
        for node in mod.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def _class_init_params(path: Path, class_name: str) -> list[str]:
    mod = _module_ast(path)
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    # Drop `self`.
                    return [arg.arg for arg in item.args.args[1:]]
    raise AssertionError(f"Class {class_name} not found in {path}")


def _fn_params(path: Path, fn_name: str) -> list[str]:
    mod = _module_ast(path)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return [arg.arg for arg in node.args.args]
    raise AssertionError(f"Function {fn_name} not found in {path}")


def test_nn_public_class_surface_parity():
    # Full class parity is expected for these top-level spyx neuron/module APIs.
    expected = {
        "ALIF",
        "LI",
        "IF",
        "LIF",
        "CuBaLIF",
        "RIF",
        "RLIF",
        "RCuBaLIF",
        "ActivityRegularization",
        "SumPool",
    }
    spyx_classes = _public_classes(SPYX_NN)
    mlx_classes = _public_classes(MLX_NN)
    assert expected.issubset(spyx_classes)
    assert expected.issubset(mlx_classes)


def test_nn_key_function_surface_parity():
    expected = {"PopulationCode", "sum_pool"}
    spyx_fns = _public_functions(SPYX_NN)
    mlx_fns = _public_functions(MLX_NN)
    assert expected.issubset(spyx_fns)
    assert expected.issubset(mlx_fns)


def test_nn_constructor_signature_compatibility():
    # Names may differ internally, but key ctor arguments should match.
    checks = {
        "IF": ["hidden_shape", "threshold", "activation"],
        "LIF": ["hidden_shape", "beta", "threshold", "activation"],
        "ALIF": ["hidden_shape", "beta", "gamma", "threshold", "activation"],
        "CuBaLIF": ["hidden_shape", "alpha", "beta", "threshold", "activation"],
        "RIF": ["hidden_shape", "threshold", "activation"],
        "RLIF": ["hidden_shape", "beta", "threshold", "activation"],
        "RCuBaLIF": ["hidden_shape", "alpha", "beta", "threshold", "activation"],
    }
    for class_name, expected_prefix in checks.items():
        spyx_params = _class_init_params(SPYX_NN, class_name)
        mlx_params = _class_init_params(MLX_NN, class_name)
        assert spyx_params[: len(expected_prefix)] == expected_prefix
        assert mlx_params[: len(expected_prefix)] == expected_prefix


def test_fn_public_surface_parity():
    expected = {
        "silence_reg",
        "sparsity_reg",
        "integral_accuracy",
        "integral_crossentropy",
        "mse_spikerate",
    }
    spyx_fns = _public_functions(SPYX_FN)
    mlx_fns = _public_functions(MLX_FN)
    assert expected.issubset(spyx_fns)
    assert expected.issubset(mlx_fns)


def test_fn_signature_compatibility():
    checks = {
        "silence_reg": ["min_spikes"],
        "sparsity_reg": ["max_spikes"],
        "integral_accuracy": ["time_axis"],
        "integral_crossentropy": ["smoothing", "time_axis"],
        "mse_spikerate": ["sparsity", "smoothing", "time_axis"],
    }
    for fn_name, expected_prefix in checks.items():
        spyx_params = _fn_params(SPYX_FN, fn_name)
        mlx_params = _fn_params(MLX_FN, fn_name)
        assert spyx_params[: len(expected_prefix)] == expected_prefix
        assert mlx_params[: len(expected_prefix)] == expected_prefix


def test_top_level_init_surface_mentions_expected_modules():
    spyx_names = _init_public_bindings(SPYX_INIT)
    mlx_names = _init_public_bindings(MLX_INIT)
    for token in ["nn", "axn", "fn", "data", "experimental", "loaders", "nir"]:
        assert token in spyx_names
        assert token in mlx_names
