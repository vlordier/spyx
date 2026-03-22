from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
hk = pytest.importorskip("haiku")
pytest.importorskip("nir")
pytest.importorskip("optax")
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
SPYX_SRC = ROOT / "spyx" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SPYX_SRC) not in sys.path:
    sys.path.insert(0, str(SPYX_SRC))

import spyx.axn as spyx_axn  # noqa: E402
import spyx.nn as spyx_nn  # noqa: E402
import spyx_mlx.nn as mlx_nn  # noqa: E402


def _state_for(neuron: str, batch: int, hidden: int):
    if neuron in {"ALIF", "CuBaLIF"}:
        return np.zeros((batch, hidden * 2), dtype=np.float32)
    return np.zeros((batch, hidden), dtype=np.float32)


def _spyx_transformed(neuron: str, hidden: int):
    hidden_shape = (hidden,)
    if neuron == "IF":
        klass = spyx_nn.IF
        kwargs = {"threshold": 1.0}
    elif neuron == "LIF":
        klass = spyx_nn.LIF
        kwargs = {"beta": 0.9, "threshold": 1.0}
    elif neuron == "ALIF":
        klass = spyx_nn.ALIF
        kwargs = {"beta": 0.9, "gamma": 0.9, "threshold": 1.0}
    elif neuron == "CuBaLIF":
        klass = spyx_nn.CuBaLIF
        kwargs = {"alpha": 0.8, "beta": 0.9, "threshold": 1.0}
    else:
        raise ValueError(f"Unsupported neuron: {neuron}")

    def model(xs, s0):
        cell = klass(hidden_shape=hidden_shape, activation=spyx_axn.superspike(), **kwargs)

        def step_fn(state, x_t):
            out, new_state = cell(x_t, state)
            return new_state, out

        sf, ys = jax.lax.scan(step_fn, s0, xs)
        return ys, sf

    return hk.without_apply_rng(hk.transform(model))


def _mlx_runner(neuron: str, hidden: int):
    hidden_shape = (hidden,)
    if neuron == "IF":
        cell = mlx_nn.IF(hidden_shape=hidden_shape, threshold=1.0)
    elif neuron == "LIF":
        cell = mlx_nn.LIF(hidden_shape=hidden_shape, beta_init=0.9, threshold=1.0)
    elif neuron == "ALIF":
        cell = mlx_nn.ALIF(
            hidden_shape=hidden_shape,
            beta_init=0.9,
            gamma_init=0.9,
            threshold=1.0,
        )
    elif neuron == "CuBaLIF":
        cell = mlx_nn.CuBaLIF(
            hidden_shape=hidden_shape,
            alpha_init=0.8,
            beta_init=0.9,
            threshold=1.0,
        )
    else:
        raise ValueError(f"Unsupported neuron: {neuron}")

    def run(xs, s0):
        outs = []
        s = s0
        for t in range(xs.shape[0]):
            y, s = cell(xs[t], s)
            outs.append(y)
        return mx.stack(outs, axis=0), s

    return run


def _forward_spyx(neuron: str, xs_np: np.ndarray, s0_np: np.ndarray):
    xs = jnp.array(xs_np)
    s0 = jnp.array(s0_np)
    transformed = _spyx_transformed(neuron, xs_np.shape[-1])
    params = transformed.init(jax.random.PRNGKey(0), xs, s0)
    fn = jax.jit(transformed.apply)
    y, s = fn(params, xs, s0)
    y = np.array(jax.block_until_ready(y))
    s = np.array(jax.block_until_ready(s))
    return y, s


def _forward_mlx(neuron: str, xs_np: np.ndarray, s0_np: np.ndarray):
    fn = _mlx_runner(neuron, xs_np.shape[-1])
    y, s = fn(mx.array(xs_np), mx.array(s0_np))
    mx.eval(y, s)
    return np.array(y), np.array(s)


@pytest.mark.parametrize("neuron", ["IF", "LIF", "ALIF", "CuBaLIF"])
def test_jax_mlx_direct_parity_small_rollout(neuron):
    rng = np.random.default_rng(321)
    steps, batch, hidden = 5, 3, 4
    atol = 1e-5
    rtol = 1e-5

    xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
    s0_np = _state_for(neuron, batch, hidden)

    y_spyx, s_spyx = _forward_spyx(neuron, xs_np, s0_np)
    y_mlx, s_mlx = _forward_mlx(neuron, xs_np, s0_np)

    np.testing.assert_allclose(y_mlx, y_spyx, atol=atol, rtol=rtol)
    np.testing.assert_allclose(s_mlx, s_spyx, atol=atol, rtol=rtol)
