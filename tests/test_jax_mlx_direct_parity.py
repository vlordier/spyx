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
import spyx_mlx.nn as mlx_nn  # noqa: E402


def _state_for(neuron: str, batch: int, hidden: int):
    if neuron in {"ALIF", "CuBaLIF"}:
        return np.zeros((batch, hidden * 2), dtype=np.float32)
    return np.zeros((batch, hidden), dtype=np.float32)


def _spyx_transformed(neuron: str, hidden: int):
    hidden_shape = (hidden,)
    if neuron == "IF":
        threshold = 1.0

        def model(xs, s0):
            spike_fn = spyx_axn.superspike()

            def step_fn(v, x_t):
                spikes = spike_fn(v - threshold)
                v = v + x_t - spikes * threshold
                return v, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    elif neuron == "LIF":
        threshold = 1.0

        def model(xs, s0):
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            beta = jnp.clip(beta, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(v, x_t):
                spikes = spike_fn(v - threshold)
                v = beta * v + x_t - spikes * threshold
                return v, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    elif neuron == "ALIF":
        threshold = 1.0

        def model(xs, s0):
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            gamma = hk.get_parameter("gamma", [], init=hk.initializers.Constant(0.9))
            beta = jnp.clip(beta, 0.0, 1.0)
            gamma = jnp.clip(gamma, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(vt, x_t):
                v, t = jnp.split(vt, 2, axis=-1)
                thresh = threshold + t
                spikes = spike_fn(v - thresh)
                v = beta * v + x_t - spikes * thresh
                t = gamma * t + (1.0 - gamma) * spikes
                vt = jnp.concatenate([v, t], axis=-1)
                return vt, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    elif neuron == "CuBaLIF":
        threshold = 1.0

        def model(xs, s0):
            alpha = hk.get_parameter("alpha", [], init=hk.initializers.Constant(0.8))
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            alpha = jnp.clip(alpha, 0.0, 1.0)
            beta = jnp.clip(beta, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(vi, x_t):
                v, i = jnp.split(vi, 2, axis=-1)
                spikes = spike_fn(v - threshold)
                reset = spikes * threshold
                v = v - reset
                i = alpha * i + x_t
                v = beta * v + i - reset
                vi = jnp.concatenate([v, i], axis=-1)
                return vi, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    elif neuron == "RLIF":
        threshold = 1.0

        def model(xs, s0):
            recurrent = hk.get_parameter(
                "w",
                hidden_shape * 2,
                init=hk.initializers.TruncatedNormal(),
            )
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(0.9))
            beta = jnp.clip(beta, 0.0, 1.0)
            spike_fn = spyx_axn.superspike()

            def step_fn(v, x_t):
                spikes = spike_fn(v - threshold)
                feedback = spikes @ recurrent
                v = beta * v + x_t + feedback - spikes * threshold
                return v, spikes

            sf, ys = jax.lax.scan(step_fn, s0, xs)
            return ys, sf

        return hk.without_apply_rng(hk.transform(model))
    else:
        raise ValueError(f"Unsupported neuron: {neuron}")


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
    elif neuron == "RLIF":
        cell = mlx_nn.RLIF(hidden_shape=hidden_shape, beta_init=0.9, threshold=1.0)
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


def _replace_param_named(tree, name: str, value):
    if isinstance(tree, dict):
        updated = {}
        for key, node in tree.items():
            if key == name:
                updated[key] = value
            else:
                updated[key] = _replace_param_named(node, name, value)
        return updated
    return tree


def _forward_spyx(
    neuron: str,
    xs_np: np.ndarray,
    s0_np: np.ndarray,
    w_rec_np: np.ndarray | None = None,
):
    xs = jnp.array(xs_np)
    s0 = jnp.array(s0_np)
    transformed = _spyx_transformed(neuron, xs_np.shape[-1])
    params = transformed.init(jax.random.PRNGKey(0), xs, s0)
    if w_rec_np is not None:
        params = _replace_param_named(params, "w", jnp.array(w_rec_np, dtype=jnp.float32))
    fn = jax.jit(transformed.apply)
    y, s = fn(params, xs, s0)
    y = np.array(jax.block_until_ready(y))
    s = np.array(jax.block_until_ready(s))
    return y, s


def _forward_mlx(
    neuron: str,
    xs_np: np.ndarray,
    s0_np: np.ndarray,
    w_rec_np: np.ndarray | None = None,
):
    cell = _mlx_runner(neuron, xs_np.shape[-1])
    if neuron == "RLIF" and w_rec_np is not None:
        # `_mlx_runner` closes over the neuron instance; recover it from closure state.
        # Rebuild explicit runner here to safely inject deterministic recurrent weights.
        rlif = mlx_nn.RLIF(hidden_shape=(xs_np.shape[-1],), beta_init=0.9, threshold=1.0)
        rlif.w_rec = mx.array(w_rec_np)

        def fn(x_in, s_in):
            outs = []
            state = s_in
            for t in range(x_in.shape[0]):
                out, state = rlif(x_in[t], state)
                outs.append(out)
            return mx.stack(outs, axis=0), state

    else:
        fn = cell

    y, s = fn(mx.array(xs_np), mx.array(s0_np))
    mx.eval(y, s)
    return np.array(y), np.array(s)


@pytest.mark.parametrize("neuron", ["IF", "LIF", "ALIF", "CuBaLIF", "RLIF"])
def test_jax_mlx_direct_parity_small_rollout(neuron):
    rng = np.random.default_rng(321)
    steps, batch, hidden = 5, 3, 4
    atol = 1e-5
    rtol = 1e-5

    xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
    s0_np = _state_for(neuron, batch, hidden)
    w_rec_np = None
    if neuron == "RLIF":
        w_rec_np = rng.standard_normal((hidden, hidden), dtype=np.float32) * 0.2

    y_spyx, s_spyx = _forward_spyx(neuron, xs_np, s0_np, w_rec_np=w_rec_np)
    y_mlx, s_mlx = _forward_mlx(neuron, xs_np, s0_np, w_rec_np=w_rec_np)

    np.testing.assert_allclose(y_mlx, y_spyx, atol=atol, rtol=rtol)
    np.testing.assert_allclose(s_mlx, s_spyx, atol=atol, rtol=rtol)


@pytest.mark.parametrize("neuron", ["IF", "LIF", "ALIF", "CuBaLIF", "RLIF"])
@pytest.mark.parametrize("seed", [11, 29, 47])
def test_jax_mlx_direct_parity_long_rollout_trials(neuron, seed):
    rng = np.random.default_rng(seed)
    steps, batch, hidden = 32, 2, 8
    atol = 1e-5
    rtol = 1e-5

    xs_np = rng.standard_normal((steps, batch, hidden), dtype=np.float32)
    s0_np = _state_for(neuron, batch, hidden)
    w_rec_np = None
    if neuron == "RLIF":
        w_rec_np = rng.standard_normal((hidden, hidden), dtype=np.float32) * 0.2

    y_spyx, s_spyx = _forward_spyx(neuron, xs_np, s0_np, w_rec_np=w_rec_np)
    y_mlx, s_mlx = _forward_mlx(neuron, xs_np, s0_np, w_rec_np=w_rec_np)

    np.testing.assert_allclose(y_mlx, y_spyx, atol=atol, rtol=rtol)
    np.testing.assert_allclose(s_mlx, s_spyx, atol=atol, rtol=rtol)
