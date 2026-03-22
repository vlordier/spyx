from pathlib import Path
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spyx_mlx.nn import ALIF, CuBaLIF, IF, LI, LIF, RCuBaLIF, RIF, RLIF

STRICT_PARITY = os.getenv("SPYX_MLX_STRICT_PARITY") == "1"
pytestmark = pytest.mark.skipif(
    not STRICT_PARITY,
    reason="Set SPYX_MLX_STRICT_PARITY=1 to run strict Spyx-vs-MLX parity checks.",
)


def _spyx_heaviside(x):
    return (x > 0).astype(np.float32)


def _to_np(x):
    mx.eval(x)
    return np.array(x)


def _spyx_if_step(x, v, threshold=1.0):
    spikes = _spyx_heaviside(v - threshold)
    v_next = v + x - spikes * threshold
    return spikes, v_next


def _spyx_lif_step(x, v, beta, threshold=1.0):
    spikes = _spyx_heaviside(v - threshold)
    v_next = beta * v + x - spikes * threshold
    return spikes, v_next


def _spyx_li_step(x, v, beta):
    v_next = beta * v + x
    return v_next, v_next


def _spyx_alif_step(x, state, beta, gamma, threshold=1.0):
    v, t = np.split(state, 2, axis=-1)
    dyn_thresh = threshold + t
    spikes = _spyx_heaviside(v - dyn_thresh)
    v_next = beta * v + x - spikes * dyn_thresh
    t_next = gamma * t + (1.0 - gamma) * spikes
    return spikes, np.concatenate([v_next, t_next], axis=-1)


def _spyx_cubalif_step(x, state, alpha, beta, threshold=1.0):
    v, i = np.split(state, 2, axis=-1)
    spikes = _spyx_heaviside(v - threshold)
    reset = spikes * threshold
    v = v - reset
    i_next = alpha * i + x
    v_next = beta * v + i_next - reset
    return spikes, np.concatenate([v_next, i_next], axis=-1)


def _spyx_rif_step(x, v, w_rec, threshold=1.0):
    spikes = _spyx_heaviside(v - threshold)
    feedback = spikes @ w_rec
    v_next = v + x + feedback - spikes * threshold
    return spikes, v_next


def _spyx_rcubalif_step(x, state, w_rec, alpha, beta, threshold=1.0):
    v, i = np.split(state, 2, axis=-1)
    spikes = _spyx_heaviside(v - threshold)
    v = v - spikes * threshold
    feedback = spikes @ w_rec
    i_next = alpha * i + x + feedback
    v_next = beta * v + i_next
    return spikes, np.concatenate([v_next, i_next], axis=-1)


def test_li_strict_parity_one_step():
    beta = 0.8
    x_np = np.array([[0.3, -0.5]], dtype=np.float32)
    v_np = np.array([[0.2, 0.1]], dtype=np.float32)

    layer = LI(hidden_shape=(2,), beta_init=beta)
    out_mlx, state_mlx = layer(mx.array(x_np), mx.array(v_np))
    out_ref, state_ref = _spyx_li_step(x_np, v_np, beta)

    np.testing.assert_allclose(_to_np(out_mlx), out_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_if_strict_parity_one_step():
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)

    neuron = IF(hidden_shape=(1,), threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = _spyx_if_step(x_np, v_np, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_lif_strict_parity_one_step():
    beta = 0.9
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)

    neuron = LIF(hidden_shape=(1,), beta_init=beta, threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = _spyx_lif_step(x_np, v_np, beta=beta, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_alif_strict_parity_one_step():
    beta = 0.9
    gamma = 0.9
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    state_np = np.array([[0.8, 0.0]], dtype=np.float32)

    neuron = ALIF(hidden_shape=(1,), beta_init=beta, gamma_init=gamma, threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(state_np))
    spike_ref, state_ref = _spyx_alif_step(x_np, state_np, beta=beta, gamma=gamma, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_cubalif_strict_parity_one_step():
    alpha = 0.8
    beta = 0.9
    threshold = 1.0
    x_np = np.array([[0.2]], dtype=np.float32)
    state_np = np.array([[1.2, 0.1]], dtype=np.float32)

    neuron = CuBaLIF(hidden_shape=(1,), alpha_init=alpha, beta_init=beta, threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(state_np))
    spike_ref, state_ref = _spyx_cubalif_step(x_np, state_np, alpha=alpha, beta=beta, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_rlif_strict_parity_one_step():
    beta = 0.9
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)
    w_np = np.array([[0.2]], dtype=np.float32)

    neuron = RLIF(hidden_shape=(1,), beta_init=beta, threshold=threshold)
    neuron.w_rec = mx.array(w_np)

    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))

    spike_ref = _spyx_heaviside(v_np - threshold)
    feedback_ref = spike_ref @ w_np
    state_ref = beta * v_np + x_np + feedback_ref - spike_ref * threshold

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_rif_strict_parity_one_step():
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)
    w_np = np.array([[0.2]], dtype=np.float32)

    neuron = RIF(hidden_shape=(1,), threshold=threshold)
    neuron.w_rec = mx.array(w_np)

    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = _spyx_rif_step(x_np, v_np, w_rec=w_np, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_rcubalif_strict_parity_one_step():
    alpha = 0.8
    beta = 0.9
    threshold = 1.0
    x_np = np.array([[0.2]], dtype=np.float32)
    state_np = np.array([[1.2, 0.1]], dtype=np.float32)
    w_np = np.array([[0.2]], dtype=np.float32)

    neuron = RCuBaLIF(hidden_shape=(1,), alpha_init=alpha, beta_init=beta, threshold=threshold)
    neuron.w_rec = mx.array(w_np)

    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(state_np))
    spike_ref, state_ref = _spyx_rcubalif_step(
        x_np,
        state_np,
        w_rec=w_np,
        alpha=alpha,
        beta=beta,
        threshold=threshold,
    )

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)