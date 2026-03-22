from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spyx_mlx.nn import ALIF, CuBaLIF, IF, LI, LIF, RCuBaLIF, RIF, RLIF
from .parity_reference import (
    alif_step,
    cubalif_step,
    if_step,
    lif_step,
    li_step,
    rcubalif_step,
    rif_step,
    rlif_step,
)


def _to_np(x):
    mx.eval(x)
    return np.array(x)


def test_li_strict_parity_one_step():
    beta = 0.8
    x_np = np.array([[0.3, -0.5]], dtype=np.float32)
    v_np = np.array([[0.2, 0.1]], dtype=np.float32)

    layer = LI(hidden_shape=(2,), beta_init=beta)
    out_mlx, state_mlx = layer(mx.array(x_np), mx.array(v_np))
    out_ref, state_ref = li_step(x_np, v_np, beta)

    np.testing.assert_allclose(_to_np(out_mlx), out_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_if_strict_parity_one_step():
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)

    neuron = IF(hidden_shape=(1,), threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = if_step(x_np, v_np, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_lif_strict_parity_one_step():
    beta = 0.9
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)

    neuron = LIF(hidden_shape=(1,), beta_init=beta, threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = lif_step(x_np, v_np, beta=beta, threshold=threshold)

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
    spike_ref, state_ref = alif_step(x_np, state_np, beta=beta, gamma=gamma, threshold=threshold)

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
    spike_ref, state_ref = cubalif_step(x_np, state_np, alpha=alpha, beta=beta, threshold=threshold)

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

    spike_ref, state_ref = rlif_step(x_np, v_np, w_rec=w_np, beta=beta, threshold=threshold)

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
    spike_ref, state_ref = rif_step(x_np, v_np, w_rec=w_np, threshold=threshold)

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
    spike_ref, state_ref = rcubalif_step(
        x_np,
        state_np,
        w_rec=w_np,
        alpha=alpha,
        beta=beta,
        threshold=threshold,
    )

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)