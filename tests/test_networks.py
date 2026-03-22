from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import pytest

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
    rollout,
)


def _to_np(x):
    mx.eval(x)
    return np.array(x)


def _rollout_mlx(cell, xs_np: np.ndarray, state0_np: np.ndarray):
    state = mx.array(state0_np)
    spikes = []
    for t in range(xs_np.shape[0]):
        spike, state = cell(mx.array(xs_np[t]), state)
        spikes.append(_to_np(spike))
    return np.stack(spikes, axis=0), _to_np(state)


def test_li_matches_spyx_reference_one_step():
    beta = 0.8
    x_np = np.array([[0.3, -0.5]], dtype=np.float32)
    v_np = np.array([[0.2, 0.1]], dtype=np.float32)

    layer = LI(hidden_shape=(2,), beta_init=beta)
    out_mlx, state_mlx = layer(mx.array(x_np), mx.array(v_np))

    out_ref, state_ref = li_step(x_np, v_np, beta)

    np.testing.assert_allclose(_to_np(out_mlx), out_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_if_matches_spyx_reference_when_input_crosses_threshold():
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)

    neuron = IF(hidden_shape=(1,), threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = if_step(x_np, v_np, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_if_matches_spyx_reference_at_exact_threshold_boundary():
    threshold = 1.0
    x_np = np.array([[0.0]], dtype=np.float32)
    v_np = np.array([[1.0]], dtype=np.float32)

    neuron = IF(hidden_shape=(1,), threshold=threshold)
    spike_mlx, _ = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, _ = if_step(x_np, v_np, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)


def test_lif_matches_spyx_reference_when_new_voltage_crosses_threshold():
    beta = 0.9
    threshold = 1.0
    x_np = np.array([[0.3]], dtype=np.float32)
    v_np = np.array([[0.8]], dtype=np.float32)

    neuron = LIF(hidden_shape=(1,), beta_init=beta, threshold=threshold)
    spike_mlx, state_mlx = neuron(mx.array(x_np), mx.array(v_np))
    spike_ref, state_ref = lif_step(x_np, v_np, beta=beta, threshold=threshold)

    np.testing.assert_allclose(_to_np(spike_mlx), spike_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(_to_np(state_mlx), state_ref, atol=1e-6, rtol=0.0)


def test_alif_matches_spyx_reference():
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


def test_cubalif_matches_spyx_reference():
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


def test_rlif_matches_spyx_reference_with_fixed_recurrent_matrix():
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


def test_rif_matches_spyx_reference_with_fixed_recurrent_matrix():
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


def test_rcubalif_matches_spyx_reference_with_fixed_recurrent_matrix():
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


@pytest.mark.parametrize("cell_name", ["RIF", "RLIF", "RCuBaLIF"])
def test_recurrent_models_match_reference_over_multiple_steps(cell_name):
    batch = 2
    hidden = 3
    steps = 4
    threshold = 1.0
    alpha = 0.8
    beta = 0.9

    xs = np.array(
        [
            [[0.2, -0.1, 0.5], [0.0, 0.4, -0.3]],
            [[0.1, 0.3, -0.2], [0.6, -0.4, 0.1]],
            [[-0.2, 0.5, 0.2], [0.3, 0.2, -0.1]],
            [[0.4, -0.3, 0.0], [0.1, -0.2, 0.6]],
        ],
        dtype=np.float32,
    )
    w_rec = np.array(
        [
            [0.2, -0.1, 0.0],
            [0.05, 0.1, -0.2],
            [-0.15, 0.0, 0.25],
        ],
        dtype=np.float32,
    )

    if cell_name == "RIF":
        state0 = np.array(
            [[0.9, 0.2, 1.1], [0.5, 1.2, -0.1]],
            dtype=np.float32,
        )
        cell = RIF(hidden_shape=(hidden,), threshold=threshold)
        cell.w_rec = mx.array(w_rec)
        spikes_ref, state_ref = rollout(
            lambda x_t, s_t: rif_step(x_t, s_t, w_rec=w_rec, threshold=threshold),
            xs,
            state0,
        )
    elif cell_name == "RLIF":
        state0 = np.array(
            [[0.9, 0.2, 1.1], [0.5, 1.2, -0.1]],
            dtype=np.float32,
        )
        cell = RLIF(hidden_shape=(hidden,), beta_init=beta, threshold=threshold)
        cell.w_rec = mx.array(w_rec)
        spikes_ref, state_ref = rollout(
            lambda x_t, s_t: rlif_step(x_t, s_t, w_rec=w_rec, beta=beta, threshold=threshold),
            xs,
            state0,
        )
    else:
        state0 = np.array(
            [
                [1.1, 0.4, -0.2, 0.2, 0.1, 0.3],
                [0.6, 1.2, 0.7, -0.1, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        cell = RCuBaLIF(hidden_shape=(hidden,), alpha_init=alpha, beta_init=beta, threshold=threshold)
        cell.w_rec = mx.array(w_rec)
        spikes_ref, state_ref = rollout(
            lambda x_t, s_t: rcubalif_step(
                x_t,
                s_t,
                w_rec=w_rec,
                alpha=alpha,
                beta=beta,
                threshold=threshold,
            ),
            xs,
            state0,
        )

    spikes_mlx, state_mlx = _rollout_mlx(cell, xs, state0)
    np.testing.assert_allclose(spikes_mlx, spikes_ref, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(state_mlx, state_ref, atol=1e-6, rtol=0.0)











