from __future__ import annotations

import numpy as np


def spyx_heaviside(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def if_step(x: np.ndarray, v: np.ndarray, threshold: float = 1.0):
    spikes = spyx_heaviside(v - threshold)
    v_next = v + x - spikes * threshold
    return spikes, v_next


def lif_step(x: np.ndarray, v: np.ndarray, beta: float, threshold: float = 1.0):
    spikes = spyx_heaviside(v - threshold)
    v_next = beta * v + x - spikes * threshold
    return spikes, v_next


def li_step(x: np.ndarray, v: np.ndarray, beta: float):
    v_next = beta * v + x
    return v_next, v_next


def alif_step(
    x: np.ndarray,
    state: np.ndarray,
    beta: float,
    gamma: float,
    threshold: float = 1.0,
):
    v, t = np.split(state, 2, axis=-1)
    dyn_thresh = threshold + t
    spikes = spyx_heaviside(v - dyn_thresh)
    v_next = beta * v + x - spikes * dyn_thresh
    t_next = gamma * t + (1.0 - gamma) * spikes
    return spikes, np.concatenate([v_next, t_next], axis=-1)


def cubalif_step(
    x: np.ndarray,
    state: np.ndarray,
    alpha: float,
    beta: float,
    threshold: float = 1.0,
):
    v, i = np.split(state, 2, axis=-1)
    spikes = spyx_heaviside(v - threshold)
    reset = spikes * threshold
    v = v - reset
    i_next = alpha * i + x
    v_next = beta * v + i_next - reset
    return spikes, np.concatenate([v_next, i_next], axis=-1)


def rif_step(x: np.ndarray, v: np.ndarray, w_rec: np.ndarray, threshold: float = 1.0):
    spikes = spyx_heaviside(v - threshold)
    feedback = spikes @ w_rec
    v_next = v + x + feedback - spikes * threshold
    return spikes, v_next


def rlif_step(
    x: np.ndarray,
    v: np.ndarray,
    w_rec: np.ndarray,
    beta: float,
    threshold: float = 1.0,
):
    spikes = spyx_heaviside(v - threshold)
    feedback = spikes @ w_rec
    v_next = beta * v + x + feedback - spikes * threshold
    return spikes, v_next


def rcubalif_step(
    x: np.ndarray,
    state: np.ndarray,
    w_rec: np.ndarray,
    alpha: float,
    beta: float,
    threshold: float = 1.0,
):
    v, i = np.split(state, 2, axis=-1)
    spikes = spyx_heaviside(v - threshold)
    v = v - spikes * threshold
    feedback = spikes @ w_rec
    i_next = alpha * i + x + feedback
    v_next = beta * v + i_next
    return spikes, np.concatenate([v_next, i_next], axis=-1)


def rollout(step_fn, xs: np.ndarray, state0: np.ndarray):
    state = state0
    spikes = []
    for x_t in xs:
        s_t, state = step_fn(x_t, state)
        spikes.append(s_t)
    return np.stack(spikes, axis=0), state
