"""FPGA-friendly SNN model templates for Spyx."""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from . import nn as snn


@dataclass
class MLPConfig:
    input_dim: int
    hidden1: int
    hidden2: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


@dataclass
class ConvConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels1: int
    channels2: int
    output_dim: int
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


def _readout(logits_seq: jnp.ndarray, mode: str) -> jnp.ndarray:
    if mode == "mean":
        return jnp.mean(logits_seq, axis=0)
    if mode == "last":
        return logits_seq[-1]
    raise ValueError(f"Unsupported readout mode {mode}")


def ternary_project(w: jnp.ndarray, threshold: float = 0.05, use_ste: bool = True) -> jnp.ndarray:
    reduce_axes = tuple(range(w.ndim - 1))
    scale = jnp.mean(jnp.abs(w), axis=reduce_axes, keepdims=True)
    ternary = jnp.where(jnp.abs(w) < threshold, 0.0, jnp.sign(w)) * scale
    if use_ste:
        return w + jax.lax.stop_gradient(ternary - w)
    return ternary


class TernaryLinear(hk.Module):
    def __init__(self, output_size: int, threshold: float = 0.05, use_ste: bool = True, with_bias: bool = True, name: str | None = None):
        super().__init__(name=name)
        self.output_size = output_size
        self.threshold = threshold
        self.use_ste = use_ste
        self.with_bias = with_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_size = x.shape[-1]
        w = hk.get_parameter("w", (in_size, self.output_size), init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"))
        y = x @ ternary_project(w, threshold=self.threshold, use_ste=self.use_ste)
        if self.with_bias:
            b = hk.get_parameter("b", (self.output_size,), init=jnp.zeros)
            y = y + b
        return y


class TernaryConv2D(hk.Module):
    def __init__(self, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: str = "SAME", threshold: float = 0.05, use_ste: bool = True, with_bias: bool = True, name: str | None = None):
        super().__init__(name=name)
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.use_ste = use_ste
        self.with_bias = with_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_ch = x.shape[-1]
        w = hk.get_parameter("w", (self.kernel_size, self.kernel_size, in_ch, self.out_ch), init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"))
        w_q = ternary_project(w, threshold=self.threshold, use_ste=self.use_ste)
        y = jax.lax.conv_general_dilated(x, w_q, (self.stride, self.stride), self.padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))
        if self.with_bias:
            b = hk.get_parameter("b", (self.out_ch,), init=jnp.zeros)
            y = y + b
        return y


class LIFMLP(hk.Module):
    def __init__(self, cfg: MLPConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.dense1 = hk.Linear(cfg.hidden1)
        self.dense2 = hk.Linear(cfg.hidden2)
        self.head = hk.Linear(cfg.output_dim)
        self.lif1 = snn.LIF((cfg.hidden1,), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((cfg.hidden2,), beta=cfg.beta, threshold=cfg.threshold)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        v1 = self.lif1.initial_state(batch)
        v2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            s1, c1 = self.lif1(self.dense1(x_t), c1)
            s2, c2 = self.lif2(self.dense2(s1), c2)
            y_t = self.head(s2)
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}


class TernaryLIFMLP(hk.Module):
    def __init__(self, cfg: MLPConfig, threshold: float = 0.05, use_ste: bool = True, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.dense1 = TernaryLinear(cfg.hidden1, threshold=threshold, use_ste=use_ste)
        self.dense2 = TernaryLinear(cfg.hidden2, threshold=threshold, use_ste=use_ste)
        self.head = TernaryLinear(cfg.output_dim, threshold=threshold, use_ste=use_ste)
        self.lif1 = snn.LIF((cfg.hidden1,), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((cfg.hidden2,), beta=cfg.beta, threshold=cfg.threshold)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        v1 = self.lif1.initial_state(batch)
        v2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            s1, c1 = self.lif1(self.dense1(x_t), c1)
            s2, c2 = self.lif2(self.dense2(s1), c2)
            y_t = self.head(s2)
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}


class ConvLIFSNN(hk.Module):
    def __init__(self, cfg: ConvConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.conv1 = hk.Conv2D(cfg.channels1, kernel_shape=cfg.kernel_size, stride=1, padding=cfg.padding)
        self.conv2 = hk.Conv2D(cfg.channels2, kernel_shape=cfg.kernel_size, stride=1, padding=cfg.padding)
        h, w = cfg.input_hw
        self.lif1 = snn.LIF((h, w, cfg.channels1), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((h, w, cfg.channels2), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        v1 = self.lif1.initial_state(batch)
        v2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            s1, c1 = self.lif1(self.conv1(x_t), c1)
            s2, c2 = self.lif2(self.conv2(s1), c2)
            y_t = self.head(jnp.mean(s2, axis=(1, 2)))
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}


class TernaryConvLIFSNN(hk.Module):
    def __init__(self, cfg: ConvConfig, threshold: float = 0.05, use_ste: bool = True, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.conv1 = TernaryConv2D(cfg.channels1, kernel_size=cfg.kernel_size, padding=cfg.padding, threshold=threshold, use_ste=use_ste)
        self.conv2 = TernaryConv2D(cfg.channels2, kernel_size=cfg.kernel_size, padding=cfg.padding, threshold=threshold, use_ste=use_ste)
        h, w = cfg.input_hw
        self.lif1 = snn.LIF((h, w, cfg.channels1), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((h, w, cfg.channels2), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        v1 = self.lif1.initial_state(batch)
        v2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            s1, c1 = self.lif1(self.conv1(x_t), c1)
            s2, c2 = self.lif2(self.conv2(s1), c2)
            y_t = self.head(jnp.mean(s2, axis=(1, 2)))
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}
