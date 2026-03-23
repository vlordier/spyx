"""FPGA-friendly SNN model templates for Spyx."""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from .. import nn as snn


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


@dataclass
class SparseConvConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels1: int
    channels2: int
    output_dim: int
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    event_threshold: float = 0.0
    padding: str = "SAME"
    readout: str = "mean"


@dataclass
class DepthwiseSepConvConfig:
    input_hw: tuple[int, int]
    input_channels: int
    depth_multiplier1: int
    pointwise1: int
    depth_multiplier2: int
    pointwise2: int
    output_dim: int
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


class DepthwiseConv2D(hk.Module):
    """Depthwise convolution with NHWC input."""

    def __init__(
        self,
        kernel_size: int = 3,
        depth_multiplier: int = 1,
        stride: int = 1,
        padding: str = "SAME",
        with_bias: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.stride = stride
        self.padding = padding
        self.with_bias = with_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_ch = x.shape[-1]
        w = hk.get_parameter(
            "w",
            (self.kernel_size, self.kernel_size, 1, in_ch * self.depth_multiplier),
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        y = jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            feature_group_count=in_ch,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.with_bias:
            b = hk.get_parameter("b", (in_ch * self.depth_multiplier,), init=jnp.zeros)
            y = y + b
        return y


class SparseEventConvLIFSNN(hk.Module):
    """Sparse event-driven convolutional LIF SNN with activity masking."""

    def __init__(self, cfg: SparseConvConfig, name: str | None = None):
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
            event_mask = (jnp.abs(x_t) > self.cfg.event_threshold).astype(x_t.dtype)
            x_sparse = x_t * event_mask
            s1, c1 = self.lif1(self.conv1(x_sparse), c1)
            s2, c2 = self.lif2(self.conv2(s1), c2)
            y_t = self.head(jnp.mean(s2, axis=(1, 2)))
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            active_ratio = jnp.mean(event_mask)
            return (c1, c2), (y_t, sr, active_ratio)

        _, (logits_seq, spike_rate_seq, active_ratio_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.mean(spike_rate_seq, axis=0),
            "active_ratio": jnp.mean(active_ratio_seq),
        }


class DepthwiseSeparableConvLIFSNN(hk.Module):
    """Depthwise-separable convolutional LIF SNN."""

    def __init__(self, cfg: DepthwiseSepConvConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.dw1 = DepthwiseConv2D(kernel_size=cfg.kernel_size, depth_multiplier=cfg.depth_multiplier1, padding=cfg.padding)
        self.pw1 = hk.Conv2D(cfg.pointwise1, kernel_shape=1, stride=1, padding=cfg.padding)
        self.dw2 = DepthwiseConv2D(kernel_size=cfg.kernel_size, depth_multiplier=cfg.depth_multiplier2, padding=cfg.padding)
        self.pw2 = hk.Conv2D(cfg.pointwise2, kernel_shape=1, stride=1, padding=cfg.padding)
        h, w = cfg.input_hw
        self.lif1 = snn.LIF((h, w, cfg.pointwise1), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((h, w, cfg.pointwise2), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        v1 = self.lif1.initial_state(batch)
        v2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            z1 = self.pw1(self.dw1(x_t))
            s1, c1 = self.lif1(z1, c1)
            z2 = self.pw2(self.dw2(s1))
            s2, c2 = self.lif2(z2, c2)
            y_t = self.head(jnp.mean(s2, axis=(1, 2)))
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.mean(spike_rate_seq, axis=0),
        }


def count_parameters(params: hk.Params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))


def benchmark_forward(forward_fn, sample_input: jnp.ndarray, seed: int = 0) -> dict[str, object]:
    transformed = hk.without_apply_rng(hk.transform(forward_fn))
    params = transformed.init(jax.random.PRNGKey(seed), sample_input)
    logits, aux = transformed.apply(params, sample_input)
    summary = {
        "params": count_parameters(params),
        "logits_shape": tuple(logits.shape),
    }
    if isinstance(aux, dict):
        if "spike_rate" in aux:
            summary["spike_rate"] = jnp.asarray(aux["spike_rate"])
        if "active_ratio" in aux:
            summary["active_ratio"] = float(jnp.asarray(aux["active_ratio"]))
    return summary


@dataclass
class ResidualConvConfig:
    input_hw: tuple[int, int]
    input_channels: int
    stem_channels: int
    block_channels: int
    output_dim: int
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


@dataclass
class MultiTimescaleConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    beta_fast: float = 0.6
    beta_mid: float = 0.8
    beta_slow: float = 0.95
    threshold: float = 1.0
    readout: str = "mean"


@dataclass
class RecurrentBlockConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


@dataclass
class HybridEncoderConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels1: int
    channels2: int
    head_hidden: int
    output_dim: int
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


class ResidualShallowSpikingCNN(hk.Module):
    def __init__(self, cfg: ResidualConvConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        k = cfg.kernel_size
        h, w = cfg.input_hw
        self.stem = hk.Conv2D(cfg.stem_channels, kernel_shape=k, stride=1, padding=cfg.padding)
        self.to_block = hk.Conv2D(cfg.block_channels, kernel_shape=1, stride=1, padding=cfg.padding)
        self.b1_c1 = hk.Conv2D(cfg.block_channels, kernel_shape=k, stride=1, padding=cfg.padding)
        self.b1_c2 = hk.Conv2D(cfg.block_channels, kernel_shape=k, stride=1, padding=cfg.padding)
        self.b2_c1 = hk.Conv2D(cfg.block_channels, kernel_shape=k, stride=1, padding=cfg.padding)
        self.b2_c2 = hk.Conv2D(cfg.block_channels, kernel_shape=k, stride=1, padding=cfg.padding)
        self.lif_stem = snn.LIF((h, w, cfg.stem_channels), beta=cfg.beta, threshold=cfg.threshold)
        self.lif_b1_1 = snn.LIF((h, w, cfg.block_channels), beta=cfg.beta, threshold=cfg.threshold)
        self.lif_b1_2 = snn.LIF((h, w, cfg.block_channels), beta=cfg.beta, threshold=cfg.threshold)
        self.lif_b2_1 = snn.LIF((h, w, cfg.block_channels), beta=cfg.beta, threshold=cfg.threshold)
        self.lif_b2_2 = snn.LIF((h, w, cfg.block_channels), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        states = (
            self.lif_stem.initial_state(batch),
            self.lif_b1_1.initial_state(batch),
            self.lif_b1_2.initial_state(batch),
            self.lif_b2_1.initial_state(batch),
            self.lif_b2_2.initial_state(batch),
        )

        def step(carry, x_t):
            s_stem, s11, s12, s21, s22 = carry
            x0, s_stem = self.lif_stem(self.stem(x_t), s_stem)
            x = self.to_block(x0)
            skip1 = x
            z11, s11 = self.lif_b1_1(self.b1_c1(x), s11)
            z12, s12 = self.lif_b1_2(self.b1_c2(z11), s12)
            x = z12 + skip1
            skip2 = x
            z21, s21 = self.lif_b2_1(self.b2_c1(x), s21)
            z22, s22 = self.lif_b2_2(self.b2_c2(z21), s22)
            x = z22 + skip2
            y_t = self.head(jnp.mean(x, axis=(1, 2)))
            sr = jnp.stack([jnp.mean(x0), jnp.mean(z12), jnp.mean(z22)])
            return (s_stem, s11, s12, s21, s22), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, states, x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}


class MultiTimescaleLIFBlock(hk.Module):
    def __init__(self, cfg: MultiTimescaleConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h = cfg.hidden_dim
        self.proj = hk.Linear(h)
        self.lif_fast = snn.LIF((h,), beta=cfg.beta_fast, threshold=cfg.threshold)
        self.lif_mid = snn.LIF((h,), beta=cfg.beta_mid, threshold=cfg.threshold)
        self.lif_slow = snn.LIF((h,), beta=cfg.beta_slow, threshold=cfg.threshold)
        self.fuse = hk.Linear(h)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        states = (
            self.lif_fast.initial_state(batch),
            self.lif_mid.initial_state(batch),
            self.lif_slow.initial_state(batch),
        )

        def step(carry, x_t):
            sf, sm, ss = carry
            x = self.proj(x_t)
            y_f, sf = self.lif_fast(x, sf)
            y_m, sm = self.lif_mid(x, sm)
            y_s, ss = self.lif_slow(x, ss)
            fused = self.fuse(jnp.concatenate([y_f, y_m, y_s], axis=-1))
            y_t = self.head(fused)
            sr = jnp.stack([jnp.mean(y_f), jnp.mean(y_m), jnp.mean(y_s)])
            return (sf, sm, ss), (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, states, x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}


class TinyRecurrentSpikingBlock(hk.Module):
    def __init__(self, cfg: RecurrentBlockConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.in_proj = hk.Linear(cfg.hidden_dim)
        self.core = snn.RLIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.core.initial_state(batch)

        def step(carry, x_t):
            h_t, carry = self.core(self.in_proj(x_t), carry)
            y_t = self.head(h_t)
            sr = jnp.mean(h_t)
            return carry, (y_t, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, state, x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.asarray([jnp.mean(spike_rate_seq)])}


class HybridSNNEncoderHead(hk.Module):
    def __init__(self, cfg: HybridEncoderConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        k = cfg.kernel_size
        h, w = cfg.input_hw
        self.conv1 = hk.Conv2D(cfg.channels1, kernel_shape=k, stride=1, padding=cfg.padding)
        self.conv2 = hk.Conv2D(cfg.channels2, kernel_shape=k, stride=1, padding=cfg.padding)
        self.lif1 = snn.LIF((h, w, cfg.channels1), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((h, w, cfg.channels2), beta=cfg.beta, threshold=cfg.threshold)
        self.head1 = hk.Linear(cfg.head_hidden)
        self.head2 = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        v1 = self.lif1.initial_state(batch)
        v2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            s1, c1 = self.lif1(self.conv1(x_t), c1)
            s2, c2 = self.lif2(self.conv2(s1), c2)
            pooled = jnp.mean(s2, axis=(1, 2))
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (pooled, sr)

        _, (enc_seq, spike_rate_seq) = hk.scan(step, (v1, v2), x_seq)
        enc = _readout(enc_seq, self.cfg.readout)
        logits = self.head2(jax.nn.relu(self.head1(enc)))
        return logits, {"encoder_seq": enc_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}

