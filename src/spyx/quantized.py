import dataclasses
import math
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp

from .axn import superspike


@dataclasses.dataclass(frozen=True)
class FixedPointConfig:
    """Configuration for fake fixed-point arithmetic."""

    total_bits: int = 16
    frac_bits: int = 8
    rounding: Literal["nearest", "floor", "ceil"] = "nearest"
    scale_mode: Literal["fixed", "max_abs"] = "fixed"
    eps: float = 1e-8

    @property
    def scale(self) -> float:
        return float(2**self.frac_bits)

    @property
    def qmin(self) -> int:
        return -(2 ** (self.total_bits - 1))

    @property
    def qmax(self) -> int:
        return 2 ** (self.total_bits - 1) - 1


def _ste(discrete: jnp.ndarray, continuous: jnp.ndarray) -> jnp.ndarray:
    """Straight-through estimator for quantized forward / dense backward."""

    return continuous + jax.lax.stop_gradient(discrete - continuous)


def quantize_fixed(x: jnp.ndarray, cfg: FixedPointConfig, ste: bool = True) -> jnp.ndarray:
    if cfg.scale_mode == "fixed":
        scale = cfg.scale
    else:
        # Dynamic symmetric scale to match tensor range to int range.
        max_abs = jnp.max(jnp.abs(x))
        denom = jnp.maximum(max_abs, cfg.eps)
        scale = cfg.qmax / denom

    x_scaled = x * scale
    if cfg.rounding == "nearest":
        scaled = jnp.round(x_scaled)
    elif cfg.rounding == "floor":
        scaled = jnp.floor(x_scaled)
    elif cfg.rounding == "ceil":
        scaled = jnp.ceil(x_scaled)
    else:
        msg = f"Unsupported rounding strategy: {cfg.rounding}"
        raise ValueError(msg)

    clipped = jnp.clip(scaled, cfg.qmin, cfg.qmax)
    quantized = clipped / scale
    if not ste:
        return quantized
    return _ste(quantized, x)


def ternarize_weights(
    w: jnp.ndarray,
    threshold: float = 0.05,
    strategy: Literal["threshold", "mean_scaled_threshold", "topk"] = "threshold",
    topk_ratio: float = 0.1,
    ste: bool = True,
) -> jnp.ndarray:
    if strategy == "threshold":
        thresh = threshold
    elif strategy == "mean_scaled_threshold":
        thresh = threshold * jnp.mean(jnp.abs(w))
    elif strategy == "topk":
        flat = jnp.ravel(jnp.abs(w))
        size = flat.shape[0]
        k = max(1, min(size, int(math.ceil(topk_ratio * size))))
        idx = size - k
        kth = jnp.sort(flat)[idx]
        thresh = kth
    else:
        msg = f"Unsupported ternary strategy: {strategy}"
        raise ValueError(msg)

    positive = w > thresh
    negative = w < -thresh
    ternary = jnp.where(positive, 1.0, jnp.where(negative, -1.0, 0.0)).astype(w.dtype)
    if not ste:
        return ternary
    return _ste(ternary, w)


class FixedPointLinear(hk.Module):
    """Linear layer with fixed-point fake-quantized arithmetic."""

    def __init__(
        self,
        output_size: int,
        cfg: FixedPointConfig | None = None,
        with_bias: bool = False,
        name: str = "FixedPointLinear",
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.cfg = cfg or FixedPointConfig()
        self.with_bias = with_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        w = hk.get_parameter(
            "w",
            shape=(in_features, self.output_size),
            init=hk.initializers.TruncatedNormal(),
        )
        x_q = quantize_fixed(x, self.cfg)
        w_q = quantize_fixed(w, self.cfg)
        y = jnp.matmul(x_q, w_q)
        y = quantize_fixed(y, self.cfg)
        if self.with_bias:
            b = hk.get_parameter("b", shape=(self.output_size,), init=jnp.zeros)
            y = quantize_fixed(y + quantize_fixed(b, self.cfg), self.cfg)
        return y


class TernaryLinear(hk.Module):
    """Linear layer with ternary weights in {-1, 0, +1}."""

    def __init__(
        self,
        output_size: int,
        threshold: float = 0.05,
        strategy: Literal["threshold", "mean_scaled_threshold", "topk"] = "threshold",
        topk_ratio: float = 0.1,
        cfg: FixedPointConfig | None = None,
        with_bias: bool = False,
        name: str = "TernaryLinear",
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.threshold = threshold
        self.strategy = strategy
        self.topk_ratio = topk_ratio
        self.cfg = cfg or FixedPointConfig()
        self.with_bias = with_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        w = hk.get_parameter(
            "w",
            shape=(in_features, self.output_size),
            init=hk.initializers.TruncatedNormal(),
        )
        x_q = quantize_fixed(x, self.cfg)
        w_t = ternarize_weights(
            w,
            threshold=self.threshold,
            strategy=self.strategy,
            topk_ratio=self.topk_ratio,
        )
        y = jnp.matmul(x_q, w_t)
        y = quantize_fixed(y, self.cfg)
        if self.with_bias:
            b = hk.get_parameter("b", shape=(self.output_size,), init=jnp.zeros)
            y = quantize_fixed(y + quantize_fixed(b, self.cfg), self.cfg)
        return y


class TernaryFixedPointLinear(TernaryLinear):
    """Alias class for Stage C compatibility in experiment scripts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="TernaryFixedPointLinear", **kwargs)


class FixedPointLIF(hk.RNNCore):
    """LIF clone with fixed-point arithmetic at each update."""

    def __init__(
        self,
        hidden_shape: tuple,
        beta: float | None = None,
        threshold: float = 1.0,
        activation=superspike(),
        cfg: FixedPointConfig | None = None,
        name: str = "FixedPointLIF",
    ):
        super().__init__(name=name)
        self.hidden_shape = hidden_shape
        self.beta = beta
        self.threshold = threshold
        self.spike = activation
        self.cfg = cfg or FixedPointConfig()

    def __call__(self, x: jnp.ndarray, v: jnp.ndarray):
        if self.beta is None:
            beta = hk.get_parameter(
                "beta",
                self.hidden_shape,
                init=hk.initializers.TruncatedNormal(0.25, 0.5),
            )
        else:
            beta = hk.get_parameter("beta", [], init=hk.initializers.Constant(self.beta))

        beta = jnp.clip(beta, 0.0, 1.0)
        x_q = quantize_fixed(x, self.cfg)
        v_q = quantize_fixed(v, self.cfg)
        beta_q = quantize_fixed(beta, self.cfg)

        spikes = self.spike(v_q - self.threshold)
        v_next = beta_q * v_q + x_q - spikes * self.threshold
        v_next = quantize_fixed(v_next, self.cfg)
        return spikes, v_next

    def initial_state(self, batch_size: int):
        return jnp.zeros((batch_size,) + self.hidden_shape)


class TernaryFixedPointLIF(FixedPointLIF):
    """Alias class for Stage C compatibility in experiment scripts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="TernaryFixedPointLIF", **kwargs)
