"""Vision-centric FPGA-friendly SNN model templates for Spyx."""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy import ndimage as jndimage

from .. import nn as snn
from .core import _readout


def _topk_mask(x: jnp.ndarray, k: int, axis: int = -1) -> jnp.ndarray:
    if k <= 0:
        return jnp.zeros_like(x)
    dim = x.shape[axis]
    if dim == 0:
        return jnp.zeros_like(x)
    k = min(k, dim)
    top_vals, _ = jax.lax.top_k(x, k)
    thresh = jnp.take(top_vals, k - 1, axis=-1)
    while thresh.ndim < x.ndim:
        thresh = jnp.expand_dims(thresh, axis=-1)
    return (x >= thresh).astype(x.dtype)


class RegionActivationRouter(hk.Module):
    """Tile router selecting top-k active patches for downstream compute."""

    def __init__(self, top_k: int = 4, patch: int = 4, name: str | None = None):
        super().__init__(name=name)
        self.top_k = top_k
        self.patch = patch

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        b, h, w, _ = x.shape
        ph = h // self.patch
        pw = w // self.patch
        x_crop = x[:, : ph * self.patch, : pw * self.patch, :]
        patches = x_crop.reshape(b, ph, self.patch, pw, self.patch, -1)
        scores = jnp.mean(jnp.abs(patches), axis=(2, 4, 5)).reshape(b, ph * pw)
        gate = _topk_mask(scores, self.top_k)
        return scores, gate


class KWTASaliencyGate(hk.Module):
    """k-WTA saliency gate over channels."""

    def __init__(self, k: int, name: str | None = None):
        super().__init__(name=name)
        self.k = k

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        saliency = jnp.mean(jnp.abs(x), axis=tuple(range(1, x.ndim - 1)))
        gate = _topk_mask(saliency, self.k, axis=-1)
        expand_shape = (slice(None),) + (None,) * (x.ndim - 2) + (slice(None),)
        x_gated = x * gate[expand_shape]
        return x_gated, gate


class TimeSurfaceEncoder(hk.Module):
    """Simple differentiable time-surface encoder using exponential decay."""

    def __init__(self, tau: float = 4.0, name: str | None = None):
        super().__init__(name=name)
        self.tau = tau

    def __call__(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        t = x_seq.shape[0]
        idx = jnp.arange(t, dtype=x_seq.dtype)
        dt = (idx[-1] - idx) / jnp.maximum(self.tau, 1e-6)
        w = jnp.exp(-dt).reshape((t, 1, 1, 1, 1))
        return x_seq * w


@dataclass
class FoveatedDualPathConfig:
    input_hw: tuple[int, int]
    input_channels: int
    fovea_hw: tuple[int, int]
    channels_fovea: int
    channels_periphery: int
    output_dim: int
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


@dataclass
class LogPolarFoveatedConvConfig:
    input_hw: tuple[int, int]
    input_channels: int
    radial_bins: int
    angular_bins: int
    channels1: int
    channels2: int
    output_dim: int
    kernel_size: int = 3
    min_radius: float = 1.0
    max_radius_scale: float = 1.0
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


def _build_log_polar_grid(
    input_hw: tuple[int, int],
    radial_bins: int,
    angular_bins: int,
    min_radius: float,
    max_radius_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    height, width = input_hw
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    max_radius = min(center_y, center_x) * max_radius_scale
    max_radius = max(max_radius, min_radius + 1e-3)
    radial_steps = jnp.linspace(0.0, 1.0, radial_bins)
    angular_steps = jnp.arange(angular_bins, dtype=jnp.float32) * (2.0 * jnp.pi / max(angular_bins, 1))
    log_min = jnp.log(jnp.maximum(min_radius, 1e-3))
    log_max = jnp.log(jnp.maximum(max_radius, min_radius + 1e-3))
    radii = jnp.exp(log_min + radial_steps * (log_max - log_min))
    ys = center_y + radii[:, None] * jnp.sin(angular_steps)[None, :]
    xs = center_x + radii[:, None] * jnp.cos(angular_steps)[None, :]
    ys = jnp.clip(ys, 0.0, height - 1.0)
    xs = jnp.clip(xs, 0.0, width - 1.0)
    return ys, xs


def _sample_log_polar_image(x: jnp.ndarray, ys: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    coords = [ys, xs]

    def sample_batch(image: jnp.ndarray) -> jnp.ndarray:
        channels_first = jnp.moveaxis(image, -1, 0)

        def sample_channel(channel: jnp.ndarray) -> jnp.ndarray:
            return jndimage.map_coordinates(channel, coords, order=1, mode="nearest")

        sampled = jax.vmap(sample_channel)(channels_first)
        return jnp.moveaxis(sampled, 0, -1)

    return jax.vmap(sample_batch)(x)


def _shift_horizontally(x: jnp.ndarray, disparity: int) -> jnp.ndarray:
    if disparity == 0:
        return x
    pad = jnp.zeros((x.shape[0], x.shape[1], disparity, x.shape[3]), dtype=x.dtype)
    return jnp.concatenate([x[:, :, disparity:, :], pad], axis=2)


def _fixed_filter_response(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    gray = jnp.mean(x, axis=-1, keepdims=True)
    kernel = kernel[:, :, None, None].astype(x.dtype)
    return jax.lax.conv_general_dilated(gray, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))


def _classical_filter_bank(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    sobel_x = jnp.asarray([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    sobel_y = sobel_x.T
    laplace = jnp.asarray([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    dog = jnp.asarray([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    gx = _fixed_filter_response(x, sobel_x)
    gy = _fixed_filter_response(x, sobel_y)
    sobel_mag = jnp.sqrt(jnp.maximum(gx * gx + gy * gy, 1e-6))
    lap = _fixed_filter_response(x, laplace)
    dog_resp = _fixed_filter_response(x, dog)
    features = jnp.concatenate([x, sobel_mag, lap, dog_resp], axis=-1)
    energy = jnp.stack([jnp.mean(jnp.abs(sobel_mag)), jnp.mean(jnp.abs(lap)), jnp.mean(jnp.abs(dog_resp))])
    return features, energy


def _event_pool_features(x: jnp.ndarray, event_threshold: float, pool_modes: tuple[str, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    activity = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    event_mask = (activity > event_threshold).astype(x.dtype)
    active = x * event_mask
    denom = jnp.maximum(jnp.sum(event_mask, axis=(1, 2)), 1.0)
    pooled_features = []
    for mode in pool_modes:
        if mode == "avg":
            pooled = jnp.sum(active, axis=(1, 2)) / denom
        elif mode == "max":
            pooled = jnp.max(active, axis=(1, 2))
        elif mode == "l2":
            pooled = jnp.sqrt(jnp.sum(active * active, axis=(1, 2)) / denom)
        else:
            raise ValueError(f"Unsupported pooling mode {mode}")
        pooled_features.append(pooled)
    return jnp.concatenate(pooled_features, axis=-1), jnp.mean(event_mask)


def _normalize_unit_interval(x: jnp.ndarray) -> jnp.ndarray:
    x_min = jnp.min(x, axis=-1, keepdims=True)
    x_max = jnp.max(x, axis=-1, keepdims=True)
    return (x - x_min) / jnp.maximum(x_max - x_min, 1e-6)


def _crop_window_batch(x: jnp.ndarray, start_y: jnp.ndarray, start_x: jnp.ndarray, window_hw: tuple[int, int]) -> jnp.ndarray:
    crop_h, crop_w = window_hw

    def crop_one(img: jnp.ndarray, y0: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.dynamic_slice(img, (y0, x0, 0), (crop_h, crop_w, img.shape[-1]))

    return jax.vmap(crop_one)(x, start_y, start_x)


class LogPolarFoveatedConvSNN(hk.Module):
    """Log-polar foveated convolutional SNN with an explicit geometric front-end."""

    def __init__(self, cfg: LogPolarFoveatedConvConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.conv1 = hk.Conv2D(cfg.channels1, kernel_shape=cfg.kernel_size, stride=1, padding=cfg.padding)
        self.conv2 = hk.Conv2D(cfg.channels2, kernel_shape=cfg.kernel_size, stride=1, padding=cfg.padding)
        self.lif1 = snn.LIF((cfg.radial_bins, cfg.angular_bins, cfg.channels1), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((cfg.radial_bins, cfg.angular_bins, cfg.channels2), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)
        self.grid_y, self.grid_x = _build_log_polar_grid(
            cfg.input_hw,
            cfg.radial_bins,
            cfg.angular_bins,
            cfg.min_radius,
            cfg.max_radius_scale,
        )

    def _transform(self, x: jnp.ndarray) -> jnp.ndarray:
        return _sample_log_polar_image(x, self.grid_y, self.grid_x)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state1 = self.lif1.initial_state(batch)
        state2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            x_lp = self._transform(x_t)
            s1, c1 = self.lif1(self.conv1(x_lp), c1)
            s2, c2 = self.lif2(self.conv2(s1), c2)
            y_t = self.head(jnp.mean(s2, axis=(1, 2)))
            sr = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            radial_energy = jnp.mean(x_lp, axis=(0, 2, 3))
            return (c1, c2), (y_t, sr, x_lp, radial_energy)

        _, (logits_seq, spike_rate_seq, logpolar_seq, radial_energy_seq) = hk.scan(step, (state1, state2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.mean(spike_rate_seq, axis=0),
            "logpolar_seq": logpolar_seq,
            "radial_energy": jnp.mean(radial_energy_seq, axis=0),
        }


class FoveatedDualPathSNN(hk.Module):
    """Dual-path fovea/periphery spiking encoder."""

    def __init__(self, cfg: FoveatedDualPathConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.fovea_conv = hk.Conv2D(cfg.channels_fovea, kernel_shape=cfg.kernel_size, padding=cfg.padding)
        self.periph_conv = hk.Conv2D(cfg.channels_periphery, kernel_shape=cfg.kernel_size, padding=cfg.padding)
        fh, fw = cfg.fovea_hw
        h, w = cfg.input_hw
        self.fovea_lif = snn.LIF((fh, fw, cfg.channels_fovea), beta=cfg.beta, threshold=cfg.threshold)
        self.periph_lif = snn.LIF((h // 2, w // 2, cfg.channels_periphery), beta=cfg.beta, threshold=cfg.threshold)
        self.fuse = hk.Linear(cfg.output_dim)

    def _crop_fovea(self, x: jnp.ndarray) -> jnp.ndarray:
        h, w = x.shape[1], x.shape[2]
        fh, fw = self.cfg.fovea_hw
        hs = (h - fh) // 2
        ws = (w - fw) // 2
        return x[:, hs : hs + fh, ws : ws + fw, :]

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        sf = self.fovea_lif.initial_state(batch)
        sp = self.periph_lif.initial_state(batch)

        def step(carry, x_t):
            c_f, c_p = carry
            fovea = self._crop_fovea(x_t)
            periph = jax.image.resize(x_t, (x_t.shape[0], x_t.shape[1] // 2, x_t.shape[2] // 2, x_t.shape[3]), method="linear")
            y_f, c_f = self.fovea_lif(self.fovea_conv(fovea), c_f)
            y_p, c_p = self.periph_lif(self.periph_conv(periph), c_p)
            z = jnp.concatenate([jnp.mean(y_f, axis=(1, 2)), jnp.mean(y_p, axis=(1, 2))], axis=-1)
            logits = self.fuse(z)
            sr = jnp.stack([jnp.mean(y_f), jnp.mean(y_p)])
            return (c_f, c_p), (logits, sr)

        _, (logits_seq, spike_rate_seq) = hk.scan(step, (sf, sp), x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(spike_rate_seq, axis=0)}


@dataclass
class WTAFoveatedStackConfig:
    input_hw: tuple[int, int]
    input_channels: int
    fovea_hw: tuple[int, int]
    channels_fovea: int
    channels_periphery: int
    output_dim: int
    router_patch: int = 4
    router_top_k: int = 1
    kwta_k: int = 4
    tau: float = 4.0
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class IntegratedWTAFoveatedSNN(hk.Module):
    """Integrated time-surface, routing, WTA, and foveated dual-path stack."""

    def __init__(self, cfg: WTAFoveatedStackConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.time_surface = TimeSurfaceEncoder(tau=cfg.tau)
        self.router = RegionActivationRouter(top_k=cfg.router_top_k, patch=cfg.router_patch)
        self.saliency = KWTASaliencyGate(k=cfg.kwta_k)
        self.fovea_conv = hk.Conv2D(cfg.channels_fovea, kernel_shape=3, padding="SAME")
        self.periph_conv = hk.Conv2D(cfg.channels_periphery, kernel_shape=3, padding="SAME")
        fh, fw = cfg.fovea_hw
        h, w = cfg.input_hw
        self.fovea_lif = snn.LIF((fh, fw, cfg.channels_fovea), beta=cfg.beta, threshold=cfg.threshold)
        self.periph_lif = snn.LIF((h // 2, w // 2, cfg.channels_periphery), beta=cfg.beta, threshold=cfg.threshold)
        self.route_proj = hk.Linear(cfg.output_dim)
        self.head = hk.Linear(cfg.output_dim)

    def _crop_fovea_from_scores(self, x: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        batch, height, width, _ = x.shape
        patch = self.cfg.router_patch
        fh, fw = self.cfg.fovea_hw
        pw = max(width // patch, 1)
        best_idx = jnp.argmax(scores, axis=-1)
        center_y = (best_idx // pw) * patch + patch // 2
        center_x = (best_idx % pw) * patch + patch // 2
        start_y = jnp.clip(center_y - fh // 2, 0, height - fh)
        start_x = jnp.clip(center_x - fw // 2, 0, width - fw)
        return _crop_window_batch(x, start_y, start_x, self.cfg.fovea_hw)

    def __call__(self, x_seq: jnp.ndarray):
        x_ts = self.time_surface(x_seq)
        _, batch, _, _, _ = x_ts.shape
        sf = self.fovea_lif.initial_state(batch)
        sp = self.periph_lif.initial_state(batch)

        def step(carry, x_t):
            c_f, c_p = carry
            scores, region_gate = self.router(x_t)
            fovea = self._crop_fovea_from_scores(x_t, scores)
            fovea_feat = self.fovea_conv(fovea)
            fovea_feat, channel_gate = self.saliency(fovea_feat)
            periph = jax.image.resize(x_t, (x_t.shape[0], x_t.shape[1] // 2, x_t.shape[2] // 2, x_t.shape[3]), method="linear")
            s_f, c_f = self.fovea_lif(fovea_feat, c_f)
            s_p, c_p = self.periph_lif(self.periph_conv(periph), c_p)
            fused = jnp.concatenate([jnp.mean(s_f, axis=(1, 2)), jnp.mean(s_p, axis=(1, 2))], axis=-1)
            logits_t = self.head(fused) + self.route_proj(scores)
            route_stats = jnp.mean(region_gate, axis=0)
            channel_stats = jnp.mean(channel_gate, axis=0)
            sr_t = jnp.stack([jnp.mean(s_f), jnp.mean(s_p)])
            return (c_f, c_p), (logits_t, sr_t, route_stats, channel_stats)

        _, (logits_seq, sr_seq, route_seq, channel_seq) = hk.scan(step, (sf, sp), x_ts)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.mean(sr_seq, axis=0),
            "region_gate": jnp.mean(route_seq, axis=0),
            "channel_gate": jnp.mean(channel_seq, axis=0),
        }


@dataclass
class EventDrivenSparseFoveatedConfig:
    input_hw: tuple[int, int]
    input_channels: int
    fovea_hw: tuple[int, int]
    channels_fovea: int
    channels_periphery: int
    output_dim: int
    event_threshold: float = 0.0
    sparsity: float = 0.5
    router_patch: int = 4
    router_top_k: int = 1
    kwta_k: int = 4
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class EventDrivenSparseFoveatedSNN(hk.Module):
    """Dedicated event-driven sparse foveated SNN with dynamic route selection."""

    def __init__(self, cfg: EventDrivenSparseFoveatedConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.router = RegionActivationRouter(top_k=cfg.router_top_k, patch=cfg.router_patch)
        self.saliency = KWTASaliencyGate(k=cfg.kwta_k)
        self.fovea_conv = StructuredSparseConv2D(cfg.channels_fovea, kernel_size=3, sparsity=cfg.sparsity, padding="SAME")
        self.periph_conv = StructuredSparseConv2D(cfg.channels_periphery, kernel_size=3, sparsity=cfg.sparsity, padding="SAME")
        fh, fw = cfg.fovea_hw
        h, w = cfg.input_hw
        self.fovea_lif = snn.LIF((fh, fw, cfg.channels_fovea), beta=cfg.beta, threshold=cfg.threshold)
        self.periph_lif = snn.LIF((h // 2, w // 2, cfg.channels_periphery), beta=cfg.beta, threshold=cfg.threshold)
        self.route_proj = hk.Linear(cfg.output_dim)
        self.head = hk.Linear(cfg.output_dim)

    def _crop_fovea_from_scores(self, x: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        _, height, width, _ = x.shape
        patch = self.cfg.router_patch
        fh, fw = self.cfg.fovea_hw
        pw = max(width // patch, 1)
        best_idx = jnp.argmax(scores, axis=-1)
        center_y = (best_idx // pw) * patch + patch // 2
        center_x = (best_idx % pw) * patch + patch // 2
        start_y = jnp.clip(center_y - fh // 2, 0, height - fh)
        start_x = jnp.clip(center_x - fw // 2, 0, width - fw)
        return _crop_window_batch(x, start_y, start_x, self.cfg.fovea_hw)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        sf = self.fovea_lif.initial_state(batch)
        sp = self.periph_lif.initial_state(batch)

        def step(carry, x_t):
            c_f, c_p = carry
            event_mask = (jnp.abs(x_t) > self.cfg.event_threshold).astype(x_t.dtype)
            x_sparse = x_t * event_mask
            scores, region_gate = self.router(x_sparse)
            fovea = self._crop_fovea_from_scores(x_sparse, scores)
            fovea_feat, channel_gate = self.saliency(self.fovea_conv(fovea))
            periph = jax.image.resize(x_sparse, (x_sparse.shape[0], x_sparse.shape[1] // 2, x_sparse.shape[2] // 2, x_sparse.shape[3]), method="linear")
            s_f, c_f = self.fovea_lif(fovea_feat, c_f)
            s_p, c_p = self.periph_lif(self.periph_conv(periph), c_p)
            fused = jnp.concatenate([jnp.mean(s_f, axis=(1, 2)), jnp.mean(s_p, axis=(1, 2))], axis=-1)
            logits_t = self.head(fused) + self.route_proj(scores)
            sr_t = jnp.stack([jnp.mean(s_f), jnp.mean(s_p)])
            active_ratio = jnp.mean(event_mask)
            route_stats = jnp.mean(region_gate, axis=0)
            channel_stats = jnp.mean(channel_gate, axis=0)
            return (c_f, c_p), (logits_t, sr_t, active_ratio, route_stats, channel_stats)

        _, (logits_seq, sr_seq, active_seq, route_seq, channel_seq) = hk.scan(step, (sf, sp), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.mean(sr_seq, axis=0),
            "active_ratio": jnp.mean(active_seq),
            "region_gate": jnp.mean(route_seq, axis=0),
            "channel_gate": jnp.mean(channel_seq, axis=0),
        }


@dataclass
class SphericalRoutingGraphConfig:
    input_hw: tuple[int, int]
    input_channels: int
    hidden_dim: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class SphericalRoutingGraphSNN(hk.Module):
    """Spherical-neighborhood spike-routing proxy using wrapped local graph aggregation."""

    def __init__(self, cfg: SphericalRoutingGraphConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.proj = hk.Conv2D(cfg.hidden_dim, kernel_shape=1, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.hidden_dim), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def _graph_aggregate(self, x: jnp.ndarray) -> jnp.ndarray:
        up = jnp.roll(x, shift=-1, axis=1)
        down = jnp.roll(x, shift=1, axis=1)
        left = jnp.roll(x, shift=-1, axis=2)
        right = jnp.roll(x, shift=1, axis=2)
        return (x + up + down + left + right) / 5.0

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            x_g = self._graph_aggregate(x_t)
            s_t, carry = self.lif(self.proj(x_g), carry)
            logits_t = self.head(jnp.mean(s_t, axis=(1, 2)))
            return carry, (logits_t, jnp.mean(s_t), jnp.mean(jnp.abs(x_g - x_t)))

        _, (logits_seq, sr_seq, route_seq) = hk.scan(step, state, x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "route_delta": jnp.mean(route_seq),
        }


@dataclass
class SphericalFrequencyConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class SphericalFrequencyDomainSNN(hk.Module):
    """Frequency-domain spherical proxy branch feeding a spiking decoder."""

    def __init__(self, cfg: SphericalFrequencyConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.conv = hk.Conv2D(cfg.channels, kernel_shape=3, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.channels), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def _frequency_features(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        gray = jnp.mean(x, axis=-1)
        spec = jnp.fft.rfft2(gray, axes=(1, 2))
        mag = jnp.log1p(jnp.abs(spec))
        mag = mag[..., None]
        mag_up = jax.image.resize(mag, (x.shape[0], x.shape[1], x.shape[2], 1), method="linear")
        spectral_energy = jnp.mean(mag)
        return jnp.concatenate([x, mag_up], axis=-1), spectral_energy

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            x_f, e_t = self._frequency_features(x_t)
            s_t, carry = self.lif(self.conv(x_f), carry)
            logits_t = self.head(jnp.mean(s_t, axis=(1, 2)))
            return carry, (logits_t, jnp.mean(s_t), e_t)

        _, (logits_seq, sr_seq, energy_seq) = hk.scan(step, state, x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "spectral_energy": jnp.mean(energy_seq),
        }


@dataclass
class SmallLiquidStateMachineConfig:
    input_dim: int
    reservoir_dim: int
    output_dim: int
    recurrent_sparsity: float = 0.8
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class SmallLiquidStateMachineSNN(hk.Module):
    """Small liquid-state-machine style spiking reservoir block."""

    def __init__(self, cfg: SmallLiquidStateMachineConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.in_proj = hk.Linear(cfg.reservoir_dim)
        self.reservoir_lif = snn.LIF((cfg.reservoir_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.reservoir_lif.initial_state(batch)
        prev_spike = jnp.zeros((batch, self.cfg.reservoir_dim), dtype=x_seq.dtype)
        w_rec = hk.get_parameter(
            "w_rec",
            (self.cfg.reservoir_dim, self.cfg.reservoir_dim),
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        scores = jnp.abs(w_rec).reshape(-1)
        keep = max(1, int(scores.shape[0] * (1.0 - self.cfg.recurrent_sparsity)))
        mask = _topk_mask(scores[None, :], keep).reshape(w_rec.shape)
        w_eff = w_rec * mask

        def step(carry, x_t):
            c_state, c_prev = carry
            recurrent_drive = c_prev @ w_eff
            s_t, c_state = self.reservoir_lif(self.in_proj(x_t) + recurrent_drive, c_state)
            logits_t = self.head(s_t)
            return (c_state, s_t), (logits_t, jnp.mean(s_t))

        _, (logits_seq, sr_seq) = hk.scan(step, (state, prev_spike), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "recurrent_active_ratio": jnp.mean(mask),
        }


@dataclass
class DelayBasedConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    max_delay: int = 4
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class DelayBasedSpikingSNN(hk.Module):
    """Delay-line driven spiking block with learnable delay mixture."""

    def __init__(self, cfg: DelayBasedConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.hidden = hk.Linear(cfg.hidden_dim)
        self.lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.lif.initial_state(batch)
        delay_buf = jnp.zeros((self.cfg.max_delay, batch, self.cfg.input_dim), dtype=x_seq.dtype)
        delay_logits = hk.get_parameter("delay_logits", (self.cfg.max_delay,), init=jnp.zeros)
        delay_weights = jax.nn.softmax(delay_logits)

        def step(carry, x_t):
            c_state, c_buf = carry
            c_buf = jnp.concatenate([x_t[None, ...], c_buf[:-1]], axis=0)
            delayed = jnp.tensordot(delay_weights, c_buf, axes=([0], [0]))
            s_t, c_state = self.lif(self.hidden(delayed), c_state)
            logits_t = self.head(s_t)
            return (c_state, c_buf), (logits_t, jnp.mean(s_t))

        _, (logits_seq, sr_seq) = hk.scan(step, (state, delay_buf), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "delay_profile": delay_weights,
        }

@dataclass
class SpikeFrequencyCodingConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    max_frequency: int = 6
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class SpikeFrequencyCodedSNN(hk.Module):
    """Spike-frequency coding variant with periodic pulse generation."""

    def __init__(self, cfg: SpikeFrequencyCodingConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.hidden = hk.Linear(cfg.hidden_dim)
        self.lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def _frequency_code(self, x_seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        steps = x_seq.shape[0]
        static_x = jnp.mean(x_seq, axis=0)
        x_norm = _normalize_unit_interval(static_x)
        max_f = max(self.cfg.max_frequency, 1)
        freq = 1 + jnp.floor(x_norm * (max_f - 1)).astype(jnp.int32)
        period = jnp.maximum(1, max_f - freq + 1)
        t = jnp.arange(steps, dtype=jnp.int32)[:, None, None]
        code = (jnp.mod(t, period[None, :, :]) == 0).astype(x_seq.dtype)
        return code, freq.astype(x_seq.dtype)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.lif.initial_state(batch)
        coded_seq, freq_profile = self._frequency_code(x_seq)

        def step(carry, x_t):
            s_t, carry = self.lif(self.hidden(x_t), carry)
            logits_t = self.head(s_t)
            return carry, (logits_t, jnp.mean(s_t))

        _, (logits_seq, sr_seq) = hk.scan(step, state, coded_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "frequency_profile": jnp.mean(freq_profile, axis=0),
        }


@dataclass
class StrictGraphSphericalConfig:
    input_hw: tuple[int, int]
    input_channels: int
    hidden_dim: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


def _spherical_adjacency_matrix(height: int, width: int) -> jnp.ndarray:
    nodes = height * width
    adj = jnp.zeros((nodes, nodes), dtype=jnp.float32)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for y in range(height):
        for x in range(width):
            src = y * width + x
            adj = adj.at[src, src].set(1.0)
            for dy, dx in neighbors:
                ny = (y + dy) % height
                nx = (x + dx) % width
                dst = ny * width + nx
                adj = adj.at[src, dst].set(1.0)
    denom = jnp.maximum(jnp.sum(adj, axis=-1, keepdims=True), 1.0)
    return adj / denom


class StrictGraphSphericalSNN(hk.Module):
    """Strict graph-based spherical model with explicit node adjacency aggregation."""

    def __init__(self, cfg: StrictGraphSphericalConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.adj = _spherical_adjacency_matrix(h, w)
        self.proj = hk.Conv2D(cfg.hidden_dim, kernel_shape=1, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.hidden_dim), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def _aggregate(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        b, h, w, c = x.shape
        flat = x.reshape(b, h * w, c)
        agg = jnp.einsum("nm,bmc->bnc", self.adj, flat)
        smooth = jnp.mean(jnp.abs(agg - flat))
        return agg.reshape(b, h, w, c), smooth

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            z_t = self.proj(x_t)
            z_g, smooth_t = self._aggregate(z_t)
            s_t, carry = self.lif(z_g, carry)
            logits_t = self.head(jnp.mean(s_t, axis=(1, 2)))
            return carry, (logits_t, jnp.mean(s_t), smooth_t)

        _, (logits_seq, sr_seq, smooth_seq) = hk.scan(step, state, x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "graph_smoothness": jnp.mean(smooth_seq),
        }


@dataclass
class TinySpikingTransformerConfig:
    input_dim: int
    model_dim: int
    output_dim: int
    num_heads: int = 2
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class TinySpikingTransformerSNN(hk.Module):
    """Tiny transformer-style temporal attention feeding a spiking readout."""

    def __init__(self, cfg: TinySpikingTransformerConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.q_proj = hk.Linear(cfg.model_dim)
        self.k_proj = hk.Linear(cfg.model_dim)
        self.v_proj = hk.Linear(cfg.model_dim)
        self.o_proj = hk.Linear(cfg.model_dim)
        self.spike_proj = hk.Linear(cfg.model_dim)
        self.lif = snn.LIF((cfg.model_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        t, batch, _ = x_seq.shape
        x_bt = jnp.swapaxes(x_seq, 0, 1)
        q = self.q_proj(x_bt)
        k = self.k_proj(x_bt)
        v = self.v_proj(x_bt)
        scale = 1.0 / jnp.sqrt(jnp.asarray(self.cfg.model_dim, dtype=x_seq.dtype))
        scores = jnp.einsum("btd,bsd->bts", q, k) * scale
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bts,bsd->btd", attn, v)
        ctx = self.o_proj(ctx)
        ctx_seq = jnp.swapaxes(ctx, 0, 1)

        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            s_t, carry = self.lif(self.spike_proj(x_t), carry)
            logits_t = self.head(s_t)
            return carry, (logits_t, jnp.mean(s_t))

        _, (logits_seq, sr_seq) = hk.scan(step, state, ctx_seq)
        attn_entropy = -jnp.mean(jnp.sum(attn * jnp.log(jnp.maximum(attn, 1e-8)), axis=-1))
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "attention_entropy": attn_entropy,
        }


@dataclass
class BioDetailedSTDPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    stdp_lr: float = 0.05
    adapt_rate: float = 0.02
    target_rate: float = 0.1
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class BioDetailedSTDPSNN(hk.Module):
    """Bio-inspired adaptive-threshold spiking block with STDP-like trace dynamics."""

    def __init__(self, cfg: BioDetailedSTDPConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.in_proj = hk.Linear(cfg.hidden_dim)
        self.post_proj = hk.Linear(cfg.hidden_dim)
        self.lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.lif.initial_state(batch)
        pre_trace = jnp.zeros((batch, self.cfg.hidden_dim), dtype=x_seq.dtype)
        post_trace = jnp.zeros((batch, self.cfg.hidden_dim), dtype=x_seq.dtype)
        adapt_th = jnp.zeros((batch, self.cfg.hidden_dim), dtype=x_seq.dtype)

        def step(carry, x_t):
            c_state, c_pre, c_post, c_th = carry
            pre = self.in_proj(x_t)
            drive = pre - c_th
            s_t, c_state = self.lif(drive, c_state)
            c_pre = 0.95 * c_pre + 0.05 * pre
            c_post = 0.95 * c_post + 0.05 * self.post_proj(s_t)
            stdp_signal = jnp.mean(c_pre * c_post)
            spike_rate = jnp.mean(s_t, axis=-1, keepdims=True)
            c_th = 0.95 * c_th + self.cfg.adapt_rate * (spike_rate - self.cfg.target_rate)
            logits_t = self.head(s_t) + self.cfg.stdp_lr * stdp_signal
            return (c_state, c_pre, c_post, c_th), (logits_t, jnp.mean(s_t), stdp_signal, jnp.mean(c_th))

        _, (logits_seq, sr_seq, stdp_seq, th_seq) = hk.scan(step, (state, pre_trace, post_trace, adapt_th), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "stdp_signal": jnp.mean(stdp_seq),
            "adaptive_threshold": jnp.mean(th_seq),
        }


@dataclass
class StructuredSparseConvConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels1: int
    channels2: int
    output_dim: int
    sparsity: float = 0.5
    kernel_size: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    padding: str = "SAME"
    readout: str = "mean"


def _block_sparse_mask(w: jnp.ndarray, sparsity: float, block: int = 4) -> jnp.ndarray:
    h, w_k, cin, cout = w.shape
    bh = h // block
    bw = w_k // block
    if bh == 0 or bw == 0:
        return jnp.ones((h, w_k, 1, 1), dtype=w.dtype)
    core = w[: bh * block, : bw * block, :, :]
    blocks = core.reshape(bh, block, bw, block, cin, cout)
    scores = jnp.mean(jnp.abs(blocks), axis=(1, 3, 4, 5)).reshape(-1)
    keep = max(1, int(scores.shape[0] * (1.0 - sparsity)))
    gate = _topk_mask(scores[None, :], keep).reshape(bh, bw)
    gate = jnp.repeat(jnp.repeat(gate, block, axis=0), block, axis=1)
    mask = jnp.ones((h, w_k), dtype=w.dtype)
    mask = mask.at[: bh * block, : bw * block].set(gate)
    return mask[:, :, None, None]


class StructuredSparseConv2D(hk.Module):
    def __init__(self, out_ch: int, kernel_size: int = 3, sparsity: float = 0.5, padding: str = "SAME", name: str | None = None):
        super().__init__(name=name)
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.sparsity = sparsity
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_ch = x.shape[-1]
        w = hk.get_parameter("w", (self.kernel_size, self.kernel_size, in_ch, self.out_ch), init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"))
        mask = _block_sparse_mask(w, self.sparsity)
        w_s = w * mask
        return jax.lax.conv_general_dilated(x, w_s, (1, 1), self.padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))


class StructuredSparseSpikingCNN(hk.Module):
    def __init__(self, cfg: StructuredSparseConvConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.conv1 = StructuredSparseConv2D(cfg.channels1, kernel_size=cfg.kernel_size, sparsity=cfg.sparsity, padding=cfg.padding)
        self.conv2 = StructuredSparseConv2D(cfg.channels2, kernel_size=cfg.kernel_size, sparsity=cfg.sparsity, padding=cfg.padding)
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

        _, (logits_seq, sr_seq) = hk.scan(step, (v1, v2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.mean(sr_seq, axis=0)}


@dataclass
class EventDrivenPoolingConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels: int
    output_dim: int
    pool_modes: tuple[str, ...] = ("avg", "max", "l2")
    event_threshold: float = 0.0
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class EventDrivenPoolingSNN(hk.Module):
    """Spiking encoder with explicit event-conditioned pooling variants."""

    def __init__(self, cfg: EventDrivenPoolingConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.conv = hk.Conv2D(cfg.channels, kernel_shape=3, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.channels), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            s_t, carry = self.lif(self.conv(x_t), carry)
            pooled, active_ratio = _event_pool_features(s_t, self.cfg.event_threshold, self.cfg.pool_modes)
            logits_t = self.head(pooled)
            return carry, (logits_t, jnp.mean(s_t), active_ratio, pooled)

        _, (logits_seq, sr_seq, ar_seq, pooled_seq) = hk.scan(step, state, x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "active_ratio": jnp.mean(ar_seq),
            "pooled_features": pooled_seq,
        }


@dataclass
class TinySpikingAutoencoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class TinySpikingAutoencoder(hk.Module):
    """Compact sequence autoencoder with spiking encoder and decoder stages."""

    def __init__(self, cfg: TinySpikingAutoencoderConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.enc_proj = hk.Linear(cfg.hidden_dim)
        self.latent_proj = hk.Linear(cfg.latent_dim)
        self.dec_proj = hk.Linear(cfg.hidden_dim)
        self.recon_head = hk.Linear(cfg.input_dim)
        self.enc_lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.latent_lif = snn.LIF((cfg.latent_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.dec_lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        enc_state = self.enc_lif.initial_state(batch)
        latent_state = self.latent_lif.initial_state(batch)
        dec_state = self.dec_lif.initial_state(batch)

        def step(carry, x_t):
            c_enc, c_latent, c_dec = carry
            s_enc, c_enc = self.enc_lif(self.enc_proj(x_t), c_enc)
            z_t, c_latent = self.latent_lif(self.latent_proj(s_enc), c_latent)
            s_dec, c_dec = self.dec_lif(self.dec_proj(z_t), c_dec)
            recon_t = self.recon_head(s_dec)
            sr_t = jnp.stack([jnp.mean(s_enc), jnp.mean(z_t), jnp.mean(s_dec)])
            return (c_enc, c_latent, c_dec), (recon_t, z_t, sr_t)

        _, (recon_seq, latent_seq, sr_seq) = hk.scan(step, (enc_state, latent_state, dec_state), x_seq)
        return _readout(recon_seq, self.cfg.readout), {
            "recon_seq": recon_seq,
            "latent_seq": latent_seq,
            "spike_rate": jnp.mean(sr_seq, axis=0),
            "reconstruction_error": jnp.mean(jnp.abs(recon_seq - x_seq)),
        }


@dataclass
class PopulationCodingConfig:
    input_dim: int
    population_size: int
    hidden_dim: int
    output_dim: int
    sigma: float = 0.2
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class PopulationCodedLIFMLP(hk.Module):
    """Population-coded input expansion followed by a spiking MLP."""

    def __init__(self, cfg: PopulationCodingConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.hidden = hk.Linear(cfg.hidden_dim)
        self.head = hk.Linear(cfg.output_dim)
        self.lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.centers = jnp.linspace(0.0, 1.0, cfg.population_size)

    def _population_code(self, x_t: jnp.ndarray) -> jnp.ndarray:
        x_norm = jax.nn.sigmoid(x_t)
        diff = x_norm[..., None] - self.centers
        code = jnp.exp(-0.5 * (diff / jnp.maximum(self.cfg.sigma, 1e-6)) ** 2)
        return code.reshape(x_t.shape[0], -1)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            code_t = self._population_code(x_t)
            s_t, carry = self.lif(self.hidden(code_t), carry)
            logits_t = self.head(s_t)
            return carry, (logits_t, jnp.mean(s_t), jnp.mean(code_t))

        _, (logits_seq, sr_seq, code_seq) = hk.scan(step, state, x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "population_activity": jnp.asarray([jnp.mean(code_seq)]),
        }


@dataclass
class HardGatedMoEConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_experts: int
    top_k: int = 1
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class HardGatedMixtureOfExpertsSNN(hk.Module):
    """Hard top-k gated spiking mixture-of-experts block."""

    def __init__(self, cfg: HardGatedMoEConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.gate_proj = hk.Linear(cfg.num_experts)
        self.expert_proj = [hk.Linear(cfg.hidden_dim, name=f"expert_proj_{idx}") for idx in range(cfg.num_experts)]
        self.expert_lif = [snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold) for _ in range(cfg.num_experts)]
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        expert_states = tuple(expert.initial_state(batch) for expert in self.expert_lif)

        def step(carry, x_t):
            gate_scores = self.gate_proj(x_t)
            gate = _topk_mask(gate_scores, self.cfg.top_k)
            gate = gate / jnp.maximum(jnp.sum(gate, axis=-1, keepdims=True), 1.0)
            next_states = []
            expert_outs = []
            for state, proj, expert in zip(carry, self.expert_proj, self.expert_lif, strict=False):
                s_t, next_state = expert(proj(x_t), state)
                next_states.append(next_state)
                expert_outs.append(s_t)
            expert_stack = jnp.stack(expert_outs, axis=1)
            fused = jnp.sum(expert_stack * gate[:, :, None], axis=1)
            logits_t = self.head(fused)
            return tuple(next_states), (logits_t, jnp.mean(expert_stack), jnp.mean(gate, axis=0))

        _, (logits_seq, sr_seq, gate_seq) = hk.scan(step, expert_states, x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "expert_usage": jnp.mean(gate_seq, axis=0),
        }


@dataclass
class LatencyCodingConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class LatencyCodedSpikingHead(hk.Module):
    """Time-to-first-spike style head using explicit latency-coded inputs."""

    def __init__(self, cfg: LatencyCodingConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.hidden = hk.Linear(cfg.hidden_dim)
        self.head = hk.Linear(cfg.output_dim)
        self.lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)

    def _latency_code(self, x_seq: jnp.ndarray) -> jnp.ndarray:
        steps = x_seq.shape[0]
        static_x = jnp.mean(x_seq, axis=0)
        x_norm = _normalize_unit_interval(static_x)
        latency = jnp.round((1.0 - x_norm) * (steps - 1)).astype(jnp.int32)
        time_index = jnp.arange(steps, dtype=jnp.int32)[:, None, None]
        code = (time_index == latency[None, :, :]).astype(x_seq.dtype)
        return code

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        state = self.lif.initial_state(batch)
        coded_seq = self._latency_code(x_seq)

        def step(carry, x_t):
            s_t, carry = self.lif(self.hidden(x_t), carry)
            logits_t = self.head(s_t)
            return carry, (logits_t, s_t)

        _, (logits_seq, spike_seq) = hk.scan(step, state, coded_seq)
        spike_any = spike_seq > 0
        first_spike = jnp.argmax(spike_any, axis=0)
        has_spike = jnp.any(spike_any, axis=0)
        last_idx = jnp.full_like(first_spike, coded_seq.shape[0] - 1)
        first_spike = jnp.where(has_spike, first_spike, last_idx)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "first_spike_time": jnp.mean(first_spike.astype(jnp.float32), axis=-1),
            "latency_code_density": jnp.mean(coded_seq),
        }


@dataclass
class EarlyExitConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    confidence_threshold: float = 0.8
    readout: str = "mean"


class EarlyExitAnytimeSNN(hk.Module):
    """Anytime head that tracks earliest confident timestep."""

    def __init__(self, cfg: EarlyExitConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.body = hk.nets.MLP([cfg.hidden_dim, cfg.hidden_dim])
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        h_seq = self.body(x_seq)
        logits_seq = self.head(h_seq)
        probs = jax.nn.softmax(logits_seq, axis=-1)
        conf = jnp.max(probs, axis=-1)
        done = conf > self.cfg.confidence_threshold
        first = jnp.argmax(done, axis=0)
        any_done = jnp.any(done, axis=0)
        last_idx = jnp.full_like(first, x_seq.shape[0] - 1)
        exit_idx = jnp.where(any_done, first, last_idx)
        batch = jnp.arange(x_seq.shape[1])
        logits_exit = logits_seq[exit_idx, batch]
        return logits_exit, {
            "logits_seq": logits_seq,
            "exit_index": exit_idx,
            "early_exit_rate": jnp.mean(any_done.astype(jnp.float32)),
        }
