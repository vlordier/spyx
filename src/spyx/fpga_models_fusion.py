"""Fusion and control-oriented FPGA-friendly SNN model templates for Spyx."""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from . import nn as snn
from .fpga_models_core import ConvConfig, ConvLIFSNN, _readout
from .fpga_models_vision import _classical_filter_bank, _event_pool_features, _normalize_unit_interval, _shift_horizontally, _topk_mask


@dataclass
class IMUConditionedConfig:
    vision_cfg: ConvConfig
    imu_dim: int
    imu_hidden: int
    gating: str = "late"  # late | hard
    readout: str = "mean"


class IMUConditionedVisualSNN(hk.Module):
    """Visual Conv-LIF encoder conditioned by IMU branch."""

    def __init__(self, cfg: IMUConditionedConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.vision = ConvLIFSNN(cfg.vision_cfg)
        self.imu_proj = hk.nets.MLP([cfg.imu_hidden, cfg.vision_cfg.output_dim])

    def __call__(self, x_seq: jnp.ndarray, imu_seq: jnp.ndarray):
        logits_v, aux_v = self.vision(x_seq)
        imu_feat = jnp.mean(self.imu_proj(imu_seq), axis=0)
        if self.cfg.gating == "hard":
            gate = (imu_feat > 0).astype(logits_v.dtype)
            logits = logits_v * gate
        else:
            logits = logits_v + imu_feat
        return logits, {"spike_rate": aux_v["spike_rate"], "imu_feature": imu_feat}


@dataclass
class VisualIMURecurrentConfig:
    vision_cfg: ConvConfig
    imu_dim: int
    traj_dim: int
    hidden_dim: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class VisualIMURecurrentFusionBlock(hk.Module):
    """Tiny recurrent fusion of visual, IMU, and trajectory latents."""

    def __init__(self, cfg: VisualIMURecurrentConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.vision = ConvLIFSNN(cfg.vision_cfg)
        self.in_proj = hk.Linear(cfg.hidden_dim)
        self.core = snn.RLIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray, imu_seq: jnp.ndarray, traj_seq: jnp.ndarray):
        _, aux = self.vision(x_seq)
        vis_seq = aux["logits_seq"]
        _, batch, _ = imu_seq.shape
        state = self.core.initial_state(batch)
        fuse_seq = jnp.concatenate([vis_seq, imu_seq, traj_seq], axis=-1)

        def step(carry, x_t):
            h_t, carry = self.core(self.in_proj(x_t), carry)
            y_t = self.head(h_t)
            return carry, y_t

        _, logits_seq = hk.scan(step, state, fuse_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": aux["spike_rate"]}


@dataclass
class KalmanFusionConfig:
    latent_dim: int
    output_dim: int
    readout: str = "mean"


class KalmanStyleSpikingFusionSurrogate(hk.Module):
    """Prediction-correction style latent fusion surrogate."""

    def __init__(self, cfg: KalmanFusionConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.predictor = hk.Linear(cfg.latent_dim)
        self.corrector = hk.Linear(cfg.latent_dim)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, visual_seq: jnp.ndarray, imu_seq: jnp.ndarray):
        pred = self.predictor(imu_seq)
        innovation = visual_seq - pred
        corr = pred + self.corrector(innovation)
        logits_seq = self.head(corr)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "innovation_norm": jnp.mean(jnp.abs(innovation))}


@dataclass
class OpticalFlowConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class SpikingOpticalFlowBranch(hk.Module):
    """Spike-compatible optical-flow proxy using temporal frame differences."""

    def __init__(self, cfg: OpticalFlowConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.flow_conv = hk.Conv2D(cfg.channels, kernel_shape=3, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.channels), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state = self.lif.initial_state(batch)
        flow_seq = x_seq[1:] - x_seq[:-1]

        def step(carry, x_t):
            s_t, carry = self.lif(self.flow_conv(x_t), carry)
            y_t = self.head(jnp.mean(s_t, axis=(1, 2)))
            return carry, (y_t, jnp.mean(s_t))

        _, (logits_seq, sr_seq) = hk.scan(step, state, flow_seq)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.asarray([jnp.mean(sr_seq)])}


@dataclass
class StereoCoincidenceConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class StereoCoincidenceSNN(hk.Module):
    """Local stereo coincidence/disparity proxy branch."""

    def __init__(self, cfg: StereoCoincidenceConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        self.conv = hk.Conv2D(cfg.channels, kernel_shape=3, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.channels), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, left_seq: jnp.ndarray, right_seq: jnp.ndarray):
        _, batch, _, _, _ = left_seq.shape
        state = self.lif.initial_state(batch)
        coincidence = jnp.concatenate([left_seq, right_seq, jnp.abs(left_seq - right_seq)], axis=-1)

        def step(carry, x_t):
            s_t, carry = self.lif(self.conv(x_t), carry)
            y_t = self.head(jnp.mean(s_t, axis=(1, 2)))
            return carry, (y_t, jnp.mean(s_t))

        _, (logits_seq, sr_seq) = hk.scan(step, state, coincidence)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "spike_rate": jnp.asarray([jnp.mean(sr_seq)])}


@dataclass
class StereoDisparityConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels: int
    output_dim: int
    max_disparity: int = 3
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class StereoDisparityCorrelationSNN(hk.Module):
    """Stereo cost-volume SNN with explicit disparity bins and consistency proxy."""

    def __init__(self, cfg: StereoDisparityConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        volume_channels = 2 * (cfg.max_disparity + 1)
        self.conv = hk.Conv2D(cfg.channels, kernel_shape=3, padding="SAME")
        self.lif = snn.LIF((h, w, cfg.channels), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)
        self.volume_proj = hk.Linear(cfg.max_disparity + 1)

    def _cost_volume(self, left: jnp.ndarray, right: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        corr_maps = []
        sad_maps = []
        disparity_score_list = []
        reverse_score_list = []
        for disparity in range(self.cfg.max_disparity + 1):
            shifted_right = _shift_horizontally(right, disparity)
            shifted_left = _shift_horizontally(left, disparity)
            corr = jnp.mean(left * shifted_right, axis=-1, keepdims=True)
            sad = jnp.mean(jnp.abs(left - shifted_right), axis=-1, keepdims=True)
            reverse_corr = jnp.mean(right * shifted_left, axis=-1)
            corr_maps.append(corr)
            sad_maps.append(sad)
            disparity_score_list.append(jnp.mean(corr, axis=(1, 2, 3)))
            reverse_score_list.append(jnp.mean(reverse_corr, axis=(1, 2)))
        disparity_scores = jnp.stack(disparity_score_list, axis=-1)
        reverse_scores = jnp.stack(reverse_score_list, axis=-1)
        best_lr = jnp.argmax(disparity_scores, axis=-1)
        best_rl = jnp.argmax(reverse_scores, axis=-1)
        consistency = jnp.mean(jnp.abs(best_lr - best_rl).astype(left.dtype))
        volume = jnp.concatenate(corr_maps + sad_maps, axis=-1)
        return volume, disparity_scores, consistency

    def __call__(self, left_seq: jnp.ndarray, right_seq: jnp.ndarray):
        _, batch, _, _, _ = left_seq.shape
        state = self.lif.initial_state(batch)

        def step(carry, x_t):
            left_t, right_t = x_t
            volume_t, disp_scores_t, consistency_t = self._cost_volume(left_t, right_t)
            s_t, carry = self.lif(self.conv(volume_t), carry)
            pooled = jnp.mean(s_t, axis=(1, 2))
            logits_t = self.head(pooled)
            disp_logits_t = self.volume_proj(disp_scores_t)
            sr_t = jnp.mean(s_t)
            return carry, (logits_t + disp_logits_t, sr_t, disp_scores_t, consistency_t)

        _, (logits_seq, sr_seq, disparity_seq, consistency_seq) = hk.scan(step, state, (left_seq, right_seq))
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.asarray([jnp.mean(sr_seq)]),
            "disparity_scores": jnp.mean(disparity_seq, axis=(0, 1)),
            "left_right_consistency": jnp.mean(consistency_seq),
        }


@dataclass
class MotionCompConfig:
    vision_cfg: ConvConfig
    imu_scale: float = 1.0


class MotionCompensatedInputFrontEnd(hk.Module):
    """IMU-conditioned coarse de-rotation front-end with visual encoder."""

    def __init__(self, cfg: MotionCompConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.encoder = ConvLIFSNN(cfg.vision_cfg)

    def __call__(self, x_seq: jnp.ndarray, imu_seq: jnp.ndarray):
        # Use first two IMU channels as coarse integer translation proxy.
        shift_xy = jnp.round(imu_seq[..., :2] * self.cfg.imu_scale).astype(jnp.int32)

        def shift_step(x_t, s_t):
            dx = s_t[:, 0]
            dy = s_t[:, 1]

            def shift_one(img, sx, sy):
                return jnp.roll(jnp.roll(img, sx, axis=0), sy, axis=1)

            return jax.vmap(shift_one)(x_t, dx, dy)

        x_comp = jax.vmap(shift_step, in_axes=(0, 0))(x_seq, shift_xy)
        logits, aux = self.encoder(x_comp)
        return logits, {"spike_rate": aux["spike_rate"], "x_comp": x_comp}


@dataclass
class HybridFilterConfig:
    input_hw: tuple[int, int]
    input_channels: int
    channels1: int
    channels2: int
    output_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class HybridClassicalFilterSNN(hk.Module):
    """Fixed classical filters feeding a trainable spiking encoder."""

    def __init__(self, cfg: HybridFilterConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        h, w = cfg.input_hw
        in_channels = cfg.input_channels + 3
        self.fuse = hk.Conv2D(cfg.channels1, kernel_shape=1, padding="SAME")
        self.conv = hk.Conv2D(cfg.channels2, kernel_shape=3, padding="SAME")
        self.lif1 = snn.LIF((h, w, cfg.channels1), beta=cfg.beta, threshold=cfg.threshold)
        self.lif2 = snn.LIF((h, w, cfg.channels2), beta=cfg.beta, threshold=cfg.threshold)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _, _, _ = x_seq.shape
        state1 = self.lif1.initial_state(batch)
        state2 = self.lif2.initial_state(batch)

        def step(carry, x_t):
            c1, c2 = carry
            x_filt, energy = _classical_filter_bank(x_t)
            s1, c1 = self.lif1(self.fuse(x_filt), c1)
            s2, c2 = self.lif2(self.conv(s1), c2)
            logits_t = self.head(jnp.mean(s2, axis=(1, 2)))
            sr_t = jnp.stack([jnp.mean(s1), jnp.mean(s2)])
            return (c1, c2), (logits_t, sr_t, energy)

        _, (logits_seq, sr_seq, energy_seq) = hk.scan(step, (state1, state2), x_seq)
        return _readout(logits_seq, self.cfg.readout), {
            "logits_seq": logits_seq,
            "spike_rate": jnp.mean(sr_seq, axis=0),
            "filter_energy": jnp.mean(energy_seq, axis=0),
        }


@dataclass
class GazeControlConfig:
    input_dim: int
    imu_dim: int
    traj_dim: int
    num_regions: int


class GazeControlPolicyHead(hk.Module):
    """Policy head predicting top-k gaze regions."""

    def __init__(self, cfg: GazeControlConfig, top_k: int = 1, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.top_k = top_k
        self.mlp = hk.nets.MLP([64, cfg.num_regions])

    def __call__(self, periph_seq: jnp.ndarray, imu_seq: jnp.ndarray, traj_seq: jnp.ndarray):
        x = jnp.concatenate([
            jnp.mean(periph_seq, axis=0),
            jnp.mean(imu_seq, axis=0),
            jnp.mean(traj_seq, axis=0),
        ], axis=-1)
        scores = self.mlp(x)
        gate = _topk_mask(scores, self.top_k)
        return scores, {"region_gate": gate}


@dataclass
class TrajectoryConditionedConfig:
    vision_dim: int
    imu_dim: int
    traj_dim: int
    hidden_dim: int
    output_dim: int
    readout: str = "mean"


class TrajectoryConditionedSpikingEncoder(hk.Module):
    """Trajectory-conditioned latent encoder with hard/soft gain modulation."""

    def __init__(self, cfg: TrajectoryConditionedConfig, hard_gate: bool = False, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.hard_gate = hard_gate
        self.base = hk.Linear(cfg.hidden_dim)
        self.gain = hk.Linear(cfg.hidden_dim)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, vis_seq: jnp.ndarray, imu_seq: jnp.ndarray, traj_seq: jnp.ndarray):
        x = jnp.concatenate([vis_seq, imu_seq, traj_seq], axis=-1)
        base = self.base(x)
        g = jax.nn.sigmoid(self.gain(traj_seq))
        if self.hard_gate:
            g = (g > 0.5).astype(g.dtype)
        fused = base * g
        logits_seq = self.head(fused)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "gate_mean": jnp.mean(g)}


@dataclass
class PredictiveCodingConfig:
    input_dim: int
    latent_dim: int
    output_dim: int
    readout: str = "mean"


class PredictiveCodingSNNBlock(hk.Module):
    """Residual-error predictive coding style block."""

    def __init__(self, cfg: PredictiveCodingConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.predictor = hk.Linear(cfg.latent_dim)
        self.encoder = hk.Linear(cfg.latent_dim)
        self.head = hk.Linear(cfg.output_dim)

    def __call__(self, obs_seq: jnp.ndarray):
        pred_seq = self.predictor(obs_seq)
        err = self.encoder(obs_seq) - pred_seq
        logits_seq = self.head(err)
        return _readout(logits_seq, self.cfg.readout), {"logits_seq": logits_seq, "error_norm": jnp.mean(jnp.abs(err))}


@dataclass
class MultiHeadConfig:
    input_dim: int
    hidden_dim: int
    collision_dim: int
    navigation_dim: int


class CollisionNavigationMultiHead(hk.Module):
    """Shared trunk with separate collision and navigation heads."""

    def __init__(self, cfg: MultiHeadConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.trunk = hk.nets.MLP([cfg.hidden_dim, cfg.hidden_dim])
        self.collision_head = hk.Linear(cfg.collision_dim)
        self.navigation_head = hk.Linear(cfg.navigation_dim)

    def __call__(self, x: jnp.ndarray):
        h = self.trunk(x)
        return {
            "collision": self.collision_head(h),
            "navigation": self.navigation_head(h),
        }


@dataclass
class SpikingMultiHeadConfig:
    input_dim: int
    hidden_dim: int
    collision_hidden: int
    navigation_hidden: int
    collision_dim: int
    navigation_dim: int
    beta: float = 0.9
    threshold: float = 1.0
    readout: str = "mean"


class SpikingCollisionNavigationMultiHead(hk.Module):
    """Fully spiking shared trunk with separate spiking collision and navigation heads."""

    def __init__(self, cfg: SpikingMultiHeadConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self.shared_proj = hk.Linear(cfg.hidden_dim)
        self.shared_lif = snn.LIF((cfg.hidden_dim,), beta=cfg.beta, threshold=cfg.threshold)
        self.collision_proj = hk.Linear(cfg.collision_hidden)
        self.navigation_proj = hk.Linear(cfg.navigation_hidden)
        self.collision_lif = snn.LIF((cfg.collision_hidden,), beta=cfg.beta, threshold=cfg.threshold)
        self.navigation_lif = snn.LIF((cfg.navigation_hidden,), beta=cfg.beta, threshold=cfg.threshold)
        self.collision_head = hk.Linear(cfg.collision_dim)
        self.navigation_head = hk.Linear(cfg.navigation_dim)

    def __call__(self, x_seq: jnp.ndarray):
        _, batch, _ = x_seq.shape
        shared_state = self.shared_lif.initial_state(batch)
        collision_state = self.collision_lif.initial_state(batch)
        navigation_state = self.navigation_lif.initial_state(batch)

        def step(carry, x_t):
            c_shared, c_collision, c_navigation = carry
            s_shared, c_shared = self.shared_lif(self.shared_proj(x_t), c_shared)
            s_collision, c_collision = self.collision_lif(self.collision_proj(s_shared), c_collision)
            s_navigation, c_navigation = self.navigation_lif(self.navigation_proj(s_shared), c_navigation)
            coll_t = self.collision_head(s_collision)
            nav_t = self.navigation_head(s_navigation)
            sr_t = jnp.stack([jnp.mean(s_shared), jnp.mean(s_collision), jnp.mean(s_navigation)])
            return (c_shared, c_collision, c_navigation), (coll_t, nav_t, sr_t)

        _, (collision_seq, navigation_seq, sr_seq) = hk.scan(step, (shared_state, collision_state, navigation_state), x_seq)
        return {
            "collision": _readout(collision_seq, self.cfg.readout),
            "navigation": _readout(navigation_seq, self.cfg.readout),
        }, {
            "collision_seq": collision_seq,
            "navigation_seq": navigation_seq,
            "spike_rate": jnp.mean(sr_seq, axis=0),
        }


