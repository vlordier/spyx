"""Small end-to-end TUM-VIE run for a visual-inertial odometry Spyx model.

TUM-VIE is a stereo event-camera dataset with IMU measurements and ground-truth
6DOF poses.  This experiment uses a subset of a selected recording, rasterizes
the left/right event streams, and trains a small SNN to predict the 6DOF
pose delta (translation + rotation) from the event frames.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from tonic import transforms

import spyx.models as fm

from .common import RegressionDataset, run_regression_experiment


# Canonical drone/VI recording (6DOF mocap ground truth).
DEFAULT_RECORDING = "mocap-6dof"

# Output pose dimension: [tx, ty, tz, rx, ry, rz] — 3 translation + 3 euler rotation.
POSE_DIM = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default=DEFAULT_RECORDING,
                        help=f"TUM-VIE recording to load (default: {DEFAULT_RECORDING})")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-t", type=int, default=32,
                        help="Number of time bins for the rasterised event frame")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--spatial-factor", type=float, default=0.25,
                        help="Spatial downscale factor for the 1280×720 sensor")
    parser.add_argument("--save-to", default="data")
    parser.add_argument("--left-right", choices=("left", "right", "both"), default="both",
                        help="Which event stream(s) to use")
    return parser.parse_args()


def _downsample_sensor(sensor_size: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    h, w, c = sensor_size
    return (max(1, int(round(h * factor))), max(1, int(round(w * factor))), c)


def _load_tumvie_recording(
    save_to: str,
    recording: str,
    sensor_size: tuple[int, int, int],
    n_time_bins: int,
    left_right: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load a TUM-VIE recording and rasterise events + extract 6DOF poses.

    Returns:
        obs_list: list of rasterised event tensors, each shape (n_time_bins, H, W, C)
        pose_list: list of 6DOF pose vectors [tx, ty, tz, rx, ry, rz]
    """
    try:
        from tonic.datasets import TUMVIE
    except ImportError as exc:
        raise ImportError(
            "Tonic is required for TUM-VIE. Install it with: pip install spyx[loaders]"
        ) from exc

    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
    ])

    try:
        dataset = TUMVIE(save_to, recording=recording, transform=transform)
    except RuntimeError as exc:
        raise RuntimeError(
            f"TUM-VIE recording '{recording}' is not available locally and Tonic "
            "could not fetch it. Check that the dataset has been downloaded and "
            "extracted under data/TUMVIE/."
        ) from exc

    obs_list: list[np.ndarray] = []
    pose_list: list[np.ndarray] = []

    for idx in range(len(dataset)):
        data_dict, target_dict = dataset[idx]
        # data_dict keys: 'events_left', 'events_right', 'imu'
        # target_dict keys: 'images_left', 'images_right', 'mocap'
        if left_right == "both":
            ev = np.concatenate([data_dict["events_left"], data_dict["events_right"]], axis=-1)
        elif left_right == "left":
            ev = data_dict["events_left"]
        else:
            ev = data_dict["events_right"]

        # events shape after ToFrame: (n_time_bins, H, W, 1 or 2)
        ev = np.asarray(ev, dtype=np.uint8)
        obs_list.append(ev)

        # mocap: [tx, ty, tz, qx, qy, qz, qw] — 7 values, convert to 6DOF
        mocap = np.asarray(target_dict["mocap"], dtype=np.float32)
        # Use translation + Euler angles (rx, ry, rz) derived from quaternion
        tx, ty, tz = mocap[0], mocap[1], mocap[2]
        qx, qy, qz, qw = mocap[3], mocap[4], mocap[5], mocap[6]

        # Quaternion to Euler (XYZ convention — TUM-VIE convention)
        # Roll (rx), Pitch (ry), Yaw (rz)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        rx = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        ry = np.arcsin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        rz = np.arctan2(siny_cosp, cosy_cosp)

        pose = np.array([tx, ty, tz, rx, ry, rz], dtype=np.float32)
        pose_list.append(pose)

    return obs_list, pose_list


def _pack_and_stack(obs_list: list[np.ndarray]) -> jnp.ndarray:
    """Pack a list of (T, H, W, C) tensors into a (N, H, W, C*T) packed bit tensor."""
    packed = [np.packbits(o, axis=0) for o in obs_list]
    return jnp.asarray(packed, dtype=jnp.uint8)


def build_tumvie_dataset(args: argparse.Namespace) -> RegressionDataset:
    sensor_size = _downsample_sensor((1280, 720, 2), args.spatial_factor)
    obs_list, pose_list = _load_tumvie_recording(
        save_to=args.save_to,
        recording=args.recording,
        sensor_size=sensor_size,
        n_time_bins=args.sample_t,
        left_right=args.left_right,
    )

    # Normalise poses to zero-mean for regression stability
    all_poses = np.stack(pose_list, axis=0)
    pose_mean = np.mean(all_poses, axis=0, keepdims=True)

    # Train / val / test split
    indices = list(range(len(obs_list)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, shuffle=True
    )
    train_idx, test_idx = train_test_split(
        train_idx, test_size=0.2, random_state=args.seed + 1, shuffle=True
    )

    if args.train_limit is not None:
        train_idx = train_idx[: args.train_limit]
    if args.eval_limit is not None:
        val_idx = val_idx[: args.eval_limit]
        test_idx = test_idx[: args.eval_limit]

    def subset(arr_list: list, idx_list: list) -> jnp.ndarray:
        return jnp.asarray(np.stack([arr_list[i] for i in idx_list], axis=0), dtype=jnp.uint8)

    def subset_poses(arr_list: list, idx_list: list) -> jnp.ndarray:
        return jnp.asarray(np.stack([arr_list[i] for i in idx_list], axis=0), dtype=jnp.float32)

    train_obs = _pack_and_stack([obs_list[i] for i in train_idx])
    val_obs = _pack_and_stack([obs_list[i] for i in val_idx])
    test_obs = _pack_and_stack([obs_list[i] for i in test_idx])

    return RegressionDataset(
        train_obs=train_obs,
        train_targets=subset_poses(pose_list, train_idx),
        val_obs=val_obs,
        val_targets=subset_poses(pose_list, val_idx),
        test_obs=test_obs,
        test_targets=subset_poses(pose_list, test_idx),
        sample_T=args.sample_t,
        target_dim=POSE_DIM,
    )


def main() -> None:
    args = parse_args()
    dataset = build_tumvie_dataset(args)

    # Determine input channels from packed tensor shape
    # packed shape: (N, H, W, C_merged*T_bit_packed)
    # We decode enough to get H, W, C from the sensor size
    sensor_size = _downsample_sensor((1280, 720, 2), args.spatial_factor)
    n_channels = sensor_size[2] * (2 if args.left_right == "both" else 1)

    def model_factory() -> fm.ConvLIFSNN:
        cfg = fm.ConvConfig(
            input_hw=(sensor_size[0], sensor_size[1]),
            input_channels=n_channels,
            channels1=24,
            channels2=32,
            output_dim=POSE_DIM,
            kernel_size=3,
            padding="SAME",
            beta=0.9,
            threshold=1.0,
        )
        return fm.ConvLIFSNN(cfg)

    run_regression_experiment(
        name=f"tumvie_{args.recording}",
        dataset=dataset,
        model_factory=model_factory,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
