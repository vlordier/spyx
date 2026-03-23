"""Small end-to-end DVS Gesture run for routing and foveated Spyx vision models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from tonic import datasets, transforms

from spyx.models.vision import (
    EventDrivenSparseFoveatedConfig,
    EventDrivenSparseFoveatedSNN,
    IntegratedWTAFoveatedSNN,
    WTAFoveatedStackConfig,
)

from .common import ClassificationDataset, run_classification_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("integrated-wta", "event-driven-sparse"), default="integrated-wta")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-t", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--spatial-factor", type=float, default=0.5)
    parser.add_argument("--save-to", default="data")
    return parser.parse_args()


def _cleanup_empty_archives(root: Path) -> None:
    for archive_name in ("ibmGestureTrain.tar.gz", "ibmGestureTest.tar.gz"):
        archive_path = root / archive_name
        if archive_path.exists() and archive_path.stat().st_size == 0:
            archive_path.unlink()


def _has_extracted_dvs_split(save_to: str, train: bool) -> bool:
    split_dir = Path(save_to) / "DVSGesture" / ("ibmGestureTrain" if train else "ibmGestureTest")
    return split_dir.exists() and any(split_dir.rglob("*.npy"))


def _load_dvs_gesture_split(save_to: str, train: bool, frame_transform: Any):
    try:
        return datasets.DVSGesture(save_to, train=train, transform=frame_transform)
    except RuntimeError:
        if not _has_extracted_dvs_split(save_to, train):
            raise

        class _LocalDVSGesture(datasets.DVSGesture):
            def _check_exists(self) -> bool:  # type: ignore[override]
                return True

        return _LocalDVSGesture(save_to, train=train, transform=frame_transform)


def _downsampled_sensor_size(sensor_size: tuple[int, int, int], spatial_factor: float) -> tuple[int, int, int]:
    height, width, channels = sensor_size
    return (max(1, int(round(height * spatial_factor))), max(1, int(round(width * spatial_factor))), channels)


def _subset_to_arrays(dataset, indices: list[int]) -> tuple[jnp.ndarray, jnp.ndarray]:
    obs = np.stack([dataset[index][0] for index in indices], axis=0)
    labels = np.asarray([dataset[index][1] for index in indices], dtype=np.uint8)
    return jnp.asarray(obs, dtype=jnp.uint8), jnp.asarray(labels, dtype=jnp.uint8)


def build_dvs_gesture_dataset(args: argparse.Namespace) -> tuple[ClassificationDataset, tuple[int, int, int], int]:
    data_root = Path(args.save_to) / "DVSGesture"
    data_root.mkdir(parents=True, exist_ok=True)
    _cleanup_empty_archives(data_root)

    sensor_size = _downsampled_sensor_size(datasets.DVSGesture.sensor_size, args.spatial_factor)
    frame_transform = transforms.Compose([
        transforms.Downsample(spatial_factor=args.spatial_factor),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.sample_t),
        lambda x: np.packbits(x, axis=0),
    ])

    try:
        train_val_dataset = _load_dvs_gesture_split(args.save_to, train=True, frame_transform=frame_transform)
        test_dataset = _load_dvs_gesture_split(args.save_to, train=False, frame_transform=frame_transform)
    except RuntimeError as exc:
        raise RuntimeError(
            "DVSGesture is not available locally and Tonic could not fetch a valid archive. "
            "Remove any stale files under data/DVSGesture and pre-download or extract "
            "ibmGestureTrain/ibmGestureTest before running this experiment."
        ) from exc

    train_indices, val_indices = train_test_split(
        list(range(len(train_val_dataset))),
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=train_val_dataset.targets,
    )

    if args.train_limit is not None:
        train_indices = train_indices[: args.train_limit]
    if args.eval_limit is not None:
        val_indices = val_indices[: args.eval_limit]
    test_indices = list(range(len(test_dataset)))
    if args.eval_limit is not None:
        test_indices = test_indices[: args.eval_limit]

    train_obs, train_labels = _subset_to_arrays(train_val_dataset, train_indices)
    val_obs, val_labels = _subset_to_arrays(train_val_dataset, val_indices)
    test_obs, test_labels = _subset_to_arrays(test_dataset, test_indices)

    dataset = ClassificationDataset(
        train_obs=train_obs,
        train_labels=train_labels,
        val_obs=val_obs,
        val_labels=val_labels,
        test_obs=test_obs,
        test_labels=test_labels,
        sample_T=args.sample_t,
    )
    return dataset, sensor_size, len(datasets.DVSGesture.classes)


def main() -> None:
    args = parse_args()
    dataset, sensor_size, output_dim = build_dvs_gesture_dataset(args)
    height, width, channels = sensor_size

    def model_factory() -> IntegratedWTAFoveatedSNN | EventDrivenSparseFoveatedSNN:
        if args.variant == "integrated-wta":
            cfg = WTAFoveatedStackConfig(
                input_hw=(height, width),
                input_channels=channels,
                fovea_hw=(24, 24),
                channels_fovea=16,
                channels_periphery=20,
                output_dim=output_dim,
                router_patch=8,
                router_top_k=1,
                kwta_k=6,
                tau=3.0,
                beta=0.9,
                threshold=1.0,
            )
            return IntegratedWTAFoveatedSNN(cfg)

        sparse_cfg = EventDrivenSparseFoveatedConfig(
            input_hw=(height, width),
            input_channels=channels,
            fovea_hw=(24, 24),
            channels_fovea=16,
            channels_periphery=20,
            output_dim=output_dim,
            event_threshold=0.0,
            sparsity=0.6,
            router_patch=8,
            router_top_k=1,
            kwta_k=6,
            beta=0.9,
            threshold=1.0,
        )
        return EventDrivenSparseFoveatedSNN(sparse_cfg)

    run_classification_experiment(
        name=f"dvs_gesture_{args.variant.replace('-', '_')}",
        dataset=dataset,
        model_factory=model_factory,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()