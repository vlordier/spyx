"""Small end-to-end NMNIST run for the log-polar foveated Spyx model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import spyx.loaders as loaders
import spyx.models as fm

from common import build_dataset, run_classification_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-t", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-subsample", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = loaders.NMNIST_loader(
        batch_size=args.batch_size,
        sample_T=args.sample_t,
        data_subsample=args.train_subsample,
        val_size=0.2,
        key=args.seed,
    )
    dataset = build_dataset(loader, args.sample_t, train_limit=args.train_limit, eval_limit=args.eval_limit)
    height, width, channels = loader.obs_shape

    def model_factory() -> fm.LogPolarFoveatedConvSNN:
        cfg = fm.LogPolarFoveatedConvConfig(
            input_hw=(height, width),
            input_channels=channels,
            radial_bins=12,
            angular_bins=16,
            channels1=16,
            channels2=24,
            output_dim=loader.act_shape[0],
            beta=0.9,
            threshold=1.0,
        )
        return fm.LogPolarFoveatedConvSNN(cfg)

    run_classification_experiment(
        name="nmnist_logpolar",
        dataset=dataset,
        model_factory=model_factory,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()