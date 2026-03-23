"""Small end-to-end SHD run for the tiny spiking transformer Spyx model."""

from __future__ import annotations

import argparse

import spyx.loaders as loaders
import spyx.models as fm

from .common import build_dataset, run_classification_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-t", type=int, default=64)
    parser.add_argument("--channels", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = loaders.SHD_loader(
        batch_size=args.batch_size,
        sample_T=args.sample_t,
        channels=args.channels,
        val_size=0.2,
    )
    dataset = build_dataset(loader, args.sample_t, train_limit=args.train_limit, eval_limit=args.eval_limit)

    def model_factory() -> fm.TinySpikingTransformerSNN:
        cfg = fm.TinySpikingTransformerConfig(
            input_dim=args.channels,
            model_dim=96,
            output_dim=loader.act_shape[0],
            num_heads=2,
            beta=0.9,
            threshold=1.0,
        )
        return fm.TinySpikingTransformerSNN(cfg)

    run_classification_experiment(
        name="shd_tiny_transformer",
        dataset=dataset,
        model_factory=model_factory,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()