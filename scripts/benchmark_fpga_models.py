#!/usr/bin/env python3
"""Benchmark FPGA-friendly Spyx model templates and write TSV output."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp

import spyx.models as fm


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    sample_input: jnp.ndarray
    forward_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, dict[str, object]]]


def _to_scalar(x: object) -> float:
    arr = jnp.asarray(x)
    return float(jnp.mean(arr))


def _make_specs(time_steps: int, batch_size: int) -> list[BenchmarkSpec]:
    mlp_cfg = fm.MLPConfig(input_dim=32, hidden1=64, hidden2=32, output_dim=10)
    conv_cfg = fm.ConvConfig(input_hw=(16, 16), input_channels=2, channels1=8, channels2=12, output_dim=10)
    sparse_cfg = fm.SparseConvConfig(input_hw=(16, 16), input_channels=2, channels1=8, channels2=12, output_dim=10, event_threshold=0.2)
    dws_cfg = fm.DepthwiseSepConvConfig(
        input_hw=(16, 16),
        input_channels=2,
        depth_multiplier1=1,
        pointwise1=8,
        depth_multiplier2=1,
        pointwise2=12,
        output_dim=10,
    )
    residual_cfg = fm.ResidualConvConfig(input_hw=(16, 16), input_channels=2, stem_channels=8, block_channels=8, output_dim=10)
    mt_cfg = fm.MultiTimescaleConfig(input_dim=32, hidden_dim=32, output_dim=10)
    recurrent_cfg = fm.RecurrentBlockConfig(input_dim=32, hidden_dim=32, output_dim=10)
    hybrid_cfg = fm.HybridEncoderConfig(input_hw=(16, 16), input_channels=2, channels1=8, channels2=12, head_hidden=16, output_dim=10)

    x_mlp = jnp.ones((time_steps, batch_size, mlp_cfg.input_dim), dtype=jnp.float32)
    x_conv = jnp.ones((time_steps, batch_size, conv_cfg.input_hw[0], conv_cfg.input_hw[1], conv_cfg.input_channels), dtype=jnp.float32)

    return [
        BenchmarkSpec("LIFMLP", x_mlp, lambda x: fm.LIFMLP(mlp_cfg)(x)),
        BenchmarkSpec("TernaryLIFMLP", x_mlp, lambda x: fm.TernaryLIFMLP(mlp_cfg)(x)),
        BenchmarkSpec("ConvLIFSNN", x_conv, lambda x: fm.ConvLIFSNN(conv_cfg)(x)),
        BenchmarkSpec("TernaryConvLIFSNN", x_conv, lambda x: fm.TernaryConvLIFSNN(conv_cfg)(x)),
        BenchmarkSpec("SparseEventConvLIFSNN", x_conv, lambda x: fm.SparseEventConvLIFSNN(sparse_cfg)(x)),
        BenchmarkSpec("DepthwiseSeparableConvLIFSNN", x_conv, lambda x: fm.DepthwiseSeparableConvLIFSNN(dws_cfg)(x)),
        BenchmarkSpec("ResidualShallowSpikingCNN", x_conv, lambda x: fm.ResidualShallowSpikingCNN(residual_cfg)(x)),
        BenchmarkSpec("MultiTimescaleLIFBlock", x_mlp, lambda x: fm.MultiTimescaleLIFBlock(mt_cfg)(x)),
        BenchmarkSpec("TinyRecurrentSpikingBlock", x_mlp, lambda x: fm.TinyRecurrentSpikingBlock(recurrent_cfg)(x)),
        BenchmarkSpec("HybridSNNEncoderHead", x_conv, lambda x: fm.HybridSNNEncoderHead(hybrid_cfg)(x)),
    ]


def _benchmark_one(spec: BenchmarkSpec, warmup: int, repeats: int, seed: int) -> dict[str, object]:
    transformed = hk.without_apply_rng(hk.transform(spec.forward_fn))
    params = transformed.init(jax.random.PRNGKey(seed), spec.sample_input)
    apply_fn = jax.jit(lambda p, x: transformed.apply(p, x))

    for _ in range(warmup):
        logits, _ = apply_fn(params, spec.sample_input)
        _ = jax.block_until_ready(logits)

    elapsed = []
    last_aux: object = None
    logits_shape = ()
    for _ in range(repeats):
        t0 = time.perf_counter()
        logits, aux = apply_fn(params, spec.sample_input)
        _ = jax.block_until_ready(logits)
        elapsed.append((time.perf_counter() - t0) * 1000.0)
        logits_shape = tuple(logits.shape)
        last_aux = aux

    row: dict[str, object] = {
        "model": spec.name,
        "params": fm.count_parameters(params),
        "batch": spec.sample_input.shape[1],
        "timesteps": spec.sample_input.shape[0],
        "latency_ms_mean": sum(elapsed) / len(elapsed),
        "latency_ms_min": min(elapsed),
        "latency_ms_max": max(elapsed),
        "logits_shape": str(logits_shape),
        "spike_rate_mean": "",
        "active_ratio": "",
    }

    if isinstance(last_aux, dict):
        if "spike_rate" in last_aux:
            row["spike_rate_mean"] = _to_scalar(last_aux["spike_rate"])
        if "active_ratio" in last_aux:
            row["active_ratio"] = _to_scalar(last_aux["active_ratio"])

    return row


def _write_tsv(rows: list[dict[str, object]], output_path: Path) -> None:
    header = [
        "model",
        "params",
        "batch",
        "timesteps",
        "latency_ms_mean",
        "latency_ms_min",
        "latency_ms_max",
        "logits_shape",
        "spike_rate_mean",
        "active_ratio",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        sep = chr(9)
        nl = chr(10)
        f.write(sep.join(header) + nl)
        for row in rows:
            f.write(sep.join(str(row[col]) for col in header) + nl)

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Spyx FPGA-friendly model templates")
    parser.add_argument("--output", type=Path, default=Path("research/misc/fpga_model_bench.tsv"), help="Path for TSV output")
    parser.add_argument("--timesteps", type=int, default=8, help="Sequence length used for benchmarks")
    parser.add_argument("--batch", type=int, default=4, help="Batch size used for benchmarks")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations per model")
    args = parser.parse_args()

    specs = _make_specs(time_steps=args.timesteps, batch_size=args.batch)
    rows = []
    for i, spec in enumerate(specs):
        rows.append(_benchmark_one(spec, warmup=args.warmup, repeats=args.repeats, seed=100 + i))

    _write_tsv(rows, args.output)
    print(f"Wrote {len(rows)} benchmark rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
