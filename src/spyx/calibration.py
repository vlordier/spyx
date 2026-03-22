"""
Quantization calibration workflow for post-training quantization (PTQ).

Collects statistics on network activations and weights to calibrate quantization
parameters for reduced precision inference while maintaining accuracy.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Callable, Any
import numpy as np


class CalibrationCollector:
    """
    Collects activation and weight statistics during network forward pass.
    Tracks min/max, percentiles, and histogram information for each layer.
    """

    def __init__(self, track_percentiles: bool = True, percentiles: list = [1, 5, 95, 99]):
        """
        Args:
            track_percentiles: Whether to track percentile values
            percentiles: Percentile values to track (0-100)
        """
        self.stats = {}
        self.track_percentiles = track_percentiles
        self.percentiles = percentiles

    def record_activation(self, name: str, x: jnp.ndarray):
        """Record activation statistics for a layer."""
        x_flat = jnp.reshape(x, (-1,))
        
        if name not in self.stats:
            self.stats[name] = {
                "type": "activation",
                "min": float("inf"),
                "max": float("-inf"),
                "mean": 0.0,
                "std": 0.0,
                "abs_max": 0.0,
                "count": 0,
                "sum": 0.0,
                "sq_sum": 0.0,
            }
            if self.track_percentiles:
                for p in self.percentiles:
                    self.stats[name][f"p{p}"] = 0.0
        
        stat = self.stats[name]
        x_np = np.array(x_flat)
        
        # Update running stats
        stat["min"] = float(min(stat["min"], np.min(x_np)))
        stat["max"] = float(max(stat["max"], np.max(x_np)))
        stat["abs_max"] = float(max(stat["abs_max"], np.abs(x_np).max()))
        stat["sum"] += float(np.sum(x_np))
        stat["sq_sum"] += float(np.sum(x_np ** 2))
        stat["count"] += len(x_np)
        
        # Update mean and std
        if stat["count"] > 0:
            stat["mean"] = stat["sum"] / stat["count"]
            stat["std"] = float(np.sqrt(max(0, stat["sq_sum"] / stat["count"] - stat["mean"] ** 2)))
        
        # Percentiles
        if self.track_percentiles:
            for p in self.percentiles:
                stat[f"p{p}"] = float(np.percentile(x_np, p))

    def record_weight(self, name: str, w: jnp.ndarray):
        """Record weight statistics for a layer."""
        w_flat = jnp.reshape(w, (-1,))
        
        if name not in self.stats:
            self.stats[name] = {
                "type": "weight",
                "min": float("inf"),
                "max": float("-inf"),
                "mean": 0.0,
                "std": 0.0,
                "abs_max": 0.0,
                "shape": w.shape,
            }
        
        stat = self.stats[name]
        w_np = np.array(w_flat)
        
        stat["min"] = float(min(stat["min"], np.min(w_np)))
        stat["max"] = float(max(stat["max"], np.max(w_np)))
        stat["abs_max"] = float(max(stat["abs_max"], np.abs(w_np).max()))
        stat["mean"] = float(np.mean(w_np))
        stat["std"] = float(np.std(w_np))

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return collected statistics."""
        return self.stats


class QuantizationCalibrator:
    """
    Calibrates quantization scales per-layer using collected activation statistics.
    Supports multiple calibration strategies: min-max, percentile-based, entropy-based.
    """

    def __init__(self, strategy: str = "minmax", bit_width: int = 8):
        """
        Args:
            strategy: Calibration strategy ("minmax", "percentile", "entropy")
            bit_width: Target quantization bit-width
        """
        self.strategy = strategy
        self.bit_width = bit_width
        self.scales = {}
        self.zero_points = {}

    def calibrate(self, stats: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """
        Compute quantization scales and zero-points from activation statistics.

        Args:
            stats: Activation statistics from CalibrationCollector

        Returns:
            Dict mapping layer name to (scale, zero_point) tuples
        """
        self.scales.clear()
        self.zero_points.clear()

        for name, stat in stats.items():
            if stat["type"] != "activation":
                continue

            if self.strategy == "minmax":
                scale, zp = self._calibrate_minmax(stat)
            elif self.strategy == "percentile":
                scale, zp = self._calibrate_percentile(stat)
            elif self.strategy == "entropy":
                scale, zp = self._calibrate_entropy(stat)
            else:
                raise ValueError(f"Unknown calibration strategy: {self.strategy}")

            self.scales[name] = scale
            self.zero_points[name] = zp

        return {name: (self.scales[name], self.zero_points[name]) for name in self.scales}

    def _calibrate_minmax(self, stat: Dict[str, Any]) -> Tuple[float, float]:
        """Min-max calibration: use observed range."""
        a_min = stat.get("min", 0.0)
        a_max = stat.get("max", 1.0)
        
        if a_min == a_max:
            return 1.0, 0.0  # Prevent division by zero

        # Symmetric quantization around 0
        abs_max = max(abs(a_min), abs(a_max))
        qmax = (1 << (self.bit_width - 1)) - 1  # e.g., 127 for uint8
        scale = abs_max / qmax if qmax > 0 else 1.0
        zero_point = 0.0

        return scale, zero_point

    def _calibrate_percentile(self, stat: Dict[str, Any], percentile: float = 99.9) -> Tuple[float, float]:
        """Percentile-based calibration: clip outliers."""
        p_key = f"p{percentile}"
        if p_key in stat:
            threshold = stat[p_key]
        else:
            threshold = stat.get("abs_max", 1.0)

        qmax = (1 << (self.bit_width - 1)) - 1
        scale = threshold / qmax if qmax > 0 else 1.0
        zero_point = 0.0

        return scale, zero_point

    def _calibrate_entropy(self, stat: Dict[str, Any]) -> Tuple[float, float]:
        """Entropy-based calibration: KL divergence minimization."""
        # Simplified: use percentile as proxy for entropy-based approach
        return self._calibrate_percentile(stat, percentile=99)


def apply_quantization(
    x: jnp.ndarray,
    scale: float,
    zero_point: float = 0.0,
    bit_width: int = 8,
    symmetric: bool = True
) -> jnp.ndarray:
    """
    Apply affine quantization to activations.

    Args:
        x: Input tensor
        scale: Quantization scale
        zero_point: Quantization zero point
        bit_width: Bit-width for quantization
        symmetric: Use symmetric quantization (no zero-point adjustment)

    Returns:
        Quantized tensor (dequantized back to float for simulation)
    """
    if symmetric:
        zero_point = 0.0

    qmax = (1 << (bit_width - 1)) - 1
    qmin = -(1 << (bit_width - 1))

    # Quantize
    q_x = jnp.round((x - zero_point) / scale)
    q_x = jnp.clip(q_x, qmin, qmax)

    # Dequantize (for simulation)
    x_dq = q_x *scale + zero_point

    return x_dq


def run_calibration_sweep(
    model_fn: Callable,
    calibration_data: Tuple[jnp.ndarray, jnp.ndarray],
    strategies: list = ["minmax", "percentile"],
    bit_widths: list = [8, 4],
    eval_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run a sweep over calibration strategies and bit-widths.
    Evaluates effect on model accuracy after quantization.

    Args:
        model_fn: Callable that runs model inference
        calibration_data: (x_cal, y_cal) calibration dataset
        strategies: List of calibration strategies to sweep
        bit_widths: List of bit-widths to sweep
        eval_fn: Optional evaluation function (x, y) -> accuracy

    Returns:
        Dict with sweep results: {strategy: {bit_width: metrics}}
    """
    results = {}
    x_cal, y_cal = calibration_data

    collector = CalibrationCollector()

    # Collect statistics
    for x_sample in x_cal:
        # Forward pass with collection hooking
        _ = model_fn(x_sample)  # In practice, would hook into forward to collect
        # (This is a simplified example; real implementation needs layer-specific hooks)

    stats = collector.get_stats()

    for strategy in strategies:
        results[strategy] = {}
        calibrator = QuantizationCalibrator(strategy=strategy)
        scales = calibrator.calibrate(stats)

        for bw in bit_widths:
            # Apply quantization and evaluate
            accuracy = 0.0
            if eval_fn:
                # Evaluate with quantized model
                # (Simplified; real implementation would apply scales to layers)
                accuracy = eval_fn(x_cal, y_cal)

            results[strategy][bw] = {
                "accuracy": accuracy,
                "scales": scales,
            }

    return results
