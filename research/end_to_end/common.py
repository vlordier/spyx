"""Shared utilities for small end-to-end Spyx research experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import optax


Array = jnp.ndarray
ModelFactory = Callable[[], Any]


@dataclass(frozen=True)
class ClassificationDataset:
    train_obs: Array
    train_labels: Array
    val_obs: Array
    val_labels: Array
    test_obs: Array
    test_labels: Array
    sample_T: int


@dataclass(frozen=True)
class RegressionDataset:
    """Dataset for regression tasks where targets are continuous-valued vectors."""

    train_obs: Array
    train_targets: Array  # shape: (N, target_dim)
    val_obs: Array
    val_targets: Array
    test_obs: Array
    test_targets: Array
    sample_T: int
    target_dim: int


@dataclass(frozen=True)
class ExperimentArtifacts:
    transformed: hk.Transformed
    optimizer: optax.GradientTransformation


def _truncate_split(obs: Array, labels: Array, limit: int | None) -> tuple[Array, Array]:
    if limit is None:
        return obs, labels
    return obs[:limit], labels[:limit]


def build_dataset(loader, sample_T: int, train_limit: int | None = None, eval_limit: int | None = None) -> ClassificationDataset:
    train_obs, train_labels = _truncate_split(loader.x_train, loader.y_train, train_limit)
    val_obs, val_labels = _truncate_split(loader.x_val, loader.y_val, eval_limit)
    test_obs, test_labels = _truncate_split(loader.x_test, loader.y_test, eval_limit)
    return ClassificationDataset(
        train_obs=train_obs,
        train_labels=train_labels,
        val_obs=val_obs,
        val_labels=val_labels,
        test_obs=test_obs,
        test_labels=test_labels,
        sample_T=sample_T,
    )


def unpack_time_major(obs_batch: Array, sample_T: int) -> Array:
    unpacked = jnp.unpackbits(obs_batch, axis=1)
    unpacked = unpacked[:, :sample_T]
    return jnp.swapaxes(unpacked.astype(jnp.float32), 0, 1)


def make_batches(obs: Array, labels: Array, batch_size: int, shuffle_key: Array | None = None) -> tuple[Array, Array]:
    count = labels.shape[0]
    usable = (count // batch_size) * batch_size
    if usable == 0:
        raise ValueError(f"Batch size {batch_size} is larger than available samples {count}")
    if shuffle_key is not None:
        indices = jax.random.permutation(shuffle_key, count)[:usable]
        obs = obs[indices]
        labels = labels[indices]
    else:
        obs = obs[:usable]
        labels = labels[:usable]
    obs = obs.reshape((usable // batch_size, batch_size) + obs.shape[1:])
    labels = labels.reshape((usable // batch_size, batch_size))
    return obs, labels


def _aux_scalar(aux: dict[str, object], name: str) -> jnp.ndarray:
    if name not in aux:
        return jnp.asarray(jnp.nan, dtype=jnp.float32)
    return jnp.asarray(jnp.mean(jnp.asarray(aux[name])), dtype=jnp.float32)


def build_experiment(model_factory: ModelFactory, sample_input: Array, learning_rate: float, seed: int) -> tuple[ExperimentArtifacts, hk.Params, optax.OptState]:
    def forward(x: Array):
        model = model_factory()
        return model(x)

    transformed = hk.without_apply_rng(hk.transform(forward))
    params = transformed.init(jax.random.PRNGKey(seed), sample_input)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return ExperimentArtifacts(transformed=transformed, optimizer=optimizer), params, opt_state


def make_train_step(artifacts: ExperimentArtifacts):
    @jax.jit
    def train_step(params: hk.Params, opt_state: optax.OptState, obs: Array, labels: Array):
        def loss_fn(current_params: hk.Params):
            logits, aux = artifacts.transformed.apply(current_params, obs)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
            metrics = {
                "loss": loss,
                "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == labels),
                "spike_rate": _aux_scalar(aux, "spike_rate"),
                "active_ratio": _aux_scalar(aux, "active_ratio"),
                "attention_entropy": _aux_scalar(aux, "attention_entropy"),
                "graph_smoothness": _aux_scalar(aux, "graph_smoothness"),
                "reconstruction_error": _aux_scalar(aux, "reconstruction_error"),
            }
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = artifacts.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    return train_step


def make_eval_step(artifacts: ExperimentArtifacts):
    @jax.jit
    def eval_step(params: hk.Params, obs: Array, labels: Array):
        logits, aux = artifacts.transformed.apply(params, obs)
        return {
            "loss": jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels)),
            "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == labels),
            "spike_rate": _aux_scalar(aux, "spike_rate"),
            "active_ratio": _aux_scalar(aux, "active_ratio"),
            "attention_entropy": _aux_scalar(aux, "attention_entropy"),
            "graph_smoothness": _aux_scalar(aux, "graph_smoothness"),
            "reconstruction_error": _aux_scalar(aux, "reconstruction_error"),
        }

    return eval_step


def summarize_epoch(metrics_seq: list[dict[str, Array]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in metrics_seq[0]:
        stacked = jnp.stack([jnp.asarray(item[key]) for item in metrics_seq])
        value = float(jnp.mean(stacked))
        if jnp.isnan(stacked).all():
            continue
        summary[key] = value
    return summary


def make_train_step_regression(artifacts: ExperimentArtifacts, target_dim: int):
    @jax.jit
    def train_step(params: hk.Params, opt_state: optax.OptState, obs: Array, targets: Array):
        def loss_fn(current_params: hk.Params):
            preds, aux = artifacts.transformed.apply(current_params, obs)
            loss = jnp.mean((preds - targets) ** 2)
            metrics = {
                "loss": loss,
                "mse": loss,
                "mae": jnp.mean(jnp.abs(preds - targets)),
                "spike_rate": _aux_scalar(aux, "spike_rate"),
                "active_ratio": _aux_scalar(aux, "active_ratio"),
            }
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = artifacts.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    return train_step


def make_eval_step_regression(artifacts: ExperimentArtifacts):
    @jax.jit
    def eval_step(params: hk.Params, obs: Array, targets: Array):
        preds, aux = artifacts.transformed.apply(params, obs)
        return {
            "loss": jnp.mean((preds - targets) ** 2),
            "mse": jnp.mean((preds - targets) ** 2),
            "mae": jnp.mean(jnp.abs(preds - targets)),
            "spike_rate": _aux_scalar(aux, "spike_rate"),
            "active_ratio": _aux_scalar(aux, "active_ratio"),
        }

    return eval_step


def run_regression_experiment(
    *,
    name: str,
    dataset: RegressionDataset,
    model_factory: ModelFactory,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run a regression experiment using MSE loss.

    Each sample consists of an observation tensor (e.g. rasterized events)
    and a continuous-valued target vector (e.g. 6DOF pose delta).
    """
    sample_input = unpack_time_major(dataset.train_obs[:batch_size], dataset.sample_T)
    artifacts, params, opt_state = build_experiment(model_factory, sample_input, learning_rate, seed)
    train_step = make_train_step_regression(artifacts, dataset.target_dim)
    eval_step = make_eval_step_regression(artifacts)

    rng = jax.random.PRNGKey(seed)
    history: dict[str, dict[str, float]] = {}

    for epoch in range(epochs):
        rng, train_key = jax.random.split(rng)
        train_obs_batches, train_target_batches = make_batches(
            dataset.train_obs, dataset.train_targets, batch_size, train_key
        )
        train_metrics = []
        for obs_batch, target_batch in zip(train_obs_batches, train_target_batches, strict=False):
            obs = unpack_time_major(obs_batch, dataset.sample_T)
            params, opt_state, metrics = train_step(params, opt_state, obs, target_batch)
            train_metrics.append(metrics)

        val_obs_batches, val_target_batches = make_batches(dataset.val_obs, dataset.val_targets, batch_size)
        val_metrics = []
        for obs_batch, target_batch in zip(val_obs_batches, val_target_batches, strict=False):
            obs = unpack_time_major(obs_batch, dataset.sample_T)
            val_metrics.append(eval_step(params, obs, target_batch))

        train_summary = summarize_epoch(train_metrics)
        val_summary = summarize_epoch(val_metrics)
        history[f"epoch_{epoch + 1}"] = {
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"val_{key}": value for key, value in val_summary.items()},
        }
        print(
            f"[{name}] epoch={epoch + 1} "
            f"train_mse={train_summary['mse']:.6f} train_mae={train_summary['mae']:.6f} "
            f"val_mse={val_summary['mse']:.6f} val_mae={val_summary['mae']:.6f}"
        )

    test_obs_batches, test_target_batches = make_batches(
        dataset.test_obs, dataset.test_targets, batch_size
    )
    test_metrics = []
    for obs_batch, target_batch in zip(test_obs_batches, test_target_batches, strict=False):
        obs = unpack_time_major(obs_batch, dataset.sample_T)
        test_metrics.append(eval_step(params, obs, target_batch))
    history["test"] = summarize_epoch(test_metrics)
    print(
        f"[{name}] test_mse={history['test']['mse']:.6f} "
        f"test_mae={history['test']['mae']:.6f}"
    )
    return history


def run_classification_experiment(
    *,
    name: str,
    dataset: ClassificationDataset,
    model_factory: ModelFactory,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    sample_input = unpack_time_major(dataset.train_obs[:batch_size], dataset.sample_T)
    artifacts, params, opt_state = build_experiment(model_factory, sample_input, learning_rate, seed)
    train_step = make_train_step(artifacts)
    eval_step = make_eval_step(artifacts)

    rng = jax.random.PRNGKey(seed)
    history: dict[str, dict[str, float]] = {}

    for epoch in range(epochs):
        rng, train_key = jax.random.split(rng)
        train_obs_batches, train_label_batches = make_batches(dataset.train_obs, dataset.train_labels, batch_size, train_key)
        train_metrics = []
        for obs_batch, label_batch in zip(train_obs_batches, train_label_batches, strict=False):
            obs = unpack_time_major(obs_batch, dataset.sample_T)
            params, opt_state, metrics = train_step(params, opt_state, obs, label_batch)
            train_metrics.append(metrics)

        val_obs_batches, val_label_batches = make_batches(dataset.val_obs, dataset.val_labels, batch_size)
        val_metrics = []
        for obs_batch, label_batch in zip(val_obs_batches, val_label_batches, strict=False):
            obs = unpack_time_major(obs_batch, dataset.sample_T)
            val_metrics.append(eval_step(params, obs, label_batch))

        train_summary = summarize_epoch(train_metrics)
        val_summary = summarize_epoch(val_metrics)
        history[f"epoch_{epoch + 1}"] = {
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"val_{key}": value for key, value in val_summary.items()},
        }
        print(
            f"[{name}] epoch={epoch + 1} "
            f"train_loss={train_summary['loss']:.4f} train_acc={train_summary['accuracy']:.4f} "
            f"val_loss={val_summary['loss']:.4f} val_acc={val_summary['accuracy']:.4f}"
        )

    test_obs_batches, test_label_batches = make_batches(dataset.test_obs, dataset.test_labels, batch_size)
    test_metrics = []
    for obs_batch, label_batch in zip(test_obs_batches, test_label_batches, strict=False):
        obs = unpack_time_major(obs_batch, dataset.sample_T)
        test_metrics.append(eval_step(params, obs, label_batch))
    history["test"] = summarize_epoch(test_metrics)
    print(
        f"[{name}] test_loss={history['test']['loss']:.4f} "
        f"test_acc={history['test']['accuracy']:.4f}"
    )
    return history