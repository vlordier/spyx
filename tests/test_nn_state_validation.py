import haiku as hk
import jax
import jax.numpy as jnp
import pytest

import spyx.nn as snn


def test_lif_raises_on_wrong_state_shape():
    def model(x, s):
        cell = snn.LIF(hidden_shape=(4,))
        return cell(x, s)

    x = jnp.ones((2, 4), dtype=jnp.float32)
    s_bad = jnp.zeros((2, 3), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(0), x, jnp.zeros((2, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="LIF state shape mismatch"):
        transformed.apply(params, x, s_bad)


def test_alif_raises_on_wrong_state_shape():
    def model(x, s):
        cell = snn.ALIF(hidden_shape=(4,))
        return cell(x, s)

    x = jnp.ones((2, 4), dtype=jnp.float32)
    s_bad = jnp.zeros((2, 7), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(1), x, jnp.zeros((2, 8), dtype=jnp.float32))

    with pytest.raises(ValueError, match="ALIF state shape mismatch"):
        transformed.apply(params, x, s_bad)
