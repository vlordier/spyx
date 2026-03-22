import jax
import jax.numpy as jnp
import pytest

from spyx.data import rate_code
from spyx.loaders import _drop_remainder


def test_rate_code_clips_values_and_returns_binary_spikes():
    encoder = rate_code(num_steps=3, max_r=1.0)
    key = jax.random.PRNGKey(0)
    x = jnp.array([[-0.5, 2.0]], dtype=jnp.float32)

    y = encoder(x, key)

    assert y.dtype == jnp.uint8
    assert y.shape == (1, 6)
    assert jnp.all(y[:, :3] == 0)
    assert jnp.all(y[:, 3:] == 1)


def test_rate_code_validates_constructor_args():
    with pytest.raises(ValueError, match="num_steps"):
        rate_code(num_steps=0)

    with pytest.raises(ValueError, match="max_r"):
        rate_code(num_steps=4, max_r=1.5)


def test_drop_remainder_keeps_full_array_when_cutoff_is_zero():
    x = jnp.arange(6)
    y = _drop_remainder(x, 0)
    assert jnp.array_equal(x, y)


def test_drop_remainder_trims_tail_when_cutoff_positive():
    x = jnp.arange(6)
    y = _drop_remainder(x, 2)
    assert jnp.array_equal(y, jnp.array([0, 1, 2, 3]))
