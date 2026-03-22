import jax
import jax.numpy as jnp

import spyx.axn as axn


def test_straight_through_has_unit_gradient():
    f = axn.straight_through()

    def loss(x):
        return jnp.sum(f(x))

    x = jnp.array([-1.0, -0.1, 0.0, 0.2, 2.0], dtype=jnp.float32)
    g = jax.grad(loss)(x)
    assert jnp.allclose(g, jnp.ones_like(x), atol=1e-6)


def test_sigmoid_surrogate_gradient_nonzero_near_threshold():
    f = axn.sigmoid(k=5)

    def loss(x):
        return jnp.sum(f(x))

    x = jnp.array([-0.2, -0.05, 0.0, 0.05, 0.2], dtype=jnp.float32)
    g = jax.grad(loss)(x)
    assert jnp.all(g > 0.0)
