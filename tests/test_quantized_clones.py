import haiku as hk
import jax
import jax.numpy as jnp

import spyx.quantized as qx


def test_quantize_fixed_uses_expected_grid():
    cfg = qx.FixedPointConfig(total_bits=8, frac_bits=2)
    x = jnp.array([-10.0, -0.3, -0.24, 0.24, 0.49, 100.0], dtype=jnp.float32)

    y = qx.quantize_fixed(x, cfg, ste=False)

    # 2 fractional bits => grid step 0.25.
    expected = jnp.array([-10.0, -0.25, -0.25, 0.25, 0.5, 31.75], dtype=jnp.float32)
    assert jnp.allclose(y, expected, atol=1e-6)


def test_ternarize_weights_outputs_three_values():
    w = jnp.array([-0.5, -0.02, 0.0, 0.03, 0.5], dtype=jnp.float32)
    y = qx.ternarize_weights(w, threshold=0.05, ste=False)
    expected = jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.array_equal(y, expected)


def test_fixed_point_linear_forward_shape():
    def model(x):
        return qx.FixedPointLinear(5, cfg=qx.FixedPointConfig(8, 3))(x)

    x = jnp.ones((2, 4), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(0), x)
    y = transformed.apply(params, x)
    assert y.shape == (2, 5)


def test_ternary_linear_forward_shape():
    def model(x):
        return qx.TernaryLinear(3, threshold=0.1)(x)

    x = jnp.ones((7, 4), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(1), x)
    y = transformed.apply(params, x)
    assert y.shape == (7, 3)


def test_fixed_point_lif_state_shape():
    def model(xs, s0):
        cell = qx.FixedPointLIF(hidden_shape=(4,), cfg=qx.FixedPointConfig(8, 4))

        def step_fn(state, x_t):
            y, new_state = cell(x_t, state)
            return new_state, y

        sf, ys = hk.scan(step_fn, s0, xs)
        return ys, sf

    xs = jnp.ones((5, 2, 4), dtype=jnp.float32)
    s0 = jnp.zeros((2, 4), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(2), xs, s0)
    y, s = transformed.apply(params, xs, s0)
    assert y.shape == (5, 2, 4)
    assert s.shape == (2, 4)
