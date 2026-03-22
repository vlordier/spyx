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


def test_quantize_fixed_floor_rounding():
    cfg = qx.FixedPointConfig(total_bits=8, frac_bits=2, rounding="floor")
    x = jnp.array([-0.24, 0.24, 0.49], dtype=jnp.float32)
    y = qx.quantize_fixed(x, cfg, ste=False)
    expected = jnp.array([-0.25, 0.0, 0.25], dtype=jnp.float32)
    assert jnp.allclose(y, expected, atol=1e-6)


def test_quantize_fixed_max_abs_mode_clips_to_range():
    cfg = qx.FixedPointConfig(total_bits=8, frac_bits=2, scale_mode="max_abs")
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    y = qx.quantize_fixed(x, cfg, ste=False)
    assert jnp.max(jnp.abs(y)) <= 2.0 + 1e-6


def test_ternarize_weights_mean_scaled_strategy():
    w = jnp.array([-0.4, -0.1, 0.0, 0.09, 0.4], dtype=jnp.float32)
    y = qx.ternarize_weights(
        w,
        threshold=0.5,
        strategy="mean_scaled_threshold",
        ste=False,
    )
    assert set(jnp.unique(y).tolist()).issubset({-1.0, 0.0, 1.0})


def test_ternarize_weights_topk_strategy():
    w = jnp.array([-0.9, -0.3, -0.1, 0.1, 0.4, 0.8], dtype=jnp.float32)
    y = qx.ternarize_weights(w, strategy="topk", topk_ratio=0.34, ste=False)
    nonzero = int(jnp.count_nonzero(y))
    assert nonzero >= 1
    assert set(jnp.unique(y).tolist()).issubset({-1.0, 0.0, 1.0})


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


def test_ternary_fixed_point_linear_topk_forward_shape():
    def model(x):
        return qx.TernaryFixedPointLinear(
            4,
            strategy="topk",
            topk_ratio=0.25,
            cfg=qx.FixedPointConfig(total_bits=8, frac_bits=3),
        )(x)

    x = jnp.ones((3, 6), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(3), x)
    y = transformed.apply(params, x)
    assert y.shape == (3, 4)


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


def test_binary_linear_forward_shape():
    def model(x):
        return qx.BinaryLinear(6, cfg=qx.FixedPointConfig(8, 3))(x)

    x = jnp.ones((4, 5), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(4), x)
    y = transformed.apply(params, x)
    assert y.shape == (4, 6)


def test_fixed_point_conv2d_forward_shape():
    def model(x):
        conv = qx.FixedPointConv2D(
            output_channels=4,
            kernel_shape=(3, 3),
            stride=1,
            padding="SAME",
            cfg=qx.FixedPointConfig(8, 3),
        )
        return conv(x)

    x = jnp.ones((2, 8, 8, 3), dtype=jnp.float32)
    transformed = hk.without_apply_rng(hk.transform(model))
    params = transformed.init(jax.random.PRNGKey(5), x)
    y = transformed.apply(params, x)
    assert y.shape == (2, 8, 8, 4)
