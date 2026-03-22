import haiku as hk
import jax
import jax.numpy as jnp

import spyx
import spyx.axn as axn
import spyx.fn as fn
import spyx.nn as nn

# ---------------------------------------------------------------------------
# Activation functions (axn)
# ---------------------------------------------------------------------------


def test_heaviside_zero():
    """Values at exactly 0 should produce 0 (strictly-greater-than threshold)."""
    result = axn.heaviside(jnp.array(0.0))
    assert float(result) == 0.0


def test_heaviside_positive():
    x = jnp.array([0.5, 1.0, 2.0])
    result = axn.heaviside(x)
    assert jnp.all(result == 1.0)


def test_heaviside_negative():
    x = jnp.array([-0.5, -1.0])
    result = axn.heaviside(x)
    assert jnp.all(result == 0.0)


def test_superspike_returns_callable():
    fn_sg = axn.superspike()
    assert callable(fn_sg)


def test_superspike_output_shape():
    fn_sg = axn.superspike()
    x = jnp.ones((4, 8))
    out = fn_sg(x - 1.0)  # threshold subtracted externally
    assert out.shape == (4, 8)


def test_arctan_returns_callable():
    fn_sg = axn.arctan()
    assert callable(fn_sg)


def test_boxcar_returns_callable():
    fn_sg = axn.boxcar()
    assert callable(fn_sg)


def test_triangular_returns_callable():
    fn_sg = axn.triangular()
    assert callable(fn_sg)


def test_tanh_returns_callable():
    fn_sg = axn.tanh()
    assert callable(fn_sg)


def test_custom_activation():
    fn_sg = axn.custom()
    x = jnp.array([0.5, -0.5])
    out = fn_sg(x)
    # default fwd is heaviside
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Neuron modules (nn)
# hk.dynamic_unroll defaults to time_major=True, so input is (time, batch, features).
# ---------------------------------------------------------------------------


def _unroll(core_cls, shape, timesteps=10, batch=2):
    """Helper: wrap a single neuron layer in a hk.transform and run it."""
    features = shape[0]

    def snn(x):
        core = hk.DeepRNN([core_cls(shape)])
        out, _ = hk.dynamic_unroll(core, x, core.initial_state(batch))
        return out

    net = hk.without_apply_rng(hk.transform(snn))
    # time-major input: (time, batch, features)
    x = jnp.zeros((timesteps, batch, features))
    params = net.init(jax.random.PRNGKey(0), x)
    return net, params, x


def test_lif_output_shape():
    timesteps, batch, features = 10, 2, 8
    net, params, x = _unroll(nn.LIF, (features,), timesteps, batch)
    out = net.apply(params, x)
    assert out.shape == (timesteps, batch, features)


def test_if_output_shape():
    timesteps, batch, features = 10, 2, 8
    net, params, x = _unroll(nn.IF, (features,), timesteps, batch)
    out = net.apply(params, x)
    assert out.shape == (timesteps, batch, features)


def test_li_output_shape():
    timesteps, batch, features = 10, 2, 8
    net, params, x = _unroll(nn.LI, (features,), timesteps, batch)
    out = net.apply(params, x)
    assert out.shape == (timesteps, batch, features)


def test_cubalif_output_shape():
    timesteps, batch, features = 10, 2, 8
    net, params, x = _unroll(nn.CuBaLIF, (features,), timesteps, batch)
    out = net.apply(params, x)
    assert out.shape == (timesteps, batch, features)


def test_lif_spikes_binary():
    """LIF output should only contain 0 or 1 (binary spikes)."""
    timesteps, batch, features = 20, 4, 16

    def snn(x):
        core = hk.DeepRNN([nn.LIF((features,))])
        spikes, _ = hk.dynamic_unroll(core, x, core.initial_state(batch))
        return spikes

    net = hk.without_apply_rng(hk.transform(snn))
    # large constant input to force spikes
    x = jnp.ones((timesteps, batch, features)) * 2.0
    params = net.init(jax.random.PRNGKey(42), x)
    out = net.apply(params, x)
    assert jnp.all((out == 0.0) | (out == 1.0))


# ---------------------------------------------------------------------------
# Loss / utility functions (fn)
# ---------------------------------------------------------------------------


def test_integral_accuracy_shape():
    acc_fn = fn.integral_accuracy()
    # fake traces: [batch=4, time=10, classes=3]
    traces = jnp.ones((4, 10, 3))
    targets = jnp.array([0, 1, 2, 0])
    acc, preds = acc_fn(traces, targets)
    assert preds.shape == (4,)
    assert 0.0 <= float(acc) <= 1.0


def test_integral_crossentropy_scalar():
    loss_fn = fn.integral_crossentropy()
    traces = jnp.ones((4, 10, 3))
    targets = jnp.array([0, 1, 2, 0])
    loss = loss_fn(traces, targets)
    assert loss.shape == ()


def test_mse_spikerate_scalar():
    loss_fn = fn.mse_spikerate()
    traces = jnp.ones((4, 10, 3))
    targets = jnp.array([0, 1, 2, 0])
    loss = loss_fn(traces, targets)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_version_string():
    assert isinstance(spyx.__version__, str)
    assert len(spyx.__version__) > 0


def test_public_modules_exposed():
    for mod in ("nn", "axn", "fn", "data", "nir", "experimental", "loaders"):
        assert hasattr(spyx, mod), f"spyx.{mod} not exposed"
