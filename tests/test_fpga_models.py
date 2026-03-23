import haiku as hk
import jax
import jax.numpy as jnp

from spyx import fpga_models as fm


def test_lif_mlp_forward_shape():
    cfg = fm.MLPConfig(input_dim=16, hidden1=12, hidden2=8, output_dim=4)

    def forward(x):
        model = fm.LIFMLP(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((6, 3, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(0), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (3, cfg.output_dim)
    assert spike_rate.shape == (2,)


def test_ternary_lif_mlp_forward_shape():
    cfg = fm.MLPConfig(input_dim=10, hidden1=10, hidden2=6, output_dim=3)

    def forward(x):
        model = fm.TernaryLIFMLP(cfg, threshold=0.1)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((5, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(1), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)


def test_conv_lif_snn_forward_shape():
    cfg = fm.ConvConfig(input_hw=(8, 8), input_channels=2, channels1=4, channels2=6, output_dim=5)

    def forward(x):
        model = fm.ConvLIFSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(2), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)


def test_ternary_conv_lif_snn_forward_shape():
    cfg = fm.ConvConfig(input_hw=(8, 8), input_channels=2, channels1=4, channels2=6, output_dim=5)

    def forward(x):
        model = fm.TernaryConvLIFSNN(cfg, threshold=0.1)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(3), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)


def test_sparse_event_conv_lif_snn_forward_shape_and_activity():
    cfg = fm.SparseConvConfig(
        input_hw=(8, 8),
        input_channels=2,
        channels1=4,
        channels2=6,
        output_dim=5,
        event_threshold=0.5,
    )

    def forward(x):
        model = fm.SparseEventConvLIFSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["active_ratio"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(4), x)
    logits, spike_rate, active_ratio = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)
    assert jnp.asarray(active_ratio).shape == ()


def test_depthwise_separable_conv_lif_snn_forward_shape():
    cfg = fm.DepthwiseSepConvConfig(
        input_hw=(8, 8),
        input_channels=2,
        depth_multiplier1=2,
        pointwise1=8,
        depth_multiplier2=1,
        pointwise2=6,
        output_dim=5,
    )

    def forward(x):
        model = fm.DepthwiseSeparableConvLIFSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(5), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)


def test_benchmark_forward_returns_summary():
    cfg = fm.MLPConfig(input_dim=12, hidden1=8, hidden2=6, output_dim=3)

    def forward(x):
        model = fm.LIFMLP(cfg)
        return model(x)

    x = jnp.ones((4, 2, cfg.input_dim), dtype=jnp.float32)
    summary = fm.benchmark_forward(forward, x, seed=7)

    assert summary["params"] > 0
    assert summary["logits_shape"] == (2, cfg.output_dim)
    assert "spike_rate" in summary


def test_residual_shallow_spiking_cnn_forward_shape():
    cfg = fm.ResidualConvConfig(
        input_hw=(8, 8),
        input_channels=2,
        stem_channels=6,
        block_channels=6,
        output_dim=5,
    )

    def forward(x):
        model = fm.ResidualShallowSpikingCNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(8), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (3,)


def test_multi_timescale_lif_block_forward_shape():
    cfg = fm.MultiTimescaleConfig(input_dim=12, hidden_dim=10, output_dim=4)

    def forward(x):
        model = fm.MultiTimescaleLIFBlock(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((5, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(9), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (3,)


def test_tiny_recurrent_spiking_block_forward_shape():
    cfg = fm.RecurrentBlockConfig(input_dim=9, hidden_dim=7, output_dim=3)

    def forward(x):
        model = fm.TinyRecurrentSpikingBlock(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((6, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(10), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (1,)


def test_hybrid_snn_encoder_head_forward_shape():
    cfg = fm.HybridEncoderConfig(
        input_hw=(8, 8),
        input_channels=2,
        channels1=4,
        channels2=6,
        head_hidden=8,
        output_dim=5,
    )

    def forward(x):
        model = fm.HybridSNNEncoderHead(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(11), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)
