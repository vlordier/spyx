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


def test_kwta_saliency_gate_shapes():
    def forward(x):
        gate = fm.KWTASaliencyGate(k=2)
        y, g = gate(x)
        return y, g

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((3, 8, 8, 6), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(12), x)
    y, g = transformed.apply(params, x)

    assert y.shape == x.shape
    assert g.shape == (3, 6)


def test_foveated_dual_path_snn_forward_shape():
    cfg = fm.FoveatedDualPathConfig(
        input_hw=(8, 8),
        input_channels=2,
        fovea_hw=(4, 4),
        channels_fovea=4,
        channels_periphery=3,
        output_dim=5,
    )

    def forward(x):
        model = fm.FoveatedDualPathSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(13), x)
    logits, spike_rate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)


def test_logpolar_foveated_conv_snn_forward_shape_and_aux():
    cfg = fm.LogPolarFoveatedConvConfig(
        input_hw=(8, 8),
        input_channels=2,
        radial_bins=5,
        angular_bins=12,
        channels1=4,
        channels2=6,
        output_dim=5,
        min_radius=0.5,
    )

    def forward(x):
        model = fm.LogPolarFoveatedConvSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["logpolar_seq"], aux["radial_energy"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.arange(4 * 2 * 8 * 8 * 2, dtype=jnp.float32).reshape((4, 2, 8, 8, 2))
    params = transformed.init(jax.random.PRNGKey(28), x)
    logits, spike_rate, logpolar_seq, radial_energy = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)
    assert logpolar_seq.shape == (4, 2, cfg.radial_bins, cfg.angular_bins, cfg.input_channels)
    assert radial_energy.shape == (cfg.radial_bins,)
    assert jnp.all(jnp.isfinite(logpolar_seq))


def test_imu_conditioned_visual_snn_forward_shape():
    vcfg = fm.ConvConfig(input_hw=(8, 8), input_channels=2, channels1=4, channels2=5, output_dim=6)
    cfg = fm.IMUConditionedConfig(vision_cfg=vcfg, imu_dim=3, imu_hidden=8, gating="hard")

    def forward(x, imu):
        model = fm.IMUConditionedVisualSNN(cfg)
        logits, aux = model(x, imu)
        return logits, aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    imu = jnp.ones((4, 2, 3), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(14), x, imu)
    logits, spike_rate = transformed.apply(params, x, imu)

    assert logits.shape == (2, vcfg.output_dim)
    assert spike_rate.shape == (2,)


def test_visual_imu_recurrent_fusion_block_forward_shape():
    vcfg = fm.ConvConfig(input_hw=(8, 8), input_channels=2, channels1=4, channels2=5, output_dim=6)
    cfg = fm.VisualIMURecurrentConfig(vision_cfg=vcfg, imu_dim=3, traj_dim=4, hidden_dim=7, output_dim=5)

    def forward(x, imu, traj):
        model = fm.VisualIMURecurrentFusionBlock(cfg)
        logits, _ = model(x, imu, traj)
        return logits

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    imu = jnp.ones((4, 2, 3), dtype=jnp.float32)
    traj = jnp.ones((4, 2, 4), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(15), x, imu, traj)
    logits = transformed.apply(params, x, imu, traj)

    assert logits.shape == (2, cfg.output_dim)


def test_flow_and_stereo_branches_forward_shape():
    flow_cfg = fm.OpticalFlowConfig(input_hw=(8, 8), input_channels=2, channels=4, output_dim=3)
    stereo_cfg = fm.StereoCoincidenceConfig(input_hw=(8, 8), input_channels=2, channels=4, output_dim=3)

    def flow_forward(x):
        model = fm.SpikingOpticalFlowBranch(flow_cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"]

    def stereo_forward(l, r):
        model = fm.StereoCoincidenceSNN(stereo_cfg)
        logits, aux = model(l, r)
        return logits, aux["spike_rate"]

    x = jnp.ones((5, 2, 8, 8, 2), dtype=jnp.float32)
    tf = hk.without_apply_rng(hk.transform(flow_forward))
    pf = tf.init(jax.random.PRNGKey(16), x)
    flow_logits, flow_sr = tf.apply(pf, x)

    ts = hk.without_apply_rng(hk.transform(stereo_forward))
    ps = ts.init(jax.random.PRNGKey(17), x, x)
    stereo_logits, stereo_sr = ts.apply(ps, x, x)

    assert flow_logits.shape == (2, flow_cfg.output_dim)
    assert flow_sr.shape == (1,)
    assert stereo_logits.shape == (2, stereo_cfg.output_dim)
    assert stereo_sr.shape == (1,)


def test_stereo_disparity_correlation_snn_forward_shape_and_aux():
    cfg = fm.StereoDisparityConfig(input_hw=(8, 8), input_channels=2, channels=5, output_dim=4, max_disparity=3)

    def forward(left, right):
        model = fm.StereoDisparityCorrelationSNN(cfg)
        logits, aux = model(left, right)
        return logits, aux["spike_rate"], aux["disparity_scores"], aux["left_right_consistency"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    left = jnp.ones((5, 2, 8, 8, 2), dtype=jnp.float32)
    right = jnp.ones((5, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(29), left, right)
    logits, spike_rate, disparity_scores, consistency = transformed.apply(params, left, right)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (1,)
    assert disparity_scores.shape == (cfg.max_disparity + 1,)
    assert jnp.asarray(consistency).shape == ()


def test_hybrid_classical_filter_snn_forward_shape_and_filter_energy():
    cfg = fm.HybridFilterConfig(input_hw=(8, 8), input_channels=2, channels1=4, channels2=6, output_dim=5)

    def forward(x):
        model = fm.HybridClassicalFilterSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["filter_energy"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(30), x)
    logits, spike_rate, filter_energy = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)
    assert filter_energy.shape == (3,)


def test_event_driven_pooling_snn_forward_shape_and_pool_features():
    cfg = fm.EventDrivenPoolingConfig(input_hw=(8, 8), input_channels=2, channels=4, output_dim=3, event_threshold=0.0)

    def forward(x):
        model = fm.EventDrivenPoolingSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["active_ratio"], aux["pooled_features"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(31), x)
    logits, spike_rate, active_ratio, pooled_features = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (1,)
    assert jnp.asarray(active_ratio).shape == ()
    assert pooled_features.shape == (4, 2, cfg.channels * len(cfg.pool_modes))


def test_integrated_wta_foveated_snn_forward_shape_and_gates():
    cfg = fm.WTAFoveatedStackConfig(
        input_hw=(8, 8),
        input_channels=2,
        fovea_hw=(4, 4),
        channels_fovea=4,
        channels_periphery=3,
        output_dim=5,
        router_patch=2,
        kwta_k=2,
    )

    def forward(x):
        model = fm.IntegratedWTAFoveatedSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["region_gate"], aux["channel_gate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(35), x)
    logits, spike_rate, region_gate, channel_gate = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (2,)
    assert region_gate.shape == (16,)
    assert channel_gate.shape == (cfg.channels_fovea,)


def test_hard_gated_moe_snn_forward_shape_and_usage():
    cfg = fm.HardGatedMoEConfig(input_dim=10, hidden_dim=8, output_dim=4, num_experts=3, top_k=1)

    def forward(x):
        model = fm.HardGatedMixtureOfExpertsSNN(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["expert_usage"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((5, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(36), x)
    logits, spike_rate, expert_usage = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (1,)
    assert expert_usage.shape == (cfg.num_experts,)


def test_spiking_collision_navigation_multihead_forward_shape():
    cfg = fm.SpikingMultiHeadConfig(
        input_dim=9,
        hidden_dim=8,
        collision_hidden=6,
        navigation_hidden=7,
        collision_dim=2,
        navigation_dim=3,
    )

    def forward(x):
        model = fm.SpikingCollisionNavigationMultiHead(cfg)
        outputs, aux = model(x)
        return outputs["collision"], outputs["navigation"], aux["spike_rate"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((5, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(37), x)
    collision, navigation, spike_rate = transformed.apply(params, x)

    assert collision.shape == (2, cfg.collision_dim)
    assert navigation.shape == (2, cfg.navigation_dim)
    assert spike_rate.shape == (3,)


def test_tiny_spiking_autoencoder_forward_shape_and_aux():
    cfg = fm.TinySpikingAutoencoderConfig(input_dim=10, hidden_dim=8, latent_dim=4)

    def forward(x):
        model = fm.TinySpikingAutoencoder(cfg)
        recon, aux = model(x)
        return recon, aux["spike_rate"], aux["latent_seq"], aux["reconstruction_error"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((5, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(32), x)
    recon, spike_rate, latent_seq, recon_error = transformed.apply(params, x)

    assert recon.shape == (2, cfg.input_dim)
    assert spike_rate.shape == (3,)
    assert latent_seq.shape == (5, 2, cfg.latent_dim)
    assert jnp.asarray(recon_error).shape == ()


def test_population_coded_lif_mlp_forward_shape_and_activity():
    cfg = fm.PopulationCodingConfig(input_dim=6, population_size=5, hidden_dim=8, output_dim=3)

    def forward(x):
        model = fm.PopulationCodedLIFMLP(cfg)
        logits, aux = model(x)
        return logits, aux["spike_rate"], aux["population_activity"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.ones((5, 2, cfg.input_dim), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(33), x)
    logits, spike_rate, population_activity = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert spike_rate.shape == (1,)
    assert population_activity.shape == (1,)


def test_latency_coded_spiking_head_forward_shape_and_timing():
    cfg = fm.LatencyCodingConfig(input_dim=7, hidden_dim=9, output_dim=4)

    def forward(x):
        model = fm.LatencyCodedSpikingHead(cfg)
        logits, aux = model(x)
        return logits, aux["first_spike_time"], aux["latency_code_density"]

    transformed = hk.without_apply_rng(hk.transform(forward))
    x = jnp.arange(5 * 2 * cfg.input_dim, dtype=jnp.float32).reshape((5, 2, cfg.input_dim))
    params = transformed.init(jax.random.PRNGKey(34), x)
    logits, first_spike_time, latency_density = transformed.apply(params, x)

    assert logits.shape == (2, cfg.output_dim)
    assert first_spike_time.shape == (2,)
    assert jnp.asarray(latency_density).shape == ()


def test_router_and_gaze_head_shapes():
    gaze_cfg = fm.GazeControlConfig(input_dim=6, imu_dim=3, traj_dim=4, num_regions=9)

    def router_forward(x):
        router = fm.RegionActivationRouter(top_k=3, patch=2)
        return router(x)

    def gaze_forward(p, imu, traj):
        head = fm.GazeControlPolicyHead(gaze_cfg, top_k=2)
        scores, aux = head(p, imu, traj)
        return scores, aux["region_gate"]

    x = jnp.ones((2, 8, 8, 4), dtype=jnp.float32)
    tr = hk.without_apply_rng(hk.transform(router_forward))
    pr = tr.init(jax.random.PRNGKey(18), x)
    scores, gate = tr.apply(pr, x)

    p = jnp.ones((4, 2, 6), dtype=jnp.float32)
    imu = jnp.ones((4, 2, 3), dtype=jnp.float32)
    traj = jnp.ones((4, 2, 4), dtype=jnp.float32)
    tg = hk.without_apply_rng(hk.transform(gaze_forward))
    pg = tg.init(jax.random.PRNGKey(19), p, imu, traj)
    gs, gg = tg.apply(pg, p, imu, traj)

    assert scores.shape == (2, 16)
    assert gate.shape == (2, 16)
    assert gs.shape == (2, gaze_cfg.num_regions)
    assert gg.shape == (2, gaze_cfg.num_regions)


def test_predictive_multihead_sparse_and_early_exit_shapes():
    pred_cfg = fm.PredictiveCodingConfig(input_dim=10, latent_dim=8, output_dim=4)
    mh_cfg = fm.MultiHeadConfig(input_dim=10, hidden_dim=8, collision_dim=2, navigation_dim=3)
    sparse_cfg = fm.StructuredSparseConvConfig(
        input_hw=(8, 8),
        input_channels=2,
        channels1=4,
        channels2=5,
        output_dim=3,
        sparsity=0.5,
    )
    ee_cfg = fm.EarlyExitConfig(input_dim=10, hidden_dim=8, output_dim=4)

    def pred_forward(x):
        m = fm.PredictiveCodingSNNBlock(pred_cfg)
        y, aux = m(x)
        return y, aux["error_norm"]

    def mh_forward(x):
        m = fm.CollisionNavigationMultiHead(mh_cfg)
        out = m(x)
        return out["collision"], out["navigation"]

    def sparse_forward(x):
        m = fm.StructuredSparseSpikingCNN(sparse_cfg)
        y, aux = m(x)
        return y, aux["spike_rate"]

    def ee_forward(x):
        m = fm.EarlyExitAnytimeSNN(ee_cfg)
        y, aux = m(x)
        return y, aux["early_exit_rate"]

    x_seq = jnp.ones((5, 2, 10), dtype=jnp.float32)
    tp = hk.without_apply_rng(hk.transform(pred_forward))
    pp = tp.init(jax.random.PRNGKey(20), x_seq)
    pred_logits, err_norm = tp.apply(pp, x_seq)

    x_lat = jnp.ones((2, 10), dtype=jnp.float32)
    tm = hk.without_apply_rng(hk.transform(mh_forward))
    pm = tm.init(jax.random.PRNGKey(21), x_lat)
    coll, nav = tm.apply(pm, x_lat)

    x_img = jnp.ones((4, 2, 8, 8, 2), dtype=jnp.float32)
    ts = hk.without_apply_rng(hk.transform(sparse_forward))
    ps = ts.init(jax.random.PRNGKey(22), x_img)
    sparse_logits, sparse_sr = ts.apply(ps, x_img)

    te = hk.without_apply_rng(hk.transform(ee_forward))
    pe = te.init(jax.random.PRNGKey(23), x_seq)
    ee_logits, ee_rate = te.apply(pe, x_seq)

    assert pred_logits.shape == (2, pred_cfg.output_dim)
    assert jnp.asarray(err_norm).shape == ()
    assert coll.shape == (2, mh_cfg.collision_dim)
    assert nav.shape == (2, mh_cfg.navigation_dim)
    assert sparse_logits.shape == (2, sparse_cfg.output_dim)
    assert sparse_sr.shape == (2,)
    assert ee_logits.shape == (2, ee_cfg.output_dim)
    assert jnp.asarray(ee_rate).shape == ()


def test_time_surface_motioncomp_kalman_and_trajectory_shapes():
    vcfg = fm.ConvConfig(input_hw=(8, 8), input_channels=2, channels1=4, channels2=5, output_dim=6)
    mcfg = fm.MotionCompConfig(vision_cfg=vcfg, imu_scale=1.0)
    kcfg = fm.KalmanFusionConfig(latent_dim=7, output_dim=4)
    tcfg = fm.TrajectoryConditionedConfig(vision_dim=6, imu_dim=3, traj_dim=4, hidden_dim=8, output_dim=5)

    def ts_forward(x):
        enc = fm.TimeSurfaceEncoder(tau=2.0)
        return enc(x)

    def mc_forward(x, imu):
        model = fm.MotionCompensatedInputFrontEnd(mcfg)
        logits, aux = model(x, imu)
        return logits, aux["spike_rate"]

    def kf_forward(v, imu):
        model = fm.KalmanStyleSpikingFusionSurrogate(kcfg)
        logits, aux = model(v, imu)
        return logits, aux["innovation_norm"]

    def tc_forward(v, imu, traj):
        model = fm.TrajectoryConditionedSpikingEncoder(tcfg, hard_gate=True)
        logits, aux = model(v, imu, traj)
        return logits, aux["gate_mean"]

    x = jnp.ones((5, 2, 8, 8, 2), dtype=jnp.float32)
    imu = jnp.ones((5, 2, 3), dtype=jnp.float32)
    vis = jnp.ones((5, 2, 7), dtype=jnp.float32)
    traj = jnp.ones((5, 2, 4), dtype=jnp.float32)

    tts = hk.without_apply_rng(hk.transform(ts_forward))
    pts = tts.init(jax.random.PRNGKey(24), x)
    x_ts = tts.apply(pts, x)

    tmc = hk.without_apply_rng(hk.transform(mc_forward))
    pmc = tmc.init(jax.random.PRNGKey(25), x, imu)
    mc_logits, mc_sr = tmc.apply(pmc, x, imu)

    tkf = hk.without_apply_rng(hk.transform(kf_forward))
    pkf = tkf.init(jax.random.PRNGKey(26), vis, vis)
    kf_logits, kf_innov = tkf.apply(pkf, vis, vis)

    ttc = hk.without_apply_rng(hk.transform(tc_forward))
    ptc = ttc.init(jax.random.PRNGKey(27), vis[:, :, :6], imu, traj)
    tc_logits, tc_gate = ttc.apply(ptc, vis[:, :, :6], imu, traj)

    assert x_ts.shape == x.shape
    assert mc_logits.shape == (2, vcfg.output_dim)
    assert mc_sr.shape == (2,)
    assert kf_logits.shape == (2, kcfg.output_dim)
    assert jnp.asarray(kf_innov).shape == ()
    assert tc_logits.shape == (2, tcfg.output_dim)
    assert jnp.asarray(tc_gate).shape == ()
