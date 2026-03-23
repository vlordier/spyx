# Models and Architectures Roadmap

## Purpose and Scope
This file consolidates architecture recommendations for a Spyx-based pipeline targeting fixed-point and FPGA deployment.

It merges all prior notes into one consistent reference, preserving the original recommendations while removing repeated chat-style restatements.

### Consolidation Status
- Consolidated source: [src/spyx/SPNN.md](src/spyx/SPNN.md).
- Consolidated content includes: core top-10 ranking, high/medium-value extension lists, foveated/spherical prioritization, ternary-robust guidance, and experiment-matrix planning guidance.
- This file is the canonical architecture roadmap to avoid divergence between duplicate documents.

## Implementation Checklist (Spyx)
Status is tracked against reference implementations in [src/spyx/fpga_models.py](src/spyx/fpga_models.py).

### Core top-10 families
- [x] Plain feedforward LIF MLP -> `LIFMLP`
- [x] Small convolutional LIF SNN -> `ConvLIFSNN`
- [x] Ternary-weight LIF MLP -> `TernaryLIFMLP`
- [x] Ternary-weight conv LIF SNN -> `TernaryConvLIFSNN`
- [x] Sparse event-driven conv SNN -> `SparseEventConvLIFSNN`
- [x] Depthwise-separable conv SNN -> `DepthwiseSeparableConvLIFSNN`
- [x] Shallow residual spiking CNN -> `ResidualShallowSpikingCNN`
- [x] Multi-timescale LIF block -> `MultiTimescaleLIFBlock`
- [x] Tiny recurrent spiking block -> `TinyRecurrentSpikingBlock`
- [x] Hybrid SNN encoder + non-spiking head -> `HybridSNNEncoderHead`

### Foveated, fusion, routing, and motion modules
- [x] k-WTA saliency gate -> `KWTASaliencyGate`
- [x] Time-surface encoding -> `TimeSurfaceEncoder`
- [x] Strict log-polar convolutional SNN -> `LogPolarFoveatedConvSNN`
- [x] Foveated dual-path SNN -> `FoveatedDualPathSNN`
- [x] IMU-conditioned visual SNN -> `IMUConditionedVisualSNN`
- [x] Visual-IMU recurrent fusion -> `VisualIMURecurrentFusionBlock`
- [x] Kalman-style fusion surrogate -> `KalmanStyleSpikingFusionSurrogate`
- [x] Spiking optical-flow branch -> `SpikingOpticalFlowBranch`
- [x] Stereo coincidence/disparity proxy -> `StereoCoincidenceSNN`
- [x] Stereo disparity / correlation family -> `StereoDisparityCorrelationSNN`
- [x] Motion-compensated input front-end -> `MotionCompensatedInputFrontEnd`
- [x] Gaze-control policy head -> `GazeControlPolicyHead`
- [x] Region-activation router -> `RegionActivationRouter`
- [x] Integrated WTA-driven foveation stack -> `IntegratedWTAFoveatedSNN`
- [x] Event-driven sparse foveated SNN -> `EventDrivenSparseFoveatedSNN`
- [x] Trajectory-conditioned encoder -> `TrajectoryConditionedSpikingEncoder`
- [x] Predictive-coding block -> `PredictiveCodingSNNBlock`
- [x] Collision + navigation multi-head -> `CollisionNavigationMultiHead`
- [x] Fully spiking collision/navigation multi-head family -> `SpikingCollisionNavigationMultiHead`
- [x] Hybrid SNN + classical filters pipeline -> `HybridClassicalFilterSNN`
- [x] Tiny spiking autoencoder -> `TinySpikingAutoencoder`
- [x] Population coding variant -> `PopulationCodedLIFMLP`
- [x] Time-to-first-spike / latency-coded heads -> `LatencyCodedSpikingHead`
- [x] Hard-gated mixture-of-experts family -> `HardGatedMixtureOfExpertsSNN`
- [x] Spherical-geometry spike-routing graph -> `SphericalRoutingGraphSNN`
- [x] Spherical harmonic / frequency-domain SNN (proxy) -> `SphericalFrequencyDomainSNN`
- [x] Small liquid state machine -> `SmallLiquidStateMachineSNN`
- [x] Delay-based SNN -> `DelayBasedSpikingSNN`
- [x] Structured-sparse spiking CNN -> `StructuredSparseSpikingCNN`
- [x] Event-driven pooling variants -> `EventDrivenPoolingSNN`
- [x] Early-exit/anytime inference head -> `EarlyExitAnytimeSNN`

### Notes
- These are reference blocks intended for iterative experiments and hardware co-design sweeps, not final production/training recipes.
- Some conceptual families in this document (for example strict log-polar transforms or graph-spherical connectivity) are represented by practical approximations in the current implementation set.

## Implementation Coverage
This section distinguishes between items that are implemented literally, items that are represented by practical approximations or compositions, and items that remain conceptual only.

### Implemented Exactly
These exist as concrete reference modules in `src/spyx/fpga_models.py` and are covered by `tests/test_fpga_models.py`.

| Roadmap item | Spyx implementation |
| --- | --- |
| Plain feedforward LIF MLP | `LIFMLP` |
| Small convolutional LIF SNN | `ConvLIFSNN` |
| Ternary-weight LIF MLP | `TernaryLIFMLP` |
| Ternary-weight conv LIF SNN | `TernaryConvLIFSNN` |
| Sparse event-driven conv SNN | `SparseEventConvLIFSNN` |
| Depthwise-separable conv SNN | `DepthwiseSeparableConvLIFSNN` |
| Shallow residual spiking CNN | `ResidualShallowSpikingCNN` |
| Multi-timescale LIF block | `MultiTimescaleLIFBlock` |
| Tiny recurrent spiking block | `TinyRecurrentSpikingBlock` |
| Hybrid SNN encoder + non-spiking head | `HybridSNNEncoderHead` |
| k-WTA saliency gate | `KWTASaliencyGate` |
| Time-surface encoding | `TimeSurfaceEncoder` |
| Strict log-polar convolutional SNN | `LogPolarFoveatedConvSNN` |
| IMU-conditioned visual SNN | `IMUConditionedVisualSNN` |
| Visual-IMU recurrent fusion | `VisualIMURecurrentFusionBlock` |
| Kalman-style fusion surrogate | `KalmanStyleSpikingFusionSurrogate` |
| Spiking optical-flow branch | `SpikingOpticalFlowBranch` |
| Stereo disparity / correlation family | `StereoDisparityCorrelationSNN` |
| Motion-compensated input front-end | `MotionCompensatedInputFrontEnd` |
| Region-activation router | `RegionActivationRouter` |
| Integrated WTA-driven foveation stack | `IntegratedWTAFoveatedSNN` |
| Trajectory-conditioned encoder | `TrajectoryConditionedSpikingEncoder` |
| Predictive-coding block | `PredictiveCodingSNNBlock` |
| Fully spiking collision/navigation multi-head family | `SpikingCollisionNavigationMultiHead` |
| Hybrid SNN + classical filters pipeline | `HybridClassicalFilterSNN` |
| Tiny spiking autoencoder | `TinySpikingAutoencoder` |
| Population coding variant | `PopulationCodedLIFMLP` |
| Time-to-first-spike / latency-coded heads | `LatencyCodedSpikingHead` |
| Hard-gated mixture-of-experts family | `HardGatedMixtureOfExpertsSNN` |
| Spherical-geometry spike-routing graph | `SphericalRoutingGraphSNN` |
| Spherical harmonic / frequency-domain SNN (proxy) | `SphericalFrequencyDomainSNN` |
| Small liquid state machine | `SmallLiquidStateMachineSNN` |
| Delay-based SNN | `DelayBasedSpikingSNN` |
| Structured-sparse spiking CNN | `StructuredSparseSpikingCNN` |
| Event-driven pooling variants | `EventDrivenPoolingSNN` |
| Early-exit / anytime head | `EarlyExitAnytimeSNN` |

### Implemented as Approximation or Composition
These are present in practical form, but not as literal one-to-one realizations of the roadmap phrase.

| Roadmap concept | Current state |
| --- | --- |
| Foveated dual-path / multi-scale SNN | Implemented as `FoveatedDualPathSNN` |
| Gaze-control policy head | Implemented as `GazeControlPolicyHead`, but not a full gaze-control SNN family |
| Time-surface + foveated SNN | Achievable by composing `TimeSurfaceEncoder` with `FoveatedDualPathSNN` |
| Motion-compensated foveated SNN | Achievable by composing `MotionCompensatedInputFrontEnd` with foveated modules |
| Ternary saliency-router stack | Represented by ternary, sparse, gating, and routing pieces rather than one named stack |

### Still Conceptual Only
These are described in the roadmap but do not yet have dedicated implementations in the current Spyx reference set.

| Roadmap concept | Status |
| --- | --- |
| Spike-frequency coding family | Not implemented as dedicated variants |
| Stereo foveated correlation family with disparity bins / left-right consistency | Partially implemented via `StereoDisparityCorrelationSNN`; not yet foveated |
| Frequency-domain or graph-based spherical models | Not implemented |

## Gap Priority
This table focuses on the remaining gaps and ranks them by implementation value rather than novelty.

### Implement Next
No remaining low-complexity gaps are queued here after the latest implementation pass.

### Implement Later
| Item | Current state in Spyx | Effort | Why later |
| --- | --- | --- | --- |
No medium-priority practical gaps remain; only deferred research-heavy families are left.

### Defer
| Item | Current state in Spyx | Effort | Why defer |
| --- | --- | --- | --- |
| Strict graph-based spherical model family | Not implemented | High | Same issue as above |
| Bio-detailed neurons, transformers, STDP-heavy models | Not implemented | High | Explicitly deprioritized by this roadmap |

### Recommended Order to Close Gaps
1. Spike-frequency coding family.
2. Strict graph-based spherical model family.
3. Bio-detailed neurons, transformers, and STDP-heavy models only for long-horizon research.

## System Context
Primary context:
- Stereo and pseudo-event vision for drone navigation.
- Path to deployment: Spyx training -> fixed-point design -> RTL/FPGA implementation.
- Target outcomes: collision avoidance features and navigation/world-model features.

Extended context (later iterations):
- Spherical input (equirectangular or fisheye).
- Foveated processing (log-polar or multi-resolution fovea/periphery).
- Event-driven sparsity as a first-order design goal.
- Additional branches/modules: stereo correlation, WTA/k-WTA gating, gaze control, IMU, and trajectory latent conditioning (Skydreamer-like).

## Core Design Philosophy
The main objective is not novelty-first model search. It is hardware-transferable SNN design that survives quantization and routing constraints.

Prioritized properties:
- Simple neuron dynamics (LIF-style).
- Structured locality over global dense connectivity.
- Sparse activity and sparse compute scheduling.
- Low-bit and ternary-friendly arithmetic.
- Controlled temporal memory.
- Deterministic and FPGA-mappable dataflow.

## Co-Design Axes to Sweep
These are the key hardware-algorithm axes:
1. Sparsity:
- Spike rate.
- Percent active neurons.
2. Precision:
- Ternary vs int8 vs fixed-point.
3. Connectivity:
- Dense vs local vs structured sparse.
4. Temporal behavior:
- Memory length.
- Latency vs accuracy.
5. Routing complexity:
- Local vs global spike communication.

## Top 10 Hardware-Transferable Model Families
This is the baseline ranking for the original stereo/pseudo-event FPGA path.

### 1) Plain feedforward LIF MLP
Why:
- Reference baseline.
- Easiest model to ternarize.
- Easiest BRAM + accumulator mapping.
- Useful for low-dimensional fused features.

Use for:
- Fixed-point behavior baseline.
- Ternary experiments.
- Latency/accuracy tradeoffs.

### 2) Small convolutional LIF SNN
Recommended shape:
- Input: pseudo-event frame or 2-channel ON/OFF.
- Conv 3x3 -> Conv 3x3 -> optional pooling -> small dense head.

Why:
- First serious vision backbone.
- Structured local connectivity is hardware-friendly.

### 3) Sparse event-driven convolutional SNN
Why:
- Matches sparse event/pseudo-event inputs.
- Reduces compute in static regions.
- High upside for event hardware efficiency.

Note:
- Implement after dense conv-LIF baseline (scheduling/routing complexity is higher).

### 4) Depthwise-separable conv SNN
Why:
- Good memory/bandwidth compromise.
- Maintains local spatial structure with lower cost.

Compare against standard conv:
- Accuracy.
- Spike sparsity.
- BRAM proxy.
- Routing proxy.

### 5) Shallow residual spiking CNN
Scope:
- Keep shallow (for example two residual blocks).
- Fixed skip paths.
- Controlled channel counts.

Why:
- Better trainability without deep complexity.

### 6) Ternary-weight LIF MLP
Why:
- Directly relevant to FPGA memory and arithmetic simplification.

Study:
- {-1, 0, +1} constraints.
- Structural sparsity effects.
- Sign-select/accumulator data paths.

### 7) Ternary-weight convolutional LIF SNN
Why:
- Surfaces real deployment issues that MLP ternarization misses.
- Kernel packing, line buffers, fan-out, activation sparsity.

### 8) Multi-timescale LIF block
Use:
- Fast/medium/slow leak variants.

Why:
- Adds temporal diversity without heavy biological complexity.

### 9) Tiny recurrent spiking block
Constraint:
- Very small recurrent core.
- Fixed recurrence pattern.
- Avoid dense all-to-all recurrence.

Why:
- Adds short memory for ego-motion and persistence cues.

### 10) Hybrid SNN encoder + non-spiking head
Why:
- Practical compromise for control/regression outputs.
- Preserves sparse front-end while keeping output head simple.

Example outputs:
- Collision score.
- Steering bias.
- Waypoint logits.
- Threat probability.

## Core Implementation Order
Recommended order for implementation in Spyx:
1. LIF MLP.
2. Small conv LIF SNN.
3. Ternary LIF MLP.
4. Ternary conv LIF SNN.
5. Sparse event-driven conv SNN.
6. Depthwise-separable conv SNN.
7. Shallow residual spiking CNN.
8. Multi-timescale LIF block.
9. Hybrid SNN encoder + dense head.
10. Tiny recurrent spiking block.

### Minimum Practical Subset
If minimizing scope, start with:
- Model A: LIF MLP.
- Model B: Small conv LIF SNN.
- Model C: Ternary conv LIF SNN.
- Model D: Sparse event-driven conv SNN.
- Model E: Hybrid SNN encoder + dense head.

## Second-Tier Extensions (After the Core 10)

### High-value extensions
1. Spiking optical-flow SNN.
- Idea: local motion extraction from event/pseudo-event stream, then downstream SNN.
- Why: motion-first for drone navigation.
- FPGA fit: local neighborhoods, streaming-friendly.

2. Correlation/stereo matching SNN.
- Idea: left/right spike correlation for disparity.
- Variants: spike coincidence, time-surface correlation.
- Why: depth from spikes without dense stereo CNN.
- FPGA fit: local window parallelism.

3. Winner-take-all (WTA) / lateral inhibition layers.
- Idea: local competition, strongest spikes survive.
- Why: sparsity, routing pressure reduction, robustness.
- FPGA fit: local max/threshold logic.

4. Time-surface encoding SNN.
- Idea: per-pixel last timestamp -> decaying feature.
- Why: temporal context without heavy recurrence.
- FPGA fit: register + simple decay.

5. Population coding.
- Idea: multi-neuron groups per feature.
- Why: robustness to noise and quantization.

6. Spike-frequency vs time-to-first-spike coding.
- Why: latency coding can reduce timesteps and enable early exit.

7. Early-exit / anytime SNN.
- Idea: confidence-triggered stopping.
- Why: latency and power reduction.

8. Structured sparsity.
- Forms: block, channel, kernel sparsity.
- Why: BRAM/routing friendliness vs random sparsity.

9. Spike gating / attention-lite SNN.
- Idea: simple binary gating between streams.
- Why: dynamic routing at low hardware cost.

### Medium-value extensions
10. Small liquid state machine.
- Use: temporal smoothing/short memory.
- Keep small and sparse.

11. Event-driven pooling variants.
- Spike pooling, temporal pooling, event accumulation pooling.

12. Tiny spiking autoencoder.
- Use: latent compression between modules (especially FPGA/MCU splits).

13. Multi-head spiking outputs.
- Example heads: obstacle, motion, target tracking.

14. Hybrid SNN + classical filters.
- Example: diff -> Sobel/gradient -> SNN.

15. Delay-based SNN (later).
- Can encode motion direction.
- Defer due to FPGA buffering/routing complexity.

## Deprioritized Early (General)
- Spiking transformers.
- Biologically detailed neurons (for example HH/AdEx).
- Very deep ANN-to-SNN conversions.
- Large dense recurrent SNNs.
- STDP-heavy online learning as a core deployment path.

## Foveated/Spherical-Specific Re-Ranking
When the pipeline becomes spherical -> foveated -> event-driven -> FPGA, priorities shift toward geometry alignment and routing simplicity.

### Top architectures for this regime
1. Log-polar/foveated convolutional SNN.
- Primary backbone candidate.

2. Multi-scale dual-path SNN (fovea + periphery).
- Practical alternative to strict log-polar mapping.

3. Event-driven foveated SNN.
- Sparse region updates.

4. Time-surface + foveated SNN.
- Strong for motion-heavy scenes.

5. Stereo foveated correlation SNN.
- Center precision + peripheral coarse depth.

6. WTA/saliency-driven foveation.
- Adaptive compute allocation.

7. Spherical-geometry spike-routing graph (advanced).
- Non-grid connectivity approximation.

8. Spherical harmonic/frequency-domain SNN (experimental).

9. Attention-lite gaze-control SNN.

10. Motion-compensated foveated SNN.
- Often near-mandatory to suppress ego-rotation noise.

### Tiered priorities for this regime
Tier 1 (must implement):
- Foveated/log-polar conv SNN.
- Multi-scale dual-path SNN.
- Time-surface encoding.
- Stereo correlation SNN.

Tier 2 (high gain):
- Event-driven sparse foveated SNN.
- WTA/saliency gating.
- Motion compensation.

Tier 3 (advanced):
- Dynamic gaze control.
- Graph-based spherical SNN.
- Frequency-domain SNN.

### Minimal strong stack in this regime
- Stereo spherical cameras.
- Foveation (log-polar or dual-path).
- Time-surface encoding.
- Sparse ternary conv SNN.
- Small dense head.

Optional upgrades:
- Stereo correlation branch.
- WTA gating.
- Gaze control.

## Specialized Modules for the Augmented Stack
These modules reflect the later expanded system with IMU and trajectory conditioning.

### 1) Sensor-fusion blocks
1. IMU-conditioned visual SNN.
- Variants: late fusion and gated fusion.

2. Visual-IMU recurrent fusion block.
- Tiny recurrent state integrator (short horizon).

3. Kalman-style spiking fusion surrogate.
- Prediction + correction decomposition.

### 2) Motion and geometry modules
4. Spiking optical-flow branch.
- Pure spiking front-end + tiny dense head variant.
- Hybrid flow branch variant for benchmarking.

5. Stereo disparity/coincidence family.
- Coincidence detector, disparity cost in foveated space, left-right consistency.

6. Time-surface encoder.

7. Motion-compensated input front-end.
- IMU-guided de-rotation/stabilization before event/foveated processing.

### 3) Selection and routing modules
8. k-WTA saliency gate.

9. Gaze-control policy head.
- Compare heuristic, tiny dense, and spiking controller variants.

10. Region-activation router.
- Tile scoring + binary/ternary masks + sparse execution proxy.

### 4) World-model and trajectory-conditioning modules
11. Trajectory-conditioned spiking encoder.
- Forms: concatenation and gain modulation.

12. Predictive-coding SNN block.
- Emphasize residual/prediction error.

13. Collision-risk head + navigation-value head.
- Shared backbone, separate heads.
- Compare fully spiking vs hybrid-head variants.

### 5) Hardware-oriented families
14. Structured-sparse spiking CNN.

15. Multi-timescale LIF family.

16. Early-exit/anytime SNN.

### Ranked shortlist for the augmented system
1. IMU-conditioned visual SNN.
2. Spiking optical-flow branch.
3. Trajectory-conditioned spiking encoder.
4. Visual-IMU short-memory recurrent fusion block.
5. k-WTA saliency gate.
6. Gaze-control policy head.
7. Region-activation router.
8. Kalman-style spiking fusion surrogate.
9. Predictive-coding SNN block.
10. Structured-sparse spiking CNN.
11. Motion-compensated input SNN.
12. Collision-risk head + navigation-value head.
13. Multi-timescale LIF family.
14. Early-exit/anytime SNN.
15. Expanded stereo disparity SNN family.

## Ternary-Robust Architecture Guidance

### Core rule
Ternary robustness is mainly about compositional stability.

Prefer blocks dominated by:
- Signed accumulation.
- Thresholding.
- Max/top-k competition.
- Simple decay.
- Masking/gating.
- Local correlation.

### Most ternary-robust patterns
1. Ternary LIF blocks with residual identity paths.
2. Multi-branch ternary networks with late fusion.
3. k-WTA/competition layers.
4. Correlation/coincidence blocks.
5. Hard-gated mixture-of-experts.
6. Predictive-coding/error-only blocks.
7. Multi-timescale ternary LIF families.
8. Structured-sparse ternary conv blocks.
9. Early-exit ternary heads.
10. Hybrid-bit systems around ternary spiking cores.

### Best module combinations for this stack
A) Ternary stereo correlation tower.
- Local correlation -> disparity candidates -> top-k selection -> late fusion.

B) Ternary IMU-gated visual backbone.
- IMU predicts masks/thresholds/gains.

C) Ternary trajectory-conditioned gating.
- Hard or low-bit gates preferred over smooth attention.

D) Ternary collision head + ternary/free-space head.
- Shared backbone, task-specific calibrated heads.

E) Ternary saliency router.
- Peripheral branch + hard tile selection + selective high-res path.

### Less ternary-robust patterns (defer)
- Soft attention everywhere.
- Deep dense recurrence.
- Bio-detailed neuron models.
- Massive global early feature fusion.
- Deep no-skip ternary stacks.

### Practical implementation rules
- Ternary weights first; do not force ternary membrane state first.
- Keep membrane/accumulator widths wider than weights.
- Prefer hard gates over soft gates.
- Prefer concat + small fusion MLP over repeated additive mixing.
- Keep recurrence local and shallow.
- Add skip paths when increasing depth.
- Track branch entropy, spike rate, saturation rate, and sign-flip sensitivity.

## Ternary-Robust Experiment Matrix
Interpretation:
- Fusion type: where/how branches combine.
- Ternary policy: what to ternarize first.
- State width: initial membrane/accumulator widths.
- Spike sparsity target: expected activity regime.
- Calibration sensitivity: fragility to thresholds/gains/scales.
- FPGA risk: implementation difficulty (memory/routing/control).

| ID  | Model family                                 | Role in system                                 | Fusion type                                      | Ternary policy                                                   | State width (start)               | Spike sparsity target        | Calibration sensitivity | FPGA risk   | Priority |
| --- | -------------------------------------------- | ---------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------- | --------------------------------- | ---------------------------- | ----------------------- | ----------- | -------- |
| M1  | Ternary Conv-LIF backbone                    | Main foveal visual encoder                     | None / single branch                             | Ternary weights first; activations binary spikes; membrane wider | 8-12b mem, 12-16b accum           | Medium                       | Medium                  | Low         | P0       |
| M2  | Residual Ternary Conv-LIF                    | Deeper visual encoder with safer composition   | Add via identity skip                            | Ternary main branch, skip path full-sign / integer               | 8-12b mem, 12-16b accum           | Medium                       | Low-Medium              | Low         | P0       |
| M3  | Structured-sparse Ternary Conv               | BRAM/routing-friendly visual backbone          | None or late concat                              | Ternary + block/channel sparsity                                 | 8-12b mem, 12-16b accum           | High                         | Medium                  | Low-Medium  | P0       |
| M4  | Stereo correlation / coincidence branch      | Depth / collision cues                         | Late fusion to shared latent                     | Ternary kernels or sign-correlation; avoid soft mixing           | 8-10b cost state, 12-16b accum    | High                         | Low                     | Medium      | P0       |
| M5  | Time-surface encoder                         | Temporal pre-encoding for events               | Pre-fusion input transform                       | Usually not ternary internally; ternary only after encoding      | 8-16b timestamps / decay          | High downstream              | Medium                  | Low         | P0       |
| M6  | k-WTA saliency gate                          | Region/channel selection                       | Hard top-k routing                               | No need for ternary internals; compare scores                    | 8-12b scores                      | Very High after gate         | Medium                  | Medium      | P0       |
| M7  | Hard gaze-control head                       | Select next fovea center / tile                | Late fusion of peripheral summary + IMU + traj   | Ternary MLP okay; hard argmax/top-k output                       | 8-12b hidden, 12-16b accum        | Very High in selected path   | High                    | Medium      | P0       |
| M8  | Region-activation router                     | Activate only chosen visual tiles              | Hard routing mask                                | Ternary scorer okay; binary mask output                          | 8-12b scores                      | Very High                    | High                    | Medium-High | P0       |
| M9  | IMU-conditioned visual gating                | Ego-motion compensation / modulation           | Multiplicative hard gate or threshold modulation | Keep visual core ternary; IMU branch can be low-bit int          | 8-12b IMU state, 12-16b gate accum | High                        | Medium-High             | Medium      | P1       |
| M10 | Late-fusion visual + IMU encoder             | Safer multimodal fusion baseline               | Concat + small fusion MLP                        | Ternary branch weights, non-ternary fusion accum okay            | 8-12b mem, 12-16b fusion accum    | Medium-High                  | Low-Medium              | Low         | P1       |
| M11 | Trajectory-conditioned gating block          | Condition perception on Skydreamer-like latent | Hard gate / gain modulation                      | Keep trajectory latent low-bit, not fully ternary at first       | 8-12b latent, 12-16b accum        | High                         | High                    | Medium      | P1       |
| M12 | Predictive-coding / residual-error block     | Emphasize mismatch vs expected motion          | Pred - Obs then late fuse                        | Ternary on residual path works well; predictor may stay >2b      | 10-16b residual state             | Very High if prediction good | High                    | Medium      | P1       |
| M13 | Hybrid-bit ternary core                      | Practical deployment compromise                | Depends on host block                            | Ternary weights/spikes, 8b states where needed                   | 8b mem, 12-16b accum              | Medium-High                  | Low                     | Low         | P1       |
| M14 | Early-exit collision head                    | Reflex obstacle decision                       | Branch off shallow shared latent                 | Ternary head fine; confidence logic low-bit                      | 8-12b head state                  | Medium                       | Medium                  | Low         | P1       |
| M15 | Navigation-value head                        | Feed world-model / planner                     | Late shared latent                               | Ternary okay if head is shallow; accum wider                     | 8-12b state, 12-16b accum         | Medium                       | Medium                  | Low         | P1       |
| M16 | Multi-timescale LIF family                   | Reflex + integration channels                  | Parallel branches, late sum/concat               | Ternary weights robust; distinct leaks per branch                | 8-12b mem per timescale           | Medium-High                  | Medium                  | Low         | P1       |
| M17 | Hard-gated mixture of experts                | Separate experts for fovea/periphery/tasks     | Hard top-k expert routing                        | Ternary experts; gate maybe low-bit int                          | 8-12b expert state                | Very High active sparsity    | High                    | High        | P2       |
| M18 | Tiny recurrent fusion core                   | Short memory over vision+IMU+traj              | Late fusion into small recurrent block           | Keep recurrence non-ternary first or hybrid-bit                  | 10-16b recurrent state            | Medium                       | Very High               | High        | P2       |
| M19 | Motion-compensated visual front-end          | De-rotation / stabilize input                  | Pre-fusion system block                          | Usually not ternary internally; ternary after warp/shift         | 8-16b motion state                | High downstream              | Medium-High             | Medium      | P2       |
| M20 | Peripheral low-res saliency branch           | Cheap scene scanning                           | Late fusion or routing only                      | Ternary convs + k-WTA very suitable                              | 8-12b mem, 12-16b accum           | Very High                    | Low-Medium              | Low         | P0       |

## Combined Stacks to Test

### Stack A (safest first)
- M20 Peripheral low-res saliency branch.
- M6 k-WTA saliency gate.
- M7 Hard gaze-control head.
- M1 or M2 Foveal ternary Conv-LIF backbone.
- M4 Stereo correlation branch.
- M10 Late-fusion visual + IMU encoder.
- M14 Early-exit collision head.
- M15 Navigation-value head.

Why:
- Late fusion.
- Hard routing.
- Minimal recurrence.

### Stack B (predictive upgrade)
- Stack A.
- M11 Trajectory-conditioned gating.
- M12 Predictive-coding/residual-error block.
- M16 Multi-timescale LIF family.

Why:
- Better world-model alignment.
- Can improve sparsity through prediction.

### Stack C (aggressive efficiency)
- M20 Peripheral branch.
- M6 k-WTA.
- M8 Region-activation router.
- M3 Structured-sparse ternary conv.
- M9 IMU-conditioned visual gating.
- M4 Stereo correlation.
- M14 Early-exit collision head.

Why later:
- More control complexity and tighter calibration coupling.

## Ternary Policy Guidance by Module

### Ternarize early
- Conv/dense weights in visual branches.
- Stereo correlation kernels/signed filters.
- Saliency branch weights.
- Shallow output heads.
- Structured-sparse conv blocks.

### Keep wider initially
- Membrane state.
- Recurrent hidden state.
- Timestamps/time surfaces.
- IMU integrators.
- Predictor residual state.
- Fusion accumulators.
- Confidence/calibration scalars.

### Practical starting compromise
- Weights ternary.
- Spikes binary.
- State 8-12 bit.
- Accumulators 12-16 bit.
- Hard top-k or mask gates.

## Calibration and Evaluation Metrics
Track per experiment:
- Task score (obstacle AUC, collision recall, nav-value error).
- Spike rate.
- Saturation rate.
- Sign-flip sensitivity.
- Branch dominance.
- Gate stability.
- Early-exit hit rate.
- Active tile ratio.
- BRAM proxy.
- DSP proxy.

## Suggested Run Waves

### Wave 1
1. M1 Ternary Conv-LIF backbone.
2. M20 Peripheral saliency branch.
3. M6 k-WTA saliency gate.
4. M4 Stereo correlation branch.
5. M10 Late-fusion visual + IMU.
6. M14 Collision head.
7. M15 Navigation head.

### Wave 2
8. M2 Residual Ternary Conv-LIF.
9. M7 Gaze-control head.
10. M8 Region router.
11. M16 Multi-timescale LIF.
12. M13 Hybrid-bit ternary core.

### Wave 3
13. M11 Trajectory-conditioned gating.
14. M12 Predictive-coding block.
15. M9 IMU-conditioned visual gating.
16. M17 Hard-gated experts.
17. M18 Tiny recurrent fusion core.

## Risk Legend
- Low: regular compute, easier export path.
- Medium: moderate routing/control complexity.
- High: significant branch-coupling/routing/recurrent calibration burden.

## Recommended Combined Direction
Most ternary-robust combined architecture for this project:
- Peripheral ternary saliency branch.
- k-WTA/hard routing.
- Foveal residual ternary Conv-LIF.
- Stereo coincidence branch.
- Late IMU fusion.
- Trajectory-conditioned hard gating.
- Shared latent.
- Collision head + navigation head.
- Early-exit reflex head.

This is safer than:
- Soft attention-centric designs.
- Deep recurrent multimodal fusion.
- Early dense global multimodal mixing.
- Fully ternary internal state everywhere.

## Experiment Registry and Execution Spec
This section consolidates the execution-spec material originally drafted in `Untitled-2.md` so architecture and experiment operations are maintained in one canonical document.

### Canonical experiment fields
Each experiment config should fully describe:
- Model family.
- Branch composition.
- Fusion strategy.
- Ternary/precision policy.
- Neuron/state precision.
- Routing and gating behavior.
- Training setup.
- Evaluation setup.
- Hardware proxy metrics.
- Promotion criteria.

Rule:
- One experiment config = one fully reproducible architecture + training + evaluation + hardware-estimation unit.

Avoid underspecified configs such as:
- `model=ternary_snn`
- `fusion=some_late_fusion`

### CSV column set
Use this for dashboards, ranking, and Optuna summaries.

Core CSV columns:

```csv
experiment_id,priority_wave,status,notes
model_family,role_in_system,architecture_variant
vision_branch,stereo_branch,imu_branch,trajectory_branch
fusion_type,fusion_stage,gating_type,router_type
ternary_policy,weight_precision,activation_precision,spike_precision
membrane_width_bits,synapse_acc_width_bits,fusion_acc_width_bits,recurrent_state_width_bits
neuron_model,leak_mode,threshold_mode,reset_mode,refractory_steps
input_representation,foveation_type,time_surface_enabled,motion_comp_enabled
saliency_enabled,kwta_enabled,gaze_control_enabled,early_exit_enabled
structured_sparsity_type,structured_sparsity_level,expected_spike_sparsity
train_dataset,eval_dataset,task_type,task_heads,loss_config
optimizer,learning_rate,batch_size,sequence_length,timesteps
quantization_aware_training,surrogate_gradient,normalization_type
calibration_sensitivity,branch_balance_risk,fpga_risk
target_metric,secondary_metrics,promotion_gate
mlflow_experiment_name,mlflow_tags,optuna_study_name,optuna_search_space_ref
dagster_job_name,dataset_version_ref,code_version_ref,seed
estimated_bram_score,estimated_dsp_score,estimated_router_complexity,estimated_latency_class
result_primary,result_secondary,spike_rate,active_tile_ratio,early_exit_rate
```

Minimal recommended subset:

```csv
experiment_id,status,model_family,architecture_variant
fusion_type,gating_type,ternary_policy
weight_precision,membrane_width_bits,synapse_acc_width_bits
input_representation,foveation_type,time_surface_enabled
stereo_branch,imu_branch,trajectory_branch
kwta_enabled,gaze_control_enabled,early_exit_enabled
structured_sparsity_type,expected_spike_sparsity
task_type,task_heads,loss_config
optimizer,learning_rate,batch_size,timesteps
target_metric,promotion_gate
fpga_risk,estimated_bram_score,estimated_router_complexity
result_primary,spike_rate,active_tile_ratio
```

### YAML schema template
Master template:

```yaml
experiment:
	id: exp_0001
	name: ternary_foveal_stereo_kwta_v1
	priority_wave: P0
	status: planned
	notes: >
		Baseline ternary-robust foveal stack with stereo branch and hard saliency routing.

tracking:
	mlflow_experiment_name: ternary_snn_fpga
	mlflow_tags:
		project: drone_foveal_snn
		family: ternary_robust
		wave: P0
		hardware_target: fpga
	optuna_study_name: ternary_snn_fpga_p0
	optuna_search_space_ref: search_spaces/p0_baseline.yaml
	dagster_job_name: train_eval_hw_estimate

repro:
	seed: 42
	code_version_ref: git:commit_sha_here
	dataset_version_ref: data:dataset_version_here

task:
	type: multi_task_navigation
	heads:
		- collision_risk
		- navigation_value
	target_metric: collision_auc
	secondary_metrics:
		- nav_value_mae
		- spike_rate
		- active_tile_ratio
		- early_exit_rate
	promotion_gate:
		collision_auc_min: 0.91
		nav_value_mae_max: 0.08
		spike_rate_max: 0.12
		active_tile_ratio_max: 0.35

data:
	train_dataset: dataset_train_v1
	eval_dataset: dataset_eval_v1
	input_representation: pseudo_event
	sequence_length: 16
	timesteps: 16
	image_geometry: spherical
	foveation_type: dual_path
	time_surface_enabled: true
	motion_comp_enabled: false

architecture:
	model_family: residual_ternary_conv_lif
	architecture_variant: foveal_backbone_with_stereo_and_kwta
	role_in_system: shared_perception_backbone

	branches:
		vision_branch:
			enabled: true
			type: residual_ternary_conv_lif
			channels: [16, 32, 48]
			kernel_sizes: [3, 3, 3]
			strides: [1, 1, 1]

		stereo_branch:
			enabled: true
			type: stereo_coincidence
			disparity_bins: 16
			local_window: 5

		imu_branch:
			enabled: true
			type: late_fusion_mlp
			hidden_dims: [16, 16]

		trajectory_branch:
			enabled: false
			type: none

	fusion:
		type: late_concat
		stage: post_branch_embedding
		gating_type: hard_topk
		router_type: tile_router

	saliency:
		enabled: true
		peripheral_branch_type: ternary_conv_lif
		kwta_enabled: true
		topk_tiles: 4

	gaze_control:
		enabled: true
		type: hard_policy_head
		candidate_tiles: 16

	early_exit:
		enabled: true
		exit_head_depth: shallow
		confidence_threshold: 0.9

neurons:
	neuron_model: lif
	leak_mode: fixed_per_layer
	threshold_mode: learned_per_channel
	reset_mode: subtract
	refractory_steps: 1

precision:
	ternary_policy: weights_only_ternary
	weight_precision: ternary
	activation_precision: binary_spike
	spike_precision: binary
	membrane_width_bits: 10
	synapse_acc_width_bits: 16
	fusion_acc_width_bits: 16
	recurrent_state_width_bits: 0

sparsity:
	structured_sparsity_type: channel
	structured_sparsity_level: 0.25
	expected_spike_sparsity: high

training:
	optimizer: adamw
	learning_rate: 0.001
	batch_size: 32
	quantization_aware_training: true
	surrogate_gradient: fast_sigmoid
	normalization_type: none
	loss_config:
		collision_risk: bce
		navigation_value: mse
		auxiliary:
			- spike_rate_regularizer
			- gate_stability_regularizer

evaluation:
	calibration_sensitivity: medium
	branch_balance_risk: medium
	fpga_risk: medium

hardware_proxy:
	estimated_bram_score: medium
	estimated_dsp_score: low
	estimated_router_complexity: medium
	estimated_latency_class: low_latency

results:
	result_primary: null
	result_secondary: null
	spike_rate: null
	active_tile_ratio: null
	early_exit_rate: null
```

### Suggested controlled vocab and enums
Model family:

```yaml
- ternary_conv_lif
- residual_ternary_conv_lif
- structured_sparse_ternary_conv
- stereo_coincidence
- stereo_correlation
- late_fusion_visual_imu
- imu_conditioned_visual_gating
- trajectory_conditioned_gating
- predictive_coding_block
- peripheral_saliency_branch
- hard_gaze_policy
- region_activation_router
- multi_timescale_lif
- early_exit_head
- hybrid_bit_ternary_core
- tiny_recurrent_fusion
```

`fusion.type`:

```yaml
- none
- late_concat
- late_sum
- hard_gate
- topk_select
- residual_error_fusion
- post_branch_embedding
```

`gating_type`:

```yaml
- none
- hard_topk
- binary_mask
- channel_gate
- tile_gate
- threshold_modulation
```

`ternary_policy`:

```yaml
- weights_only_ternary
- weights_and_preactivation_ternary
- ternary_core_hybrid_state
- full_ternary_except_accumulators
```

`foveation_type`:

```yaml
- none
- log_polar
- dual_path
- tiled_fovea
- learned_gaze_crop
```

`structured_sparsity_type`:

```yaml
- none
- channel
- block
- kernel
- tile
```

`expected_spike_sparsity`:

```yaml
- low
- medium
- high
- very_high
```

`fpga_risk`:

```yaml
- low
- medium
- high
```

### Concrete experiment entries
Experiment 1 (safest baseline):

```yaml
experiment:
	id: exp_p0_001
	name: peripheral_kwta_foveal_stereo_lateimu
	priority_wave: P0
	status: planned

task:
	type: multi_task_navigation
	heads: [collision_risk, navigation_value]
	target_metric: collision_auc
	secondary_metrics: [nav_value_mae, spike_rate, active_tile_ratio]
	promotion_gate:
		collision_auc_min: 0.90
		spike_rate_max: 0.15

data:
	input_representation: pseudo_event
	image_geometry: spherical
	foveation_type: dual_path
	time_surface_enabled: true
	motion_comp_enabled: false
	sequence_length: 16
	timesteps: 16

architecture:
	model_family: residual_ternary_conv_lif
	architecture_variant: peripheral_kwta_foveal_stereo_lateimu
	role_in_system: shared_perception_backbone
	branches:
		vision_branch:
			enabled: true
			type: residual_ternary_conv_lif
		stereo_branch:
			enabled: true
			type: stereo_coincidence
		imu_branch:
			enabled: true
			type: late_fusion_mlp
		trajectory_branch:
			enabled: false
			type: none
	fusion:
		type: late_concat
		stage: post_branch_embedding
		gating_type: hard_topk
		router_type: tile_router
	saliency:
		enabled: true
		peripheral_branch_type: ternary_conv_lif
		kwta_enabled: true
		topk_tiles: 4
	gaze_control:
		enabled: true
		type: hard_policy_head
		candidate_tiles: 16
	early_exit:
		enabled: true
		exit_head_depth: shallow
		confidence_threshold: 0.9

neurons:
	neuron_model: lif
	leak_mode: fixed_per_layer
	threshold_mode: learned_per_channel
	reset_mode: subtract
	refractory_steps: 1

precision:
	ternary_policy: weights_only_ternary
	weight_precision: ternary
	activation_precision: binary_spike
	spike_precision: binary
	membrane_width_bits: 10
	synapse_acc_width_bits: 16
	fusion_acc_width_bits: 16
	recurrent_state_width_bits: 0

sparsity:
	structured_sparsity_type: channel
	structured_sparsity_level: 0.25
	expected_spike_sparsity: high

training:
	optimizer: adamw
	learning_rate: 0.001
	batch_size: 32
	quantization_aware_training: true
	surrogate_gradient: fast_sigmoid
	normalization_type: none
```

Experiment 2 (trajectory-conditioned):

```yaml
experiment:
	id: exp_p1_002
	name: trajectory_conditioned_predictive_foveal_stack
	priority_wave: P1
	status: planned

task:
	type: multi_task_navigation
	heads: [collision_risk, navigation_value]
	target_metric: collision_auc
	secondary_metrics: [nav_value_mae, spike_rate, active_tile_ratio, gate_stability]
	promotion_gate:
		collision_auc_min: 0.91
		nav_value_mae_max: 0.08

data:
	input_representation: pseudo_event
	image_geometry: spherical
	foveation_type: tiled_fovea
	time_surface_enabled: true
	motion_comp_enabled: false
	sequence_length: 20
	timesteps: 20

architecture:
	model_family: trajectory_conditioned_gating
	architecture_variant: predictive_error_conditioned_navigation_stack
	role_in_system: predictive_perception_encoder
	branches:
		vision_branch:
			enabled: true
			type: residual_ternary_conv_lif
		stereo_branch:
			enabled: true
			type: stereo_correlation
		imu_branch:
			enabled: true
			type: late_fusion_mlp
		trajectory_branch:
			enabled: true
			type: lowbit_projection
	fusion:
		type: residual_error_fusion
		stage: post_branch_embedding
		gating_type: binary_mask
		router_type: tile_router
	saliency:
		enabled: true
		peripheral_branch_type: ternary_conv_lif
		kwta_enabled: true
		topk_tiles: 4
	gaze_control:
		enabled: true
		type: hard_policy_head
		candidate_tiles: 25
	early_exit:
		enabled: true
		exit_head_depth: shallow
		confidence_threshold: 0.92

neurons:
	neuron_model: lif
	leak_mode: fixed_per_layer
	threshold_mode: learned_per_channel
	reset_mode: subtract
	refractory_steps: 1

precision:
	ternary_policy: ternary_core_hybrid_state
	weight_precision: ternary
	activation_precision: binary_spike
	spike_precision: binary
	membrane_width_bits: 10
	synapse_acc_width_bits: 16
	fusion_acc_width_bits: 16
	recurrent_state_width_bits: 0

sparsity:
	structured_sparsity_type: tile
	structured_sparsity_level: 0.5
	expected_spike_sparsity: very_high

training:
	optimizer: adamw
	learning_rate: 0.0007
	batch_size: 24
	quantization_aware_training: true
	surrogate_gradient: fast_sigmoid
	normalization_type: none
```

Experiment 3 (IMU-conditioned gating):

```yaml
experiment:
	id: exp_p1_003
	name: imu_conditioned_visual_gate_stack
	priority_wave: P1
	status: planned

architecture:
	model_family: imu_conditioned_visual_gating
	architecture_variant: hard_threshold_modulated_vision
	role_in_system: ego_motion_conditioned_backbone
	branches:
		vision_branch:
			enabled: true
			type: structured_sparse_ternary_conv
		stereo_branch:
			enabled: true
			type: stereo_coincidence
		imu_branch:
			enabled: true
			type: lowbit_mlp
		trajectory_branch:
			enabled: false
			type: none
	fusion:
		type: hard_gate
		stage: post_branch_embedding
		gating_type: threshold_modulation
		router_type: tile_router

precision:
	ternary_policy: ternary_core_hybrid_state
	weight_precision: ternary
	activation_precision: binary_spike
	spike_precision: binary
	membrane_width_bits: 10
	synapse_acc_width_bits: 16
	fusion_acc_width_bits: 14
	recurrent_state_width_bits: 0

sparsity:
	structured_sparsity_type: channel
	structured_sparsity_level: 0.4
	expected_spike_sparsity: high

evaluation:
	calibration_sensitivity: high
	branch_balance_risk: medium
	fpga_risk: medium
```

Experiment 4 (aggressive efficiency):

```yaml
experiment:
	id: exp_p2_004
	name: region_router_structured_sparse_efficiency_stack
	priority_wave: P2
	status: planned

architecture:
	model_family: region_activation_router
	architecture_variant: sparse_tile_routed_foveal_stack
	role_in_system: aggressive_compute_reduction
	branches:
		vision_branch:
			enabled: true
			type: structured_sparse_ternary_conv
		stereo_branch:
			enabled: true
			type: stereo_coincidence
		imu_branch:
			enabled: true
			type: late_fusion_mlp
		trajectory_branch:
			enabled: true
			type: lowbit_projection
	fusion:
		type: late_concat
		stage: post_branch_embedding
		gating_type: tile_gate
		router_type: tile_router
	saliency:
		enabled: true
		peripheral_branch_type: ternary_conv_lif
		kwta_enabled: true
		topk_tiles: 2
	gaze_control:
		enabled: true
		type: hard_policy_head
		candidate_tiles: 36
	early_exit:
		enabled: true
		exit_head_depth: shallow
		confidence_threshold: 0.95

precision:
	ternary_policy: full_ternary_except_accumulators
	weight_precision: ternary
	activation_precision: binary_spike
	spike_precision: binary
	membrane_width_bits: 8
	synapse_acc_width_bits: 14
	fusion_acc_width_bits: 16
	recurrent_state_width_bits: 0

sparsity:
	structured_sparsity_type: tile
	structured_sparsity_level: 0.6
	expected_spike_sparsity: very_high

evaluation:
	calibration_sensitivity: high
	branch_balance_risk: high
	fpga_risk: high
```

### Optuna search-space templates
`search_spaces/p0_baseline.yaml`:

```yaml
search_space:
	learning_rate:
		type: float
		low: 0.0003
		high: 0.003
		log: true

	batch_size:
		type: categorical
		choices: [16, 24, 32, 48]

	membrane_width_bits:
		type: categorical
		choices: [8, 10, 12]

	synapse_acc_width_bits:
		type: categorical
		choices: [12, 14, 16]

	structured_sparsity_level:
		type: categorical
		choices: [0.0, 0.25, 0.4]

	topk_tiles:
		type: categorical
		choices: [2, 4, 6]

	confidence_threshold:
		type: float
		low: 0.85
		high: 0.97

	leak_mode_variant:
		type: categorical
		choices:
			- fixed_per_layer
			- fixed_per_channel

	threshold_mode_variant:
		type: categorical
		choices:
			- learned_per_layer
			- learned_per_channel
```

`search_spaces/p1_predictive.yaml`:

```yaml
search_space:
	learning_rate:
		type: float
		low: 0.0002
		high: 0.002
		log: true

	structured_sparsity_level:
		type: categorical
		choices: [0.25, 0.4, 0.5, 0.6]

	topk_tiles:
		type: categorical
		choices: [2, 4, 8]

	membrane_width_bits:
		type: categorical
		choices: [8, 10, 12]

	synapse_acc_width_bits:
		type: categorical
		choices: [14, 16]

	trajectory_gate_scale:
		type: float
		low: 0.25
		high: 2.0

	predictive_error_weight:
		type: float
		low: 0.1
		high: 2.0
		log: true
```

### CSV starter rows

```csv
experiment_id,priority_wave,status,model_family,architecture_variant,fusion_type,gating_type,ternary_policy,weight_precision,membrane_width_bits,synapse_acc_width_bits,input_representation,foveation_type,time_surface_enabled,stereo_branch,imu_branch,trajectory_branch,kwta_enabled,gaze_control_enabled,early_exit_enabled,structured_sparsity_type,structured_sparsity_level,expected_spike_sparsity,task_type,task_heads,target_metric,fpga_risk
exp_p0_001,P0,planned,residual_ternary_conv_lif,peripheral_kwta_foveal_stereo_lateimu,late_concat,hard_topk,weights_only_ternary,ternary,10,16,pseudo_event,dual_path,true,true,true,false,true,true,true,channel,0.25,high,multi_task_navigation,"collision_risk|navigation_value",collision_auc,medium
exp_p1_002,P1,planned,trajectory_conditioned_gating,predictive_error_conditioned_navigation_stack,residual_error_fusion,binary_mask,ternary_core_hybrid_state,ternary,10,16,pseudo_event,tiled_fovea,true,true,true,true,true,true,true,tile,0.5,very_high,multi_task_navigation,"collision_risk|navigation_value",collision_auc,medium
exp_p1_003,P1,planned,imu_conditioned_visual_gating,hard_threshold_modulated_vision,hard_gate,threshold_modulation,ternary_core_hybrid_state,ternary,10,16,pseudo_event,dual_path,true,true,true,false,true,false,true,channel,0.4,high,multi_task_navigation,"collision_risk|navigation_value",collision_auc,medium
exp_p2_004,P2,planned,region_activation_router,sparse_tile_routed_foveal_stack,late_concat,tile_gate,full_ternary_except_accumulators,ternary,8,14,pseudo_event,tiled_fovea,true,true,true,true,true,true,true,tile,0.6,very_high,multi_task_navigation,"collision_risk|navigation_value",collision_auc,high
```

### Recommended MLflow tags

```yaml
mlflow_tags:
	project: drone_foveal_snn
	modality: spherical_stereo_event
	hardware_target: fpga
	precision_family: ternary
	fusion_family: late_fusion
	routing_family: hard_topk
	branch_config: vision_stereo_imu
	wave: P0
	promotion_stage: baseline
```

Additional useful tags:
- `time_surface=true|false`
- `motion_comp=true|false`
- `traj_conditioned=true|false`
- `gaze_control=true|false`
- `router=tile_router`
- `risk=medium`

### Promotion logic template

```yaml
promotion:
	promote_to_wave: P1
	requires:
		collision_auc_min: 0.90
		nav_value_mae_max: 0.10
		spike_rate_max: 0.15
		active_tile_ratio_max: 0.40
		fpga_risk_allowed:
			- low
			- medium
	tie_break_order:
		- collision_auc
		- active_tile_ratio
		- spike_rate
		- estimated_router_complexity
```

### Hardware proxy block template

```yaml
hardware_proxy:
	estimated_bram_score: medium
	estimated_dsp_score: low
	estimated_router_complexity: medium
	estimated_latency_class: low_latency
	notes:
		- stereo branch local only
		- topk tile router active
		- no recurrent state
```

Numeric proxy variant:

```yaml
hardware_proxy_numeric:
	estimated_weight_bytes: 184320
	estimated_state_bytes: 32768
	estimated_active_tiles_mean: 3.4
	estimated_spike_events_per_step: 5210
	estimated_mac_equivalent_per_step: 14320
	estimated_router_edges: 896
```

### Suggested repo structure for experiment ops

```text
experiments/
	registry.csv
	p0/
		exp_p0_001.yaml
		exp_p0_005.yaml
	p1/
		exp_p1_002.yaml
		exp_p1_003.yaml
	p2/
		exp_p2_004.yaml

search_spaces/
	p0_baseline.yaml
	p1_predictive.yaml
	p2_router.yaml

schemas/
	experiment_schema.yaml
	search_space_schema.yaml
```

Keep code modules separated by responsibility:
- Config loading.
- Optuna sampling override.
- MLflow logging.
- Hardware estimation pass.
- Promotion decision.

### Naming convention
Pattern:

```text
{wave}_{vision}_{fusion}_{special}_{precision}_{version}
```

Examples:

```text
p0_foveal_latefuse_stereo_kwta_ternary_v1
p1_foveal_prederr_trajgate_ternary_v2
p2_router_sparse_tilegate_ternary_v1
```

### First six configs to instantiate
1. `p0_foveal_latefuse_stereo_kwta_ternary_v1`
2. `p0_foveal_latefuse_stereo_kwta_hybridbit_v1`
3. `p0_foveal_latefuse_stereo_imu_ternary_v1`
4. `p1_foveal_trajgate_prederr_ternary_v1`
5. `p1_foveal_imugate_sparse_ternary_v1`
6. `p2_foveal_router_tilegate_sparse_ternary_v1`

### Search layering rule
Do not let Optuna mutate everything at once. Split search into layers.

Layer A:
- Learning rate.
- Thresholds.
- Leak.
- Top-k.
- Confidence threshold.

Layer B:
- Membrane width.
- Accumulator width.
- Sparsity level.

Layer C:
- Fusion variant.
- Branch enable/disable.
- Trajectory conditioning.
- Router policy.

## References
General and foundational links:
1. Spyx docs: https://spyx.readthedocs.io/?utm_source=chatgpt.com
2. Spyx introduction: https://spyx.readthedocs.io/en/latest/introduction/?utm_source=chatgpt.com
3. FPGA SNN survey: https://arxiv.org/pdf/2307.03910?utm_source=chatgpt.com
4. SNNAX in JAX: https://juser.fz-juelich.de/record/1038043/files/SNNAX-Spiking%20Neural%20Networks%20in%20JAX.pdf?utm_source=chatgpt.com
5. Event-driven spike sparse convolution: https://arxiv.org/abs/2412.07360?utm_source=chatgpt.com
6. SNN architecture search survey: https://arxiv.org/html/2510.14235v1?utm_source=chatgpt.com
7. Spyx paper entry used in prior notes: https://arxiv.org/html/2402.18994v1?utm_source=chatgpt.com
8. Sparse spike-driven hardware accelerator: https://arxiv.org/html/2501.07825v1?utm_source=chatgpt.com
9. Loihi 2 brief: https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf?utm_source=chatgpt.com
10. Event-camera sparse spiking learning paper: https://arxiv.org/pdf/2104.12579?utm_source=chatgpt.com
11. Intel neuromorphic overview: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html?utm_source=chatgpt.com

Spherical/foveated/mobile navigation references used in prior notes:
12. Event camera + mobile embodied perception: https://arxiv.org/html/2503.22943v4?utm_source=chatgpt.com
13. Real-time neuromorphic navigation: https://arxiv.org/html/2503.09636v1?utm_source=chatgpt.com
14. Spiking neural-invariant Kalman fusion: https://arxiv.org/html/2601.08248v1?utm_source=chatgpt.com
15. Spike-FlowNet entry: https://arxiv.org/pdf/2003.06696?utm_source=chatgpt.com
16. Multi-step SNN depth prediction: https://arxiv.org/pdf/2211.12156?utm_source=chatgpt.com
17. SNN hardware implementations and methods review: https://arxiv.org/pdf/2005.01467?utm_source=chatgpt.com
18. Event-based vision on FPGAs survey: https://arxiv.org/html/2407.08356v1?utm_source=chatgpt.com
19. Neuromorphic vision sensor entry used in prior notes: https://arxiv.org/html/2504.08588v1?utm_source=chatgpt.com
20. Hybrid quantization co-design (Frontiers): https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1665778/full?utm_source=chatgpt.com
