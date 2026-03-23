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
- [x] Foveated dual-path SNN -> `FoveatedDualPathSNN`
- [x] IMU-conditioned visual SNN -> `IMUConditionedVisualSNN`
- [x] Visual-IMU recurrent fusion -> `VisualIMURecurrentFusionBlock`
- [x] Kalman-style fusion surrogate -> `KalmanStyleSpikingFusionSurrogate`
- [x] Spiking optical-flow branch -> `SpikingOpticalFlowBranch`
- [x] Stereo coincidence/disparity proxy -> `StereoCoincidenceSNN`
- [x] Motion-compensated input front-end -> `MotionCompensatedInputFrontEnd`
- [x] Gaze-control policy head -> `GazeControlPolicyHead`
- [x] Region-activation router -> `RegionActivationRouter`
- [x] Trajectory-conditioned encoder -> `TrajectoryConditionedSpikingEncoder`
- [x] Predictive-coding block -> `PredictiveCodingSNNBlock`
- [x] Collision + navigation multi-head -> `CollisionNavigationMultiHead`
- [x] Structured-sparse spiking CNN -> `StructuredSparseSpikingCNN`
- [x] Early-exit/anytime inference head -> `EarlyExitAnytimeSNN`

### Notes
- These are reference blocks intended for iterative experiments and hardware co-design sweeps, not final production/training recipes.
- Some conceptual families in this document (for example strict log-polar transforms or graph-spherical connectivity) are represented by practical approximations in the current implementation set.

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

## Potential Next Deliverables
- Spyx implementation roadmap: tensor shapes, neuron state, fixed-point formats, ternary constraints, FPGA risk, and success metrics by model family.
- CSV/YAML-ready experiment sheet for Dagster/MLflow/Optuna with ranking and hardware proxies.

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
