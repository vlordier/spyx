# Tonic Dataset Exploration

This document covers all Tonic datasets requested by the user, grouped by task category.
Status key:

| Symbol | Meaning |
|--------|---------|
| ✅ Ready | Works out-of-the-box in this environment |
| ⚠️ Fetch risk | URL is reachable but Tonic's download helper can be blocked by WAF |
| 🔶 Skeleton | Class exists but download/URL information is missing or incomplete |
| ❌ Missing | Not present in the installed Tonic version |

For all datasets that require downloads, set `save_to` to a local directory. Tonic will
auto-download and extract on first access if the data is not already present.

---

## Pose Estimation · Visual Odometry · SLAM

### DAVISDATA
- **Class:** `tonic.datasets.DAVISDATA`
- **Constructor:** `(save_to, recording, transform=None, target_transform=None, transforms=None)`
- **Signature requires:** `recording` — must be a `str` or list of strs naming the recording(s) to load from the DAVIS dataset.
- **Sensor:** `[240, 180, 2]`
- **Source:** http://rpg.ifi.uzh.ch/datasets/davis/
- **Notes:** Unlike other datasets, this one does **not** have a `train` split flag — you must know
  in advance which recording(s) you want. Multiple recordings can be passed as a list. Not a
  classification dataset; primarily used for optical-flow, VO, and SLAM benchmarks.

### DSEC
- **Class:** `tonic.datasets.DSEC`
- **Constructor:** `(save_to, split, data_selection, target_selection=None, transform=None, …)`
- **Signature requires:** `split`, `data_selection`, and `target_selection` — all must be specified.
- **Sensor:** `[640, 480, 2]`
- **Source:** https://download.ifi.uzh.ch/rpg/DSEC/
- **Notes:** Designed for stereo event-based perception. The multi-parameter constructor makes it
  the most complex in this group. `split` is typically `"train"` or `"test"`; `data_selection`
  and `target_selection` control which sub-sequences and labels are loaded. Likely requires
  significant disk space and careful parameter selection.

### MVSEC
- **Class:** `tonic.datasets.MVSEC`
- **Constructor:** `(save_to, scene, transform=None, target_transform=None, transforms=None)`
- **Signature requires:** `scene` — must name a specific MVSEC scene.
- **Sensor:** `[346, 260, 2]`
- **Source:** http://visiondata.cis.upenn.edu/mvsec/
- **Notes:** Multi-vehicle stereo event camera dataset. Like DAVISDATA, `scene` must be specified
  explicitly. Covers indoor and outdoor scenes for odometry and depth estimation.

### ThreeET_Eyetracking
- **Class:** ❌ **Missing** — not in the installed Tonic version.
- **Notes:** The class name in Tonic's registry may differ (e.g. `ThreeET` or `ThreeETEye`).
  Check `tonic.datasets.__dir__()` if you need this.

---

## Object Tracking

### EBSSA
- **Class:** `tonic.datasets.EBSSA`
- **Constructor:** `(save_to, split='labelled', transform=None, …)`
- **Sensor:** `[240, 180, 2]`
- **Notes:** Event-based salient scene analytics. Only one split (`'labelled'`). Simpler than DSEC
  or MVSEC — just pass `save_to` and optionally the split. Likely a tracking / saliency dataset.

### TUMVIE
- **Class:** `tonic.datasets.TUMVIE`
- **Constructor:** `(save_to, recording, transform=None, target_transform=None, transforms=None)`
- **Signature requires:** `recording` — name or list of recording identifiers.
- **Sensor:** `[1280, 720, 2]` — notably high-resolution compared to most event cameras.
- **Source:** https://tumevent-vi.vision.in.tum.de/
- **Notes:** TUM event-based VI dataset for visual-inertial odometry. Like DAVISDATA, requires
  knowing specific recording names upfront. High resolution means larger storage.

### VPR
- **Class:** `tonic.datasets.VPR`
- **Constructor:** `(save_to, transform=None, target_transform=None, transforms=None)`
- **Sensor:** `[346, 260, 2]`
- **Source:** https://zenodo.org/record/4302805/files/
- **Notes:** Visual Place Recognition. Simplest of this group — just `save_to`. No `train` flag,
  no recording names. Returns event recordings labelled by location for place-recognition tasks.

---

## Visual Event Stream Classification

### ASLDVS
- **Class:** `tonic.datasets.ASLDVS`
- **Constructor:** `(save_to, transform=None, target_transform=None, transforms=None)`
- **Sensor:** `[240, 180, 2]`
- **Classes:** 26 (fingerspelling alphabet: a–z)
- **Source:** Dropbox (MD5: `33f8b87bf45edc0bfed0de41822279b9`)
- **Status:** ✅ Ready — download URL is present. Dropbox URLs are generally more accessible than
  Figshare in restricted network environments.
- **Use case:** ASL fingerspelling classification — good compact alternative to DVS Gesture.

### CIFAR10DVS
- **Class:** `tonic.datasets.CIFAR10DVS`
- **Constructor:** `(save_to, transform=None, target_transform=None, transforms=None)`
- **Sensor:** `[128, 128, 2]`
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Source:** Figshare (MD5: `ce3a4a0682dc0943703bd8f749a7701c`)
- **Status:** ⚠️ Fetch risk — Figshare endpoint was returning WAF challenges in this environment.
- **Use case:** Drop-in replacement for CIFAR-10 with event-camera data. Already pre-converted.
  The 128×128 sensor size is manageable for most models.

### DVSGesture (IBM DVS Gestures)
- **Class:** `tonic.datasets.DVSGesture`
- **Constructor:** `(save_to, train=True, transform=None, …)`
- **Sensor:** `[128, 128, 2]`
- **Classes:** 11 (Hand clapping, Right/Left hand wave, Right arm cw/ccw, and others)
- **Source:** Figshare (train MD5: `3a8b0d4120a166bac7591f77409cb105`; test MD5: `56070e45dadaa85fff82e0fbfbc06de5`)
- **Status:** ⚠️ Fetch risk — Figshare WAF blocked Tonic's helper in this environment. However,
  if you already have `ibmGestureTrain/` and `ibmGestureTest/` folders extracted under
  `data/DVSGesture/`, the `research/end_to_end/dvs_gesture_foveated.py` experiment will use
  them directly without invoking the download path.
- **Use case:** Gesture classification with 11 action classes. Well-established benchmark.
  Existing `spyx` experiment: `research/end_to_end/dvs_gesture_foveated.py`.

### NCALTECH101
- **Class:** `tonic.datasets.NCALTECH101`
- **Constructor:** `(save_to, transform=None, target_transform=None, transforms=None)`
- **Sensor:** ❓ `null` — the class attribute is `None` in the installed version.
- **Source:** Mendeley (MD5: `66201824eabb0239c7ab992480b50ba3`)
- **Status:** ⚠️ Caution — `sensor_size = None` in the class definition. This may cause issues
  with transforms that rely on it (e.g. `ToFrame` without an explicit `sensor_size` argument).
  Validate before using in a pipeline that hard-codes sensor dimensions.
- **Use case:** Caltech 101 object classification with event camera. 101 classes.

### NMNIST
- **Class:** `tonic.datasets.NMNIST`
- **Constructor:** `(save_to, train=True, first_saccade_only=False, stabilize=False, transform=None, …)`
- **Sensor:** `[34, 34, 2]`
- **Classes:** 10 (digits 0–9)
- **Source:** Mendeley (train MD5: `20959b8e626244a1b502305a9e6e2031`; test MD5: `69ca8762b2fe404d9b9bad1103e97832`)
- **Status:** ✅ Works — already used by `research/end_to_end/nmnist_logpolar.py` and
  `research/end_to_end/nmnist_event_pooling.py` in this repo.
- **Notes:** `first_saccade_only=True` gives only the first saccade (faster, ~1/3 of full data).
  `stabilize=True` includes the stabilization phase. Already has a Spyx loader wrapper.

### POKERDVS
- **Class:** `tonic.datasets.POKERDVS`
- **Constructor:** `(save_to, train=True, transform=None, …)`
- **Sensor:** `[35, 35, 2]`
- **Classes:** 4 (`cl`, `he`, `di`, `sp` — card suits)
- **Source:** Nextcloud (MD5s: train `412bcfb96826e4fcb290558e8c150aae`, test `eef2bf7d0d3defae89a6fa98b07c17af`)
- **Status:** ⚠️ Fetch risk — Nextcloud URLs may require specific headers or session cookies that
  Tonic's simple fetcher does not handle.
- **Use case:** Sequential digit/spike classification on poker-chip sorting tasks. Very small
  sensor (35×35) — good for fast prototyping.

### SMNIST
- **Class:** `tonic.datasets.SMNIST`
- **Constructor:** `(save_to, train=True, duplicate=True, num_neurons=99, dt=1000.0, transform=None, …)`
- **Sensor:** N/A — generates events from MNIST digit images on the fly.
- **Classes:** 10 (digits 0–9)
- **Source:** Google Cloud Storage (MNIST original) — no separate archive needed.
- **Status:** ✅ Ready — no external download needed; events are synthesised from MNIST pixels
  using a Poisson process with `num_neurons` and `dt` (in microseconds). Good for rapid
  prototyping without waiting for a dataset download.
- **Notes:** `duplicate=True` repeats each sample; `num_neurons` controls event rate. The
  resulting event stream is not a real DVS recording but a simulation.

### DVSLip
- **Class:** `tonic.datasets.DVSLip`
- **Constructor:** `(save_to, train=True, transform=None, …)`
- **Sensor:** `[128, 128, 2]`
- **Classes:** 100 (words from the Lip Reading in the Wild dataset)
- **Source:** Google Drive (MD5: `2dcb959255122d4cdeb6094ca282494b`)
- **Status:** ⚠️ Caution — Google Drive URLs require Tonic's gdown-style download path.
  May be blocked in some environments. Validate before relying on it in a pipeline.
- **Use case:** Word-level lipreading from DVS. 100-class classification. Largest label set
  in this group.

---

## Summary Table

| Dataset | Task | Sensor | Classes | Train/Test | Download Risk | Ready in This Env |
|---------|------|--------|---------|-----------|---------------|-------------------|
| **DAVISDATA** | VO/SLAM | 240×180 | — | recording-based | Low (UZH HTTP) | ✅ |
| **DSEC** | Stereo VO | 640×480 | — | param-based | Medium | ⚠️ |
| **MVSEC** | VO/Depth | 346×260 | — | scene-based | Low (UPenn) | ✅ |
| **ThreeET_Eyetracking** | Eye tracking | — | — | — | — | ❌ Missing |
| **EBSSA** | Saliency/Tracking | 240×180 | — | labelled | Unknown | 🔶 Skeleton |
| **TUMVIE** | VI Odometry | 1280×720 | — | recording-based | Medium (TUM) | ⚠️ |
| **VPR** | Place Recognition | 346×260 | — | single split | Low (Zenodo) | ✅ |
| **ASLDVS** | Classification | 240×180 | 26 | single | Low (Dropbox) | ✅ |
| **CIFAR10DVS** | Classification | 128×128 | 10 | single | ⚠️ Figshare WAF | ⚠️ |
| **DVSGesture** | Gestures | 128×128 | 11 | ✅ | ⚠️ Figshare WAF | ⚠️* |
| **NCALTECH101** | Classification | null ⚠️ | 101 | single | Mendeley | ⚠️ |
| **NMNIST** | Digits | 34×34 | 10 | ✅ | Mendeley | ✅ |
| **POKERDVS** | Sequential | 35×35 | 4 | ✅ | ⚠️ Nextcloud | ⚠️ |
| **SMNIST** | Digits | N/A sim | 10 | ✅ | None (synth) | ✅ |
| **DVSLip** | Lipreading | 128×128 | 100 | ✅ | ⚠️ Google Drive | ⚠️ |

*DVSGesture: ready if you already have extracted `ibmGestureTrain/` / `ibmGestureTest/` folders.

---

## Adding a New Spyx Experiment for Another Dataset

The pattern used by `dvs_gesture_foveated.py` is:

```python
# 1. Pick a variant parameter so one script can serve multiple models
parser.add_argument("--variant", choices=(...))

# 2. Use a small subset materialisation path (not full DataLoader):
def _subset_to_arrays(dataset, indices):
    obs = np.stack([dataset[i][0] for i in indices], axis=0)
    labels = np.asarray([dataset[i][1] for i in indices], dtype=np.uint8)
    return jnp.asarray(obs, dtype=jnp.uint8), jnp.asarray(labels, dtype=np.uint8)

# 3. Build ClassificationDataset like the existing experiments
# 4. Wrap with run_classification_experiment() from common.py
```

Key things to check before writing the script:

1. **Sensor size** — is it defined? If `None`, pass an explicit `sensor_size` to `ToFrame`.
2. **Classes** — how many? Check `Dataset.classes` on the class itself (not instance).
3. **train flag** — does it have `train=True/False` splits?
4. **Download URL** — is it a well-known host (Mendeley, Zenodo, Dropbox) or risky (Figshare,
   Google Drive, Nextcloud)?
5. **Transform** — use `transforms.Compose` with `ToFrame` and `np.packbits` like the existing
   NMNIST/SHD experiments. Add `Downsample` for large sensors.
