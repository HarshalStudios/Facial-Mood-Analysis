# Facial Mood Analysis — Backend

Multi-Representation Based Real-Time Facial Mood Analysis System Using a
Single RGB Camera. This repo is the **backend only**; the frontend and
final integration are built separately (Google AI Studio) against the
API contract this backend exposes.

## Status: Backend integrated — Phases 1–9 infrastructure + production API contract

Phases 1-4 established architecture, camera/face detection, the
multi-representation DIP pipeline, and 22D feature extraction. Phase 5
adds the **FER2013 → feature-dataset pipeline**: given the real FER2013
file (supplied by the project developer, never auto-downloaded), it
reuses the exact same `RepresentationEngine` and `StandardFeatureExtractor`
from Phases 3-4 to produce `features/raw/{train,validation,test}_features.csv`
— each row a 22D raw feature vector plus its FER2013 ground-truth label.
**No scaler is fit and no model is trained yet** — that starts in Phase 6.

### Phase 5 additions

- `app/dataset/fer2013_loader.py` — `FER2013Loader`: detects the source
  file's schema (emotion/pixels/Usage column names, several common
  aliases), parses each row, and reports (not silently drops) malformed
  rows.
- `app/dataset/label_mapping.py` — maps FER2013's raw label (numeric or
  string) onto the project's one canonical emotion order
  (`app.emotion.validation.CANONICAL_EMOTION_LABELS`) — not a second,
  parallel mapping.
- `app/dataset/preprocessing.py` — parses the classic space-separated
  pixel string, and converts a loaded (grayscale) FER2013 image into the
  `(H, W, 3)` BGR uint8 shape `RepresentationEngine.generate()` expects
  by channel replication. This is a compatibility conversion only — it
  does not create new color information, and is documented as such in
  the generated dataset metadata (`grayscale_to_bgr_strategy`).
- `app/dataset/splitter.py` — preserves FER2013's official
  Training/PublicTest/PrivateTest split when the supplied file has one;
  otherwise computes a deterministic stratified split with a recorded
  seed and configured ratios. Refuses (raises) on an ambiguous dataset
  (an official-split column present for some rows but not others, or
  containing unrecognized values) rather than guessing.
- `app/dataset/builder.py` — `FeatureDatasetBuilder`: the batch loop that
  drives `RepresentationEngine` + `StandardFeatureExtractor` per sample,
  with resumability (skips sample IDs already present in the output
  CSV), checkpointed/flushed writes, and periodic progress logging. A
  sample that fails at preprocessing/representation/feature-extraction
  is logged to `outputs/dataset_failures.csv` and excluded from the
  feature CSV — never given an invented value.
- `app/dataset/validation.py` — independent, on-disk validation of the
  generated CSVs: exact schema match, no NaN/Inf, `emotion`/`emotion_id`
  consistency, and a cross-split `sample_id` overlap check.
- `app/dataset/metadata.py` — writes `dataset_metadata.json` (dataset
  source, schema, representation/feature config, split strategy +
  seed, software versions) and `feature_schema.json`.
- `scripts/build_fer2013_features.py` — the Phase 5 entrypoint; run this
  to generate the feature dataset. `scripts/inspect_dataset.py` and
  `scripts/inspect_sample.py` are read-only inspection tools for the
  already-generated output.
- New config in `app/core/config.py`: `dataset_path`, `dataset_name`,
  `features_dir`, `dataset_failures_path`, `dataset_summary_path`,
  `dataset_split_seed`, `dataset_{train,val,test}_ratio`,
  `dataset_checkpoint_batch_size`, `dataset_progress_log_every`.
- New exception: `DatasetError` (`app/core/exceptions.py`).

**Not yet run against a real dataset.** No FER2013 file was supplied in
this development sandbox (`dataset/` only contains `.gitkeep`) — per this
phase's own rule, a missing dataset is reported, never substituted with
synthetic data. The pipeline was verified with a small synthetic,
clearly-not-FER2013 CSV (load → stratified split → build all three
splits → full validation pass → rerun to confirm resumability skips
completed rows) and with the pydantic-independent modules
(`label_mapping.py`, `preprocessing.py`) exercised directly. This
sandbox also has no network access and does not have `pydantic`/`pytest`
installed, so the actual `pytest` suite (Phase 1-5) has not been run
here — the same limitation already documented for Phase 4's DeepFace
integration test. Before trusting this on the real project: `pip install
-r requirements.txt`, place the real FER2013 file at `dataset_path`, run
`pytest`, then `python scripts/build_fer2013_features.py`.

### Phase 4 additions

- `app/features/extractor.py` — `StandardFeatureExtractor`
  (`FeatureExtractor` impl). The single canonical implementation meant
  for both real-time inference (Phase 7) and FER2013 dataset generation
  (Phase 5) — see its module docstring. Requires `grayscale`, `canny`,
  and `lbp` from `Representations` (raises `FeatureExtractionError` if
  any is `None`, e.g. disabled for a Phase 9 ablation run that this
  extractor doesn't support); does **not** require `sobel`, since
  `gradient_energy` recomputes Sobel from `grayscale` directly (see
  below).
- `app/features/lbp_histogram.py`, `edge_density.py`, `image_quality.py`
  — the five handcrafted-feature calculations, each independently
  testable and documented (exact formulas below).
- `app/features/debug.py` — `print_feature_vector()` / `format_feature_vector()`.
- `app/emotion/validation.py` — canonical emotion label order +
  `validate_and_order_emotion_probabilities()`; raises `PredictionError`
  (not `FeatureExtractionError`) for provider output problems.
- `app/emotion/deepface_provider.py` — `DeepFaceEmotionModel`
  (`EmotionModel` impl). Lazy one-time import/init, BGR→RGB conversion
  at its boundary, and converts DeepFace's 0-100 percentages to 0-1
  probabilities. **Not exercised in this sandbox** — no network access
  here to install `deepface`/TensorFlow; see
  `tests/test_deepface_integration.py`.
- `app/emotion/mock_provider.py` — `FixedEmotionModel` /
  `FunctionEmotionModel`, used by every other Phase 4 test so they don't
  depend on DeepFace.
- `app/emotion/factory.py` — `get_emotion_model(settings)`, selects
  `deepface` (default) or `mock` via `emotion_model_backend`.
- `app/schemas/feature_extraction.py` — `FeatureExtractionResult`
  (`vector` + named `features` dict + `metadata`). `FeatureExtractor.extract()`
  now returns this instead of a bare `FeatureVector` — see
  `app/features/base.py` docstring for why this was safe to widen now.
- `app/schemas/feature_vector.py` — extended (additively) with
  `FEATURE_INDICES`, `FEATURE_COUNT`, `FeatureMeta`, `FEATURE_METADATA`;
  `FeatureVector` now also rejects non-finite values.

### The 22 features, exactly

| # | Feature | Source | Definition | Expected range |
|---|---|---|---|---|
| 0-6 | `emotion_angry`...`emotion_neutral` | DeepFace | Per-class probability (DeepFace % ÷ 100), reordered to canonical label order | 0-1 |
| 7-16 | `lbp_bin_0`...`lbp_bin_9` | Phase 3 LBP representation | Normalized histogram of uniform-LBP codes (P=8 ⇒ exactly 10 codes, one-to-one into 10 bins — no truncation) | 0-1, sums to ~1 |
| 17 | `edge_density` | Phase 3 Canny representation | `count_nonzero(canny) / canny.size` | 0-1 |
| 18 | `gradient_energy` | Grayscale (Sobel recomputed) | `mean(Gx² + Gy²)` on grayscale normalized to [0,1] — see `image_quality.py` docstring for why this recomputes rather than reusing the Phase 3 Sobel representation | small positive float, not bounded to [0,1] |
| 19 | `brightness` | Grayscale | `mean(grayscale)` | 0-255 |
| 20 | `contrast` | Grayscale | `std(grayscale)` | ≥ 0 |
| 21 | `sharpness` | Grayscale | `var(Laplacian(grayscale))` | ≥ 0 |

Raw features only — no scaler is fit here (that's Phase 6, on the
training split only, to avoid leakage).

New config: `emotion_model_backend` already existed (default `deepface`,
now also accepts `mock`); no other Phase 4 config was needed since LBP
and Sobel parameters were already added in Phase 3.

### Phase 3 additions

- `app/representations/engine.py` — `RepresentationEngine`
  (`RepresentationGenerator` impl). One grayscale conversion is computed
  and reused by CLAHE/Canny/Sobel/LBP rather than recomputed per
  representation. Each of the five derived representations can be
  independently enabled/disabled via config (`enable_grayscale`,
  `enable_clahe`, `enable_canny`, `enable_sobel`, `enable_lbp`) — this is
  what Phase 9 ablation experiments will toggle. Invalid input
  (`None`, empty, wrong shape/dtype/channel count) raises
  `RepresentationError` instead of a raw cv2/skimage exception.
- `app/schemas/representations.py` — extended (additively) with
  `RepresentationMeta` (name, source, width, height, dtype, color_space,
  params) and `Representations.metadata` / `.enabled_names()`.
- `app/representations/visualization.py` — dev-only 2x3 grid builder +
  `show_representations()` (OpenCV window) + `save_representation_grid()`.
  No GUI dependency in the engine itself.
- `app/representations/export.py` — `export_representations()` saves
  each enabled representation as a PNG under `representation_debug_dir`
  (`outputs/representations/` by default). Explicit opt-in only; nothing
  auto-saves every frame.
- `scripts/test_representations.py` — manual webcam integration script
  (`python scripts/test_representations.py`); not part of the pytest
  suite since it needs a real camera and display.
- `tests/test_representation_engine.py` — unit tests against a small
  deterministic synthetic image (gradient + shapes), not a webcam or a
  real face; see the module docstring for why.

New config (`app/core/config.py`, all overridable via `.env`):
`enable_grayscale`, `enable_clahe`, `enable_canny`, `enable_sobel`,
`enable_lbp`, `canny_aperture_size`, `canny_l2_gradient`,
`sobel_kernel_size`, `sobel_scale`, `sobel_delta`, `lbp_method`,
`representation_debug_dir`. (`clahe_clip_limit`, `clahe_tile_grid_size`,
`canny_low_threshold`, `canny_high_threshold`, `lbp_radius`,
`lbp_n_points`, `lbp_num_bins` already existed from Phase 1/2 scaffolding.)

### Color space note (Phase 3)

The `"rgb"` representation is, honestly, still **BGR**-ordered — it is
the untouched face crop, consistent with the Phase 2 convention that
nothing in this backend silently converts to true RGB. Its metadata's
`color_space` field says `"BGR"` explicitly. A future consumer that
needs true RGB channel order (e.g. DeepFace in Phase 4) must convert at
its own boundary.

### Color space contract

Frames are BGR, `uint8`, shape `(H, W, 3)` end-to-end through camera
capture, detection, and cropping — this matches OpenCV's native format
and is documented on `Frame`/`Face`. Nothing in this backend silently
converts to RGB. Any future component that needs RGB (e.g. DeepFace)
must convert explicitly at its own boundary.

### Phase 2 additions

- `app/camera/opencv_camera.py` — `OpenCVCameraSource` (`CameraSource` impl):
  open/read/close around `cv2.VideoCapture`, raises `CameraError` if the
  camera can't be opened, returns `None` from `read()` on a failed frame
  instead of raising, usable as a context manager.
- `app/detection/opencv_detector.py` — `HaarCascadeFaceDetector`
  (`FaceDetector` impl) using OpenCV's bundled frontal-face cascade.
  Reports `confidence=None` for every detection — Haar cascades don't
  produce a calibrated confidence score, and the project requires not
  inventing one.
- `app/detection/bbox_utils.py` — clamps raw detector boxes to frame
  bounds, rejects degenerate/too-small boxes, applies optional padding
  (`face_padding_ratio`, default `0.0`).
- `app/detection/cropping.py` — crops a validated bbox out of a frame
  without resizing (real-time vs. FER2013 preprocessing may need
  different target sizes later — deliberately not decided here).
- `app/detection/selector.py` — pluggable primary-face selection:
  `largest` (default), `highest_confidence` (falls back to largest when
  no detector confidence is available), `center_most`.
- `app/detection/service.py` — `FaceDetectionService` orchestrates the
  above into one call: `Frame` in, validated `list[Face]` out with at
  most one `is_primary=True`.

New config (`app/core/config.py`, all overridable via `.env`):
`face_padding_ratio`, `min_face_width`, `min_face_height`,
`primary_face_strategy`, `haar_scale_factor`, `haar_min_neighbors`.

## Project structure

```
app/
├── api/              FastAPI routes + dependencies (only /health so far)
├── camera/           CameraSource interface            (impl: Phase 2)
├── detection/         FaceDetector interface             (impl: Phase 2)
├── preprocessing/     face crop preprocessing            (impl: Phase 2/3)
├── representations/   RepresentationEngine: RGB/Gray/CLAHE/Canny/Sobel/LBP (Phase 3)
├── features/          StandardFeatureExtractor: 22D vector    (impl: Phase 4)
├── emotion/           EmotionModel: DeepFace + mock providers (impl: Phase 4)
├── dataset/           FER2013Loader, splitter, FeatureDatasetBuilder, validation (impl: Phase 5)
├── fusion/            FusionModel interface (LogReg)         (impl: Phase 6)
├── prediction/        RealTimePredictor, model loader, PredictionResult (impl: Phase 7)
├── temporal/          TemporalSmoother interface             (impl: Phase 8)
├── quality/           QualityAnalyzer interface               (impl: Phase 8)
├── models/            saved model artifacts live here (gitignored)
├── core/              config.py, logging.py, exceptions.py
├── schemas/           Frame, Face, Representations, FeatureVector, Prediction, dataset (Phase 5)
└── main.py            FastAPI app factory

tests/        pytest suite
configs/      reserved for non-.env config files (later phases)
dataset/      FER2013 goes here (not included in repo)
features/     generated feature-dataset CSVs (Phase 5): raw/, metadata/
experiments/  ablation run outputs (Phase 9)
outputs/      logs, dataset_failures.csv, dataset_summary.txt, misc run artifacts
scripts/      one-off utility scripts (incl. Phase 5's build/inspect scripts)
```

## The 10-phase roadmap

1. Architecture foundation
2. Camera + face detection
3. Multi-representation DIP pipeline (grayscale, CLAHE, Canny, Sobel, LBP)
4. 22-dimensional feature extraction
5. FER2013 dataset → feature-dataset generation
6. Fusion model training (Logistic Regression)
7. Real-time prediction engine ← you are here
8. Temporal smoothing + input quality control
9. Ablation experiments + evaluation
10. Production API + documentation, ready for frontend integration

Each phase's implementation goes behind the abstract interface defined
in Phase 1 (`FaceDetector`, `RepresentationGenerator`, `FeatureExtractor`,
`EmotionModel`, `FusionModel`, `TemporalSmoother`, `QualityAnalyzer`),
so no phase needs to restructure what an earlier phase built.

## The feature vector contract

The fusion model consumes a fixed 22-dimensional vector, defined once in
`app/schemas/feature_vector.py`:

```
0-6    7 deep-learning emotion probabilities (angry..neutral)
7-16   10 LBP histogram bins
17     edge density
18     gradient energy
19     brightness
20     contrast
21     sharpness
```

This ordering must not change after models are trained against it. See
the module docstring for details.

## Phase 3 → Phase 4 interface (as implemented)

Phase 4 consumes a `Representations` instance without needing to know
how it was produced:

**Input to Phase 4:** `Representations` (`app/schemas/representations.py`),
as returned by `RepresentationEngine.generate(face_crop)`:

- `rgb: np.ndarray` — BGR, uint8, shape `(H, W, 3)`, same `H, W` as the
  Phase 2 face crop that produced it. Fed to the emotion provider.
- `grayscale`, `canny`, `lbp: np.ndarray | None` — **required** by
  `StandardFeatureExtractor` (raises `FeatureExtractionError` if `None`).
- `clahe`, `sobel: np.ndarray | None` — not consumed by
  `StandardFeatureExtractor` (`gradient_energy` recomputes Sobel from
  `grayscale` directly instead — see `image_quality.py`); safe to
  disable for ablation without breaking Phase 4.
- `metadata` / `enabled_names()` — not currently read by
  `StandardFeatureExtractor`, available for debugging/Phase 9.

## Phase 4 → Phase 5 handoff contract

Phase 5 (FER2013 → feature dataset) must reuse the exact same objects,
not a parallel implementation:

```python
representations = representation_engine.generate(face_crop)   # Phase 3
result = feature_extractor.extract(representations)             # Phase 4
vector_22d = result.vector.values                                # -> save + label
```

- `feature_extractor` should be one shared `StandardFeatureExtractor`
  instance (constructed once with a real `DeepFaceEmotionModel`, via
  `app.emotion.factory.get_emotion_model()`), reused across every
  FER2013 image — not reconstructed per image, since DeepFace model
  loading is expensive (see `DeepFaceEmotionModel`'s docstring).
- `face_crop` for each FER2013 row should go through the same
  `RepresentationEngine` used at inference time; Phase 5 should not
  write its own resizing/preprocessing that diverges from Phase 2/3's
  conventions (see Phase 3 README notes on why 48×48 resizing wasn't
  hard-coded into Phase 3).
- `result.vector.values` (shape `(22,)`) is what gets written to
  `features/{train,validation,test}_features.csv` alongside the
  FER2013 label — `result.features` (the named dict) is convenient for
  a human-readable debug export but the model-facing artifact is the
  raw `(22,)` array in `FEATURE_NAMES` order.
- Phase 5 should expect `FeatureExtractionError`/`PredictionError` on a
  small fraction of rows (e.g. a face crop degenerate enough that a
  handcrafted feature can't be computed, or DeepFace failing on an
  unusual image) and must decide explicitly how to handle/skip/log
  those rows — Phase 4 will not silently invent a value to keep the
  pipeline running.
- No scaler is fit in Phase 4 or Phase 5; that happens in Phase 6, on
  the training split only.

## Phase 5 → Phase 6 handoff contract (as implemented)

Phase 6 reads the CSVs Phase 5 wrote — it does not call anything in
`app/dataset` directly:

```python
import pandas as pd
from app.schemas.feature_vector import FEATURE_NAMES

train = pd.read_csv("features/raw/train_features.csv")
val   = pd.read_csv("features/raw/validation_features.csv")
test  = pd.read_csv("features/raw/test_features.csv")

X_train, y_train = train[FEATURE_NAMES].to_numpy(), train["emotion"]
X_val,   y_val   = val[FEATURE_NAMES].to_numpy(),   val["emotion"]
X_test,  y_test  = test[FEATURE_NAMES].to_numpy(),  test["emotion"]
```

- Columns are exactly `sample_id, split, emotion, emotion_id,` then the
  22 `FEATURE_NAMES` in their fixed order (`app.dataset.builder.
  CSV_FIELDNAMES`) — Phase 6 should not re-derive or reorder them.
- `emotion` is the canonical lowercase label string; `emotion_id` is its
  index into `CANONICAL_EMOTION_LABELS` — either is usable as `y`.
- Fit `StandardScaler` on `X_train` only, then `.transform()` on
  validation/test — never re-fit or fit on combined data (Phase 5 spec
  sec. 31, already enforced by Phase 5 not touching a scaler at all).
- `features/metadata/dataset_metadata.json` records the split strategy,
  seed, and representation/feature config this run actually used —
  Phase 6 should reference it in the eventual report rather than
  re-guessing those values.
- `outputs/dataset_failures.csv` lists every sample Phase 5 could not
  featurize; Phase 6 does not need to handle these; they are already
  absent from the feature CSVs.
- Run `python scripts/inspect_dataset.py` to confirm split counts, class
  distribution, and the NaN/Inf/overlap checks before training.

## Phase 6 — Fusion Model Training (status: pipeline complete, not yet run on real data)

`app/fusion/`, `scripts/train_fusion_model.py`, `scripts/evaluate_fusion_model.py`,
and `tests/test_fusion_*.py` implement the full StandardScaler + Logistic
Regression training/evaluation pipeline described in the Phase 6 spec.

**This has not been run against real FER2013 data yet**, because
`features/raw/{train,validation,test}_features.csv` do not exist in this
checkout — Phase 5's builder script has not been run against a real
FER2013 file here. `scripts/train_fusion_model.py` detects this and exits
with a clear error rather than fabricating results (spec sec. 43). To
produce a real trained model:

```bash
python scripts/build_fer2013_features.py --dataset-path <path-to-fer2013.csv>
python scripts/train_fusion_model.py
```

What has been verified without real data:
- All 30 unit tests in `tests/test_fusion_*.py` pass (`pytest tests/test_fusion_*.py`),
  covering: canonical feature-order enforcement on load, label→emotion_id
  mapping, rejection of malformed/leaky datasets, `classes_` ordering
  ([0..6] in `CANONICAL_EMOTION_LABELS` order — not sklearn's default
  alphabetical string sort), the full select→fit→evaluate flow, model
  save/load round-tripping, and `load()` rejecting a model with the
  wrong feature or class count.
- A full end-to-end dry run (`train_fusion_model.py` against synthetic,
  clearly-separable placeholder feature vectors, not FER2013 data)
  confirmed the CLI produces all expected artifacts without error. Those
  synthetic artifacts were deleted afterward and are not part of this
  checkout — nothing in `models/` or `experiments/fusion/` should be
  treated as a real result until the script has been run against the
  real dataset.

## Phase 6 → Phase 7 handoff contract

```python
from app.fusion.model import LogisticRegressionFusionModel

model = LogisticRegressionFusionModel()
model.load(str(settings.fusion_model_path), str(settings.fusion_scaler_path))
prediction = model.predict(feature_vector)  # feature_vector: app.schemas.feature_vector.FeatureVector
# prediction.emotion, prediction.confidence, prediction.probabilities
```

- `load()` validates the artifact expects exactly 22 features and exactly
  7 classes before accepting it — a schema mismatch raises
  `ModelLoadError` rather than silently truncating/reshaping input.
- `predict()` always applies the same fitted `StandardScaler` used at
  training time, because scaler + classifier are one `Pipeline` object —
  Phase 7 never needs to load or apply a scaler itself.
- `confidence` is the raw maximum class probability from Logistic
  Regression — not a calibrated probability (spec sec. 30-31).
- `models/fusion_model_metadata.json` records the exact hyperparameters,
  dataset sizes, and `model_version` used for whatever model is currently
  saved at `models/fusion_model.pkl` — Phase 7 (and the eventual API
  layer) should surface this alongside predictions for traceability.

## Phase 7 — Real-Time Prediction Engine (status: implemented, not yet run against a real trained model/webcam here)

`app/prediction/` implements the frame-level real-time inference engine
described in the Phase 7 spec, reusing every Phase 1-6 component
unchanged (no duplicate detector/representation/feature-extraction
logic):

- `app/prediction/predictor.py` — `RealTimePredictor`: orchestrates
  `Frame → FaceDetectionService → RepresentationEngine →
  StandardFeatureExtractor → LogisticRegressionFusionModel →
  PredictionResult`. All heavy resources (detector, emotion model,
  fusion model) are constructed once in `__init__` and reused for every
  `predict(frame)` call. No face → `prediction_available=False` and no
  invented emotion. A recoverable per-frame failure (bad representation,
  failed feature extraction, model-inference failure, invalid
  probabilities) is logged and returned as a structured unavailable
  result with a specific `error_code` — the loop keeps running. A fatal
  initialization failure (missing/incompatible model artifact) raises
  `ModelLoadError` out of `__init__` and is never silently worked
  around. Frame-level only: no temporal smoothing, no quality gating
  (Phase 8), no REST/WebSocket API (Phase 10).
- `app/prediction/loader.py` — `load_fusion_model()`: loads the Phase 6
  artifact via `LogisticRegressionFusionModel.load()` (which already
  enforces the 22-feature / 7-class contract against the actual fitted
  estimator), and additionally cross-checks `fusion_model_metadata.json`
  (`feature_count`, `classes`, `feature_schema_version`) against the
  feature schema the running code uses, raising `ModelLoadError` on any
  mismatch rather than silently loading a model trained against a
  different definition.
- `app/prediction/schemas.py` — `PredictionResult` / `TimingBreakdown`:
  the structured, serializable result every `predict()` call returns —
  `face_detected`, `prediction_available`, `number_of_faces`,
  `primary_face_bbox`, `predicted_emotion`, `confidence`,
  `probabilities`, per-stage timing, `error_code`, and
  `model_version`/`feature_schema_version` for traceability.
  `feature_vector` is only populated when the predictor is constructed
  with `include_feature_vector=True` (default `False`) — kept out of
  the default result since a future API should not transmit the raw 22D
  array unless something needs it. `to_api_dict()` sketches the shape a
  future Phase 10 endpoint would expose.
- `scripts/run_realtime_prediction.py` — the Phase 7 CLI/demo:
  `python scripts/run_realtime_prediction.py` opens the webcam and
  overlays the face bbox, predicted emotion, confidence, FPS, and
  latency on the video feed (`q` to quit); `--show-representations`
  additionally shows the live Phase 3 representation grid;
  `--benchmark N` runs headless, measures actual per-frame and
  per-stage latency/FPS over `N` real captured frames, and writes the
  measured (never fabricated) results to
  `experiments/phase7_realtime_benchmark.json`.
- `tests/test_prediction_loader.py` — model-loading/metadata schema
  compatibility tests (spec sec. 35 tests 1-2), on top of the
  pipeline-level checks already covered by `tests/test_fusion_model.py`.
- `tests/test_predictor.py` — the full mocked pipeline test set (spec
  sec. 35 tests 3, 5-9): deterministic prediction from a known
  probability vector, no-face handling, multi-face primary selection
  (reusing the real `FaceDetectionService`/selector, not a test-only
  reimplementation), representation/feature-extraction/model-inference
  failure handling, invalid-probability rejection, and an end-to-end
  mocked run with no webcam required.

**Verified in this sandbox:** all of the above tests, plus a real
(non-mocked) integration run — real `HaarCascadeFaceDetector`, real
`RepresentationEngine`, real `StandardFeatureExtractor` (with the
`mock` emotion backend — no network access here to install
`deepface`/TensorFlow, same limitation already documented for Phase 4),
a real `LogisticRegressionFusionModel` artifact, and a real face image
(`skimage.data.astronaut()`, no webcam in this sandbox) — correctly
detected the face, ran the complete pipeline, and returned a structured
`PredictionResult` with all four timing stages populated. A blank/
non-face image correctly produced `face_detected=False,
prediction_available=False` with no invented emotion.

**Not yet run here:** the actual `pytest` suite (this sandbox has no
network access and does not have `pydantic`/`pydantic-settings`/
`fastapi`/`pytest` installed — the same limitation already documented
for Phases 1-6); a real webcam smoke test
(`python scripts/run_realtime_prediction.py`); a real benchmark
(`--benchmark N`); and everything above was exercised against a
synthetic, clearly-not-trained fusion model (no real FER2013-trained
`models/fusion_model.pkl` exists in this checkout — see the Phase 6
section above), so no FPS/latency/accuracy numbers from this sandbox
should be treated as representative. Before trusting this on the real
project: `pip install -r requirements.txt`, produce a real trained
model per the Phase 5/6 instructions above, then `pytest`, then
`python scripts/run_realtime_prediction.py`.

New config: none — Phase 7 reuses `fusion_model_path`, `fusion_scaler_path`,
`camera_*`, and every Phase 2-6 setting already in `app/core/config.py`
without adding new tunables.

## Phase 7 → Phase 8 handoff contract

Phase 8 (temporal smoothing + quality control) consumes the raw,
frame-level `PredictionResult` produced here — it does not call the
detector/representation/feature/fusion layers directly:

```python
from app.prediction.predictor import RealTimePredictor

predictor = RealTimePredictor()  # loads/validates the fusion model once
result = predictor.predict(frame)  # per frame, from Phase 8's own camera loop

# Phase 8 will feed a sequence of these into a TemporalSmoother:
result.face_detected        # bool
result.prediction_available # bool -- Phase 8 should skip/hold on False, not invent a value
result.predicted_emotion    # str | None
result.confidence           # float | None
result.probabilities        # dict[str, float] | None -- smooth these, not just the label
result.primary_face_bbox    # for a QualityAnalyzer to read face size from
result.timing / .processing_time_ms  # already-measured latency, reusable in Phase 8 reporting
```

- Phase 8 should construct exactly one `RealTimePredictor` (like Phase
  7's own CLI does) and call `.predict(frame)` per frame — never
  reconstruct the detector/representation engine/fusion model itself.
- A frame with `prediction_available=False` (no face, or any recoverable
  failure) must not be treated as a data point to smooth over as if it
  were a real prediction — Phase 8's moving-average/EMA window should
  either skip it or explicitly hold the previous stable state, per
  whatever policy Phase 8 defines.
- `error_code` on an unavailable result (see
  `app.prediction.predictor.ERROR_*` constants) lets Phase 8 (and later
  Phase 9's evaluation) distinguish *why* a frame had no prediction
  without re-deriving that from logs.

## Phase 8 — Temporal Analysis and Quality Control (status: implemented, not yet run against a real trained model/webcam here)

`app/temporal/` and `app/quality/` implement the post-processing layer
described in the Phase 8 spec, sitting directly on top of Phase 7's
`RealTimePredictor` without modifying it, retraining anything, or
duplicating the detection/representation/feature/model pipeline:

```text
Frame
  -> RealTimePredictor.predict()        (Phase 7, unchanged)
  -> QualityAnalyzer.analyze()          (this phase)
  -> valid prediction?                  (spec sec 21-23)
  -> TemporalSmoother.update()          (only valid predictions enter history)
  -> TemporalPredictionResult           (raw AND smoothed, never just one)
```

- `app/temporal/engine.py` — `TemporalPredictionEngine`: the single
  orchestrator for this phase. Constructs exactly one
  `RealTimePredictor` internally (with `include_feature_vector=True` so
  the quality analyzer can reuse its brightness/contrast/sharpness
  instead of recomputing them — spec sec 15-17) and calls
  `.predict(frame)` per frame, exactly like Phase 7's own CLI. Also
  accepts an externally-supplied predictor/analyzer/smoother for
  testing or for Phase 9's ablation engine.
- `app/temporal/base.py` — `TemporalSmoother` interface: smooths a
  stream of 7-class **probability dicts**, never labels (spec sec 7).
  `app/temporal/moving_average.py` — `MovingAverageSmoother`: the
  primary/default method, a bounded `deque(maxlen=window_size)` with no
  artificial startup delay (spec sec 9). `app/temporal/ema.py` —
  `EMASmoother`: optional `S_t = α·P_t + (1-α)·S_{t-1}`.
  `app/temporal/no_smoothing.py` — `NoOpSmoother`: pure passthrough,
  used when `temporal_enabled=False` or `smoothing_method="none"` so
  Phase 9 can flip smoothing off through the same code path (spec sec
  50-51). `app/temporal/factory.py` selects between them from
  configuration.
- `app/temporal/continuity.py` — a bounding-box IoU check (**not** face
  recognition/tracking — spec sec 12) used only to decide whether a
  sudden jump in the primary face's bbox should reset temporal history
  because a different subject likely took over as primary.
- `app/quality/analyzer.py` — `StandardQualityAnalyzer`: face-size
  ratio, brightness, contrast, and sharpness, each checked against a
  configurable threshold (`app.schemas.quality.QualityWarning`
  constants). `is_acceptable` is a separate *policy* decision from the
  warnings themselves (spec sec 20): only `FACE_TOO_SMALL`/`NO_FACE`
  are ever "severe" enough to reject a prediction, and only in
  `quality_mode="strict"` — `"advisory"` mode (the default) always
  keeps warnings informational, per spec sec 21/17 ("do not
  automatically reject ... just because contrast is low").
  `quality_score`, when computed, is an explicit engineering heuristic
  (spec sec 19), never presented as a calibrated confidence measure.
- `app/schemas/temporal.py` — `TemporalPredictionResult`: exposes
  `raw_emotion/raw_confidence/raw_probabilities` **and**
  `smoothed_emotion/smoothed_confidence/smoothed_probabilities`
  side by side (spec sec 25-26) so Phase 9 can evaluate exactly what
  smoothing changed, plus `history_length` and the embedded
  `QualityResult`.
- Missing-face reset (spec sec 10-11): after
  `max_missing_face_frames` consecutive no-face frames, temporal
  history is cleared so a stale emotion is never displayed
  indefinitely. Face-continuity reset (spec sec 12): a primary-face
  bbox jump below `face_continuity_iou_threshold` IoU triggers the
  same reset. Both are also available manually via
  `TemporalPredictionEngine.reset()` (spec sec 33).
- Invalid frames (no face, a Phase 7 failure, or — in strict mode only
  — a severe quality failure) never enter temporal history (spec sec
  22-23); the smoothed output holds at the last known stable state
  instead of going blank.
- Jitter/stability counters (`TemporalPredictionEngine.stability_stats`)
  count actual observed raw/smoothed label transitions (spec sec
  28-29) — stability, never accuracy, and never fabricated.
- `scripts/run_temporal_prediction.py` — the Phase 8 CLI demo (webcam
  or `--benchmark N`), analogous to Phase 7's own script, displaying
  raw emotion/confidence, stable emotion/confidence, quality status,
  history length, and FPS (spec sec 36-37). Does not modify or
  duplicate `scripts/run_realtime_prediction.py`.
- Configuration: `temporal_enabled`, `smoothing_method`,
  `smoothing_window_size`, `ema_alpha`, `max_missing_face_frames`,
  `face_continuity_iou_threshold`, `quality_enabled`, `quality_mode`,
  `min_face_area_ratio`, `min_brightness`, `max_brightness`,
  `min_sharpness`, `min_contrast` — all explicitly documented as
  initial engineering defaults (spec sec 31-32), never claimed to be
  scientifically optimized; that evaluation belongs to Phase 9.

**Tests included:** `tests/test_moving_average_smoother.py`,
`tests/test_ema_smoother.py`, `tests/test_no_smoothing.py`,
`tests/test_continuity.py`, `tests/test_temporal_factory.py`,
`tests/test_quality_analyzer.py`, `tests/test_temporal_engine.py` —
covering the deterministic moving-average/EMA math (spec sec 38-39),
reset (sec 33/40), the no-face timeout (sec 41), quality-warning
generation and the advisory/strict policy split (sec 42-43), invalid
predictions never entering history (sec 44), raw-vs-smoothed
divergence (sec 45), and primary-face-jump continuity reset (sec 46),
all against a fully mocked Phase 7 predictor (sec 47).

**What was actually verified in this sandbox:** this sandbox has no
network access and no `pydantic`/`pydantic-settings`/`pytest`, so the
real `pytest` run could not happen here. Instead the 7 Phase 8 test
modules were executed through a throwaway local shim (minimal stand-ins
for `pydantic`, `pydantic_settings`, and `pytest.raises/approx`, not
shipped): **45 tests passed, 0 failed**. Fixture-free Phase 1-7 modules
(`test_config`, `test_predictor`, `test_feature_vector_contract`,
`test_emotion_validation`, `test_bbox_utils`, `test_face_selector`,
`test_fusion_metrics`) also still passed under the same shim; modules
needing `tmp_path`/fixtures were not runnable there. This is weaker
evidence than a real `pytest` run in your environment — please run it.
**Not run:** real `pytest`, a webcam smoke test, a benchmark, and
anything against a real FER2013-trained `models/fusion_model.pkl`, so no
FPS/latency/stability numbers exist yet. Before trusting this:
`pip install -r requirements.txt`, then `pytest`, then
`python scripts/run_temporal_prediction.py`.

New config: `temporal_enabled`, `smoothing_method`,
`smoothing_window_size`, `ema_alpha`, `max_missing_face_frames`,
`face_continuity_iou_threshold`, `quality_enabled`, `quality_mode`,
`min_face_area_ratio`, `min_brightness`, `max_brightness`,
`min_sharpness`, `min_contrast` (all in `app/core/config.py` and
`.env.example`, replacing the earlier Phase 8 placeholder fields
`temporal_window_size`/`min_face_size_px`/`quality_confidence_threshold`,
which were never used anywhere else in the codebase).

## Phase 8 → Phase 9 handoff contract

Phase 9 (ablation testing + evaluation) consumes
`TemporalPredictionResult` from `TemporalPredictionEngine.process()`,
plus `TemporalPredictionEngine.stability_stats`:

```python
from app.temporal.engine import TemporalPredictionEngine

engine = TemporalPredictionEngine()  # constructs its own RealTimePredictor
result = engine.process(frame)       # per frame, from Phase 9's own camera/dataset loop

result.raw_emotion / .raw_confidence / .raw_probabilities        # Phase 7 output, untouched
result.smoothed_emotion / .smoothed_confidence / .smoothed_probabilities  # this phase's output
result.quality            # QualityResult: is_acceptable, warnings, metrics, quality_score, status
result.history_length     # size of the temporal window at this frame
engine.stability_stats    # {"raw_transitions": int, "smoothed_transitions": int}
```

- Phase 9's planned experiment comparisons (RGB baseline, RGB+CLAHE,
  handcrafted features, heuristic fusion, ML-based fusion) are all
  **upstream** of this phase and untouched by it — this phase only
  ever sees the fusion model's final `PredictionResult` and never
  reaches back into the representation/feature layers.
- To compare "no smoothing" vs "moving average" vs "EMA" (spec sec 50),
  construct separate `TemporalPredictionEngine` instances with
  `temporal_enabled=False` / `smoothing_method="moving_average"` /
  `smoothing_method="ema"` respectively (or pass an explicit
  `smoother=` override) and run the same frame sequence through each —
  never mutate one engine's settings mid-run.
- `stability_stats` gives actual observed transition counts for a
  jitter-reduction metric, but Phase 9 must still independently
  evaluate accuracy/precision/recall/F1 against ground-truth labels —
  a lower transition count is not by itself evidence of higher
  accuracy (spec sec 29/49).
- Test-set integrity (spec sec 32/52): any threshold tuning
  (`smoothing_window_size`, `ema_alpha`, the quality thresholds) must
  only ever use training/validation data; the held-out test set stays
  untouched until Phase 9's final evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
python run.py
# or
uvicorn app.main:app --reload
```

Then check `GET http://localhost:8000/health`.

## Test

```bash
pytest
```

## Notes for later phases

- Only add a dependency to `requirements.txt` when a phase actually uses
  it — Phase 1 intentionally does not include OpenCV, scikit-learn,
  DeepFace, etc.
- All new configuration values go in `app/core/config.py`, not scattered
  through modules.
- All new custom errors extend `BackendError` in `app/core/exceptions.py`.
- New API endpoints (`/predict`, `/config`, `/metrics`, `/representations`,
  `/settings`) go in `app/api/routes/` as new router modules, following
  the pattern in `health.py`, and get registered in `app/main.py`.


---

## Repaired backend integration

The backend now exposes a real FastAPI inference contract while keeping the
research pipeline modular.

### API

Start the server:

```bash
python run.py
```

or:

```bash
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health` — liveness; does not require the model.
- `GET /ready` — inference readiness; reports whether the Phase 6 model and
  scaler artifacts exist.
- `GET /status` — backend configuration, 22D feature schema, classes, model
  artifact status, and temporal configuration.
- `POST /predict/image` — send raw JPEG/PNG/WebP bytes as the request body.
- `POST /predict/base64` — JSON endpoint accepting a base64 image.
- `POST /predict/temporal/image` — frame prediction plus Phase 8 quality and
  temporal smoothing.
- `POST /predict/temporal/reset` — reset the process-local temporal state.

Example browser/client request:

```javascript
const response = await fetch("http://localhost:8000/predict/image", {
  method: "POST",
  headers: { "Content-Type": "image/jpeg" },
  body: imageBlob
});
const result = await response.json();
```

The API does not invent an emotion when no face is detected. If the required
Phase 6 model artifacts are missing, prediction requests return a clear
service-unavailable error; `/health`, `/ready`, and `/status` remain usable so
the frontend can discover the backend state.

### Research evaluation

Phase 9 infrastructure is now under `app/experiments/`.

Run readiness validation:

```bash
python scripts/validate_backend.py
```

Run real Phase 9 experiments only after the validated Phase 5 feature CSVs
exist:

```bash
python scripts/run_phase9.py --experiment all
```

The runner refuses to execute real experiments when the required files are
missing. Synthetic data is never substituted for project data.

Current valid feature-group ablations are based on the actual 22D schema:

- DeepFace probabilities
- LBP
- edge density
- gradient energy
- brightness/contrast/sharpness

There is currently no independently derived CLAHE or Sobel feature in the
22D vector, so the backend does **not** falsely claim separate CLAHE/Sobel
ablation results.

### Required artifacts for actual prediction/evaluation

The repository intentionally does not contain a fabricated model or
fabricated FER2013 results. Before live prediction is possible, provide:

```text
models/fusion_model.pkl
models/scaler.pkl
models/fusion_model_metadata.json
```

For Phase 9 evaluation, provide:

```text
features/raw/train_features.csv
features/raw/validation_features.csv
features/raw/test_features.csv
```

plus the Phase 5 metadata.

Once those real artifacts exist, the backend can run the same inference and
evaluation code without changing the architecture.


### Offline emotion-provider note
For this validated run, DeepFace could not be installed because the execution environment has no package-network access. To keep the project real-data-only, the 7 emotion-probability inputs were generated by the explicit `fer2013_linear` provider: a supervised LogisticRegression classifier trained on the FER2013 Training split. These probabilities must not be described as DeepFace outputs. The original `deepface` backend remains available when its package and model weights are installed.
