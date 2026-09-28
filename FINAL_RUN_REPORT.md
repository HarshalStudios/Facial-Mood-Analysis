# Final Real-Data Run Report

Date: 2026-09-28

## Dataset
- Dataset: FER2013
- File: `dataset/fer2013.csv`
- Total: 35,887
- Training: 28,709
- Validation/PublicTest: 3,589
- Test/PrivateTest: 3,589
- Image size: 48x48 grayscale
- Malformed pixel rows: 0
- Classes: angry, disgust, fear, happy, sad, surprise, neutral

## Emotion probability provider
DeepFace could not be installed in the offline execution environment. No fake DeepFace values were used.

The packaged run uses the explicit `fer2013_linear` provider: a supervised scikit-learn LogisticRegression classifier trained on the FER2013 Training split. It produces seven real class probabilities and is used as the first seven values of the 22D feature vector.

Validation of this provider before fusion:
- Accuracy: 0.3385
- Macro F1: 0.3018
- Probabilities finite and row-normalized.

These metrics describe the emotion-probability provider, not the final fusion model.

## 22D feature pipeline
1. Seven FER2013-trained emotion probabilities
2. Ten normalized uniform-LBP histogram bins
3. Edge density
4. Gradient energy
5. Brightness
6. Contrast
7. Sharpness / Laplacian variance

All feature CSVs contain 22 numeric feature columns, with 0 NaN/Inf values.

## Fusion model
- StandardScaler fitted on training data only
- LogisticRegression
- Candidate C values: 0.01, 0.1, 1, 10, 100
- Selected C: 0.1 using validation macro F1
- Test evaluated after selection

Validation:
- Accuracy: 0.3834
- Macro F1: 0.3198

Test:
- Accuracy: 0.3761
- Macro F1: 0.3076
- Weighted F1: 0.3597

## Phase 9
All seven registered experiments completed on real FER2013-derived feature data.

| Experiment | Test accuracy | Test macro F1 |
|---|---:|---:|
| Emotion-probability baseline | 0.3639 | 0.2972 |
| + LBP | 0.3711 | 0.3022 |
| + Edge | 0.3700 | 0.3027 |
| + Gradient | 0.3697 | 0.3053 |
| + Image Quality | 0.3703 | 0.3026 |
| + All Handcrafted | 0.3761 | 0.3076 |
| Handcrafted Only | 0.2803 | 0.1746 |

These are measured results, not estimates.

## Validation
- `pytest -q`: 276 passed, 1 skipped
- Python compileall: passed
- `/health`: 200
- `/ready`: 200, ready
- `/status`: 200, ready
- Real 48x48 FER2013 sample through representation + 22D extraction + trained fusion model: successful

## Important limitation
The packaged model does **not** claim to use DeepFace. DeepFace remains an optional backend in the source, but its package/model weights were unavailable in the offline environment. The current trained artifact therefore uses `fer2013_linear` and is documented accordingly.
