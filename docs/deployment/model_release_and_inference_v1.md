# Fixed GSTRIDE model release and inference

This workflow separates the one-time historical model build from scoring new records. A released package stores the fitted imputer, scaler when applicable, estimator, validation-selected threshold, and an integrity manifest. New records are never used to fit, recalibrate, tune thresholds, or update performance metrics.

## Create the release once

From the repository root, run:

```powershell
D:\python-3.12.8\python.exe scripts\release_gstride_fall_model.py
```

The default release uses the random forest and fits it on all 163 approved historical GSTRIDE feature-table rows. The decision threshold is still selected once using the fixed validation partition before the full-history refit. The release is written to:

```text
artifacts/model_releases/gstride_fall_v1_random_forest_full_historical/
```

It contains `model.joblib` and `manifest.json`. The manifest records the model hash, exact data/configuration hashes, feature contract, fit scope, row count, and threshold-selection boundary. `model.joblib` is a Python serialization and must only be loaded from a trusted release directory.

Use `--fit-scope train` only when a release must preserve the original train-only fitting boundary. `--overwrite` deliberately replaces the release files. Releasing a new package is a controlled model-version change; it is not performed by the scoring command.

## Score one new record without training

Create a UTF-8 JSON file, for example `new_record.json`:

```json
{
  "participant_id": "user-001",
  "step_speed_m_s": 0.82,
  "cadence_strides_per_min": 52.0,
  "stride_length_m": 0.95,
  "double_support_pct": 24.0,
  "swing_to_stance_ratio": 0.31,
  "stride_time_cv_pct": 8.5
}
```

Then score it using only the release package:

```powershell
D:\python-3.12.8\python.exe scripts\score_gstride_fall_release.py `
  --release-dir artifacts\model_releases\gstride_fall_v1_random_forest_full_historical `
  --input-json new_record.json `
  --output reports\new_record_score.json
```

For batch scoring, provide `--input-csv path\to\records.csv` instead. A JSON input may also be an array of record objects. Scores contain the original model score, display percentage, fixed display band, technical predicted label, threshold, model name, and data version.

The six GSTRIDE fields must retain their documented definitions and units. In particular, `double_support_pct` and `stride_time_cv_pct` are 0--100 percentages. The output is a retrospective score for the public GSTRIDE faller label, not a future-fall probability, fracture risk, diagnosis, screening result, or treatment recommendation.
