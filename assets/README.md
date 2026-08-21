# Local assets

This directory is intentionally ignored except for small manifests.

Recommended local layout:

```text
assets/
  external/
    models/RFamLlama-base/
    models/GenerRNA/
    repos/IRES_Prediction_Design/
    data/iresbase/All_IRES.fa
    ires_dm/
  private/
    models/local_rfam_ar/
  manifests/
```

Prefer environment variables described in `docs/BASELINES.md`. Never copy secrets, licensed raw data, model weights, or nested Git repositories into commits.

