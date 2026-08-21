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
  papers/
    core/                 # ignored local PDF cache
    supplementary/       # ignored local supplementary data cache
```

Prefer environment variables described in `docs/BASELINES.md`. Never copy secrets, licensed raw data, model weights, or nested Git repositories into commits.

`assets/manifests/literature_sources.json` records URLs and hashes for reviewed papers without committing the PDFs themselves.
