# Legacy asset audit and migration map

Audit source: the historical local directory `/9950backfile/lant/RFamLlama` on 2026-08-21. The source directory remains intact and is not the new Git repository.

## Storage summary

| Historical asset | Approximate size | Decision |
|---|---:|---|
| `checkpoints/` | 83 GB | Local only. Create manifests for selected checkpoints; never commit weights or optimizer state. |
| `external/` | 2.5 GB | Local cache only. Multiple duplicate clones exist; select one revision per upstream method and record it. |
| `wandb/` | 571 MB | Local provenance archive. Export only small run summaries needed for reproducibility. |
| `ires_rfamllama_ft_run1/` | 582 MB | Historical local IRES fine-tuning output; supplement only after audit. |
| `IRES_Prediction_Design/` | 286 MB | Nested upstream repository; do not vendor. Use `IRES_EA_ROOT` and commit hash. |
| `results/` | 106 MB | Historical evidence. Copy only frozen small tables after protocol review; never bulk-import. |
| `rfamllama/` | 27 MB | Old papers, patent/PPT material, and BIBM draft; documentation archive only. |
| `scripts/` | 1.1 MB | Mixed old MPRA/SFT/DPO and reusable evaluators; migrate file-by-file with tests. |
| `data/` | 2.1 MB plus external source files | Rebuild as manifest-driven raw/interim/processed data; do not assume current splits are valid. |

## Code classification

### Migrate first

| Historical code | New destination/use |
|---|---|
| `scripts/generate_external_rfamllama.py` and `rfam_eval_utils.py` | Official RFamLlama adapter after tokenizer/output tests. |
| `scripts/evaluate_external_ires_distribution.py` | External-reference evaluator, split into pure metrics and CLI. |
| `scripts/evaluate_public_ires_ai_kmer_classifier.py` | Locked public IRES evaluator with explicit train/test provenance. |
| `scripts/evaluate_mfe.py` | Structure feature module with ViennaRNA version capture. |
| `benchmark/metrics.py` | Pure sequence/distribution metrics after eliminating old Ridge coupling. |
| `scripts/train_ireslm_distilled_surrogate.py` | Optional diagnostic scorer only; never the sole judge. |
| `scripts/explore_mpra_data.py`, `audit_mpra_data.py`, `clean_mpra_data.py` | New immutable data-preparation pipeline and split manifest. |

### Preserve as historical evidence

- `scripts/prepare_mpra_sft_data.py`, `train_sft.py`, `generate_sft_seqs.py`.
- `scripts/prepare_dpo_data.py`, `train_dpo.py`, all reward-model scripts.
- `scripts/mpra_mainline_protocol.py` and old aggregation/report scripts.
- root `my_train.py`, `train.py`, and old shell launchers.

These files document earlier experiments but do not define the new mainline. They may be cited in internal notes or migrated into `legacy/` later; they should not be copied into the public repository without source/license review.

### Do not import

- raw checkpoints, optimizer states, WandB binary logs, caches, `__pycache__`, generated PDFs, copied papers, or nested `.git` directories;
- duplicated upstream repositories;
- absolute machine paths such as `/9950backfile/lant/...`;
- old result tables whose decoding protocols do not match.

## Current local migration bridge

Until assets are consolidated, use environment variables rather than symlinks or hard-coded paths. The new repository's `assets/` directory is ignored so that local model and data layouts can evolve without entering Git history.

