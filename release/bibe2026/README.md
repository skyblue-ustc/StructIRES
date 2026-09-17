# BIBE 2026 reproducibility snapshot

This directory is the minimal, sequence-free evidence package for the
StructIRES manuscript. It is intended to let reviewers audit the reported
numbers without redistributing benchmark sequences or model weights.

## Contents

- `table1_per_fold_metrics.csv`: locked fold-level recognition results.
- `table1_aggregate_metrics.csv`: mean and sample standard deviation across folds 0--2.
- `candidate_selection_summary.csv`: aggregate results for the shared candidate pools.
- `parent_level_effects.csv`: paired StructIRES-Rank minus score-only effects across ten parents.
- `provenance.json`: hashes, protocols, exclusions, and known limitations.
- `../../configs/bibe2026_structires_adapter.json`: model and training contract.

Run `python scripts/verify_bibe2026_release.py` from the repository root to
recompute the recognition aggregates and validate the release contract.

## Boundaries

The recognition experiments use three fixed folds (seed 1337), with 6,315
evaluation sequences per fold (1,238 positive and 5,077 negative). AUROC and
AUPR use continuous scores. IRESfinder uses its fixed 0.5 threshold; learned
models use validation-selected thresholds for F1, MCC, and accuracy.

Candidate-selection results are computational diagnostics, not measurements
of translation activity. Raw candidates are intentionally excluded pending
disclosure review. The UTR-LM fold-2 run was stopped at epoch 18 and its best
epoch-14 metrics were recovered from the training log; no fold-2 checkpoint or
raw prediction file was retained. This limitation is recorded rather than
silently replacing the value.

## Not redistributed

Benchmark sequences, RNA-FM/UTR-LM/DeepIRES weights, trained checkpoints,
candidate sequences, W&B metadata, and cluster logs are excluded. Obtain
third-party assets from their cited sources and verify their licenses.
