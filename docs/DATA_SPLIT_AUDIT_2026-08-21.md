# IRES-AI released random-split audit — 2026-08-21

## Finding

The released CSV ZIP is a long-form table with ten independent stratified 90/10 holdouts. It is
not an ordinary mutually exclusive ten-fold partition. The file has 467,740 rows representing
46,774 unique IDs/sequences repeated once for each released fold.

| Number of released test assignments per sequence | Unique sequences |
|---:|---:|
| 0 | 16,176 |
| 1 | 18,304 |
| 2 | 9,048 |
| 3 | 2,676 |
| 4 | 504 |
| 5 | 60 |
| 6 | 6 |

Dataset SHA-256:
`993b0db756efe05cc3598a8c28330d97cdcb6ab4fadad5194a7660b0a7d6f23b`.

## Project protocol

`src/ires_design/prediction.py` recovers one canonical record per sequence and preserves every
released test assignment as `published_test_folds`. It also constructs a deterministic,
mutually-exclusive ten-fold project partition by stable-hash round robin within label, source, and
coarse length strata. For test fold $k$, fold $(k+1)\bmod 10$ selects the classification threshold;
neither is used to fit the estimator.

This reconstructed partition supports fast leakage-aware pilots only. The primary paper result
still requires frozen sequence-cluster and viral-family splits. Published values must remain marked
as external references because their split and checkpoint-selection protocol differs.

## First reviewed pilot

Run: `runs/prediction_lightweight_reconstructed_nested10_v2_20260821_s42`

| Model | AUC | AUPR | F1 |
|---|---:|---:|---:|
| Composition logistic | 0.710 ± 0.009 | 0.405 ± 0.015 | 0.436 ± 0.016 |
| Hashed character 3–6-mer linear | 0.723 ± 0.011 | 0.452 ± 0.016 | 0.452 ± 0.013 |

These values establish shortcut-sensitive lower baselines; they are not cluster-split headline
results and do not demonstrate direct-RNA transfer.
