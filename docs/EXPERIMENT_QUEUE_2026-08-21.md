# Classification reproduction queue

Last updated: 2026-08-22 (Asia/Shanghai)

This queue is restricted to reproducing published IRES classification benchmarks from public data.
It does not generate or optimize biological sequences and contains no wet-lab work.

## Reporting rule

Keep three result statuses separate:

1. `published_reference`: a value copied from the paper or upstream release;
2. `released_metrics_recomputed`: an aggregate recomputed from upstream prediction/metric files;
3. `checkpoint_rerun`: a model actually executed in our environment.

The released RNA-FM and UTR-LM scripts evaluate the test set every epoch and select the checkpoint
on test AUPR. Their native results therefore reproduce the paper protocol but are not independent
blind-test estimates. Corrected runs must use validation data for checkpoint and threshold selection.

## Execution order and gates

| Priority | Run | Protocol | Resource | Scale gate | Required output |
|---|---|---|---|---|---|
| R0 | Released IRES-LM audit | Recompute the 10 released RNA-FM/UTR-LM fold CSVs and audit source control flow | CPU, minutes | Complete | Fold metrics, means, hashes, selection audit |
| R1 | RNA-FM native fold 0 | Frozen released checkpoint, author-native test holdout, inference only | 1 x A800, complete | AUC within 0.03 and AUPR within 0.04 of released fold 0, or a documented explanation | Log, checkpoint hash, fold predictions, metrics, manifest |
| R2 | RNA-FM native folds 1--9 | Same frozen-checkpoint repeated-holdout protocol | 1 x unoccupied A800, complete | Run only after R1 passes | Per-fold and aggregate reproduction statistics |
| R3 | UTR-LM released-checkpoint rerun | Exact released fold checkpoints on author-native test folds | 1 x A800, complete | Ten folds match released CSVs within floating-point tolerance | Metrics, predictions, checkpoint hashes and run manifest |
| R4 | Corrected RNA-FM/UTR-LM | Validation-selected heads/checkpoints on the frozen 90% identity component split | 3 seeds; 1--3 A800 | No test access before final evaluation | AUC/AUPR/F1, calibration, CIs, predictions |
| R5 | IRES-specific predictors | DeepCIP, IRESpy, IRESfinder using released executables where licensing permits | CPU/GPU as needed | Native smoke result matches its release | Native and frozen-split tables |
| R6 | Assay transfer | Freeze each trained scorer, then evaluate direct-RNA labels with no retuning | CPU/GPU inference | R2/R3/R5 completed | All/non-control/high-similarity strata and calibration |

## Current allocation and download state

- RNA-FM downloads on nodes 46, 47 and 48 are complete and their background processes have ended.
- Node 56 physical GPU 2 was used only while it reported 4 MiB and no compute process. All P1
  inference jobs are complete and the GPU has been released.
- Nodes 62 and 64 were not used: node 62 had no fully empty GPU and node 64 did not answer the
  latest probe.
- Every R2 fold was started only after R1 passed and its checkpoint passed size, SHA-256 and
  `torch.load` checks.

### Handoff audit at 2026-08-22 02:44 CST

R1 passed using the released fold-0 checkpoint. The checkpoint rerun produced AUC 0.777114,
AUPR 0.594668 and F1 0.524311, so available-fold inference may proceed. The following list is a
point-in-time operational audit, not a claim that an incomplete checkpoint is usable:

| Item | Status | Evidence / next action |
|---|---|---|
| UTR-LM folds 0--9 | `completed` | All checkpoints rerun in `released_utrlm_native_10fold_v1_20260821`. |
| RNA-FM fold 0 | `completed` | `torch.load` valid; SHA-256 `3df27b...6fb2`; native rerun complete. |
| RNA-FM fold 9 | `completed` | `torch.load` valid; SHA-256 `962d58...5d2b`; native rerun completed on a subsequently released GPU 2 of node 56. |
| RNA-FM fold 7 old file | `failed_quarantined` | Correct size but `torch.load` reports a corrupt archive; preserved as `.corrupt_sha256_558517...`. |
| RNA-FM folds 0--9 | `completed` | All exact-size files load successfully and have per-fold native run directories. Fresh fold 7 SHA-256 is `012944...288`. |
| DeepCIP official inference | `environment_blocked` | Released extensions require Python 3.8; two isolated prefixes are empty and both cached PyTorch archives failed compression-integrity checks. |

The checkpoint inventory was checked by exact size, SHA-256 for complete files and `torch.load`,
not by filename alone. At the initial audit, every low-utilization GPU on nodes 46, 56 and 62 still
had another user's process, and node 64 was heavily utilized. GPU 2 of node 56 later became empty;
fold 9 ran there and released it after completion. No inference was placed on an occupied GPU. The
independent DeepCIP CPU dependency download is allowed to continue without blocking R2.

### Execution update at 2026-08-22 03:25 CST

- Native checkpoint reruns are complete for RNA-FM folds 0, 1, 2, 5, 6 and 9.
- `released_rnafm_native_available6_summary_v1_20260822` reports the six-fold interim
  AUC/AUPR/F1 as 0.777740$\pm$0.011009 / 0.611767$\pm$0.016782 /
  0.519049$\pm$0.023373. This is not a ten-fold result.
- Nodes 47, 48 and 46 continue the non-overlapping remaining downloads 3--4, 5 and 8, and 7,
  respectively.
- DeepCIP P2 is paused after a time-boxed exact-environment attempt: both cached PyTorch archives
  fail integrity checks and the independent replacement fetch exhausted proxy retries. The released
  data-overlap audit remains valid; official model inference is not marked reproduced.
- Six-checkpoint direct-RNA transfer is complete for the currently valid folds. The frozen
  arithmetic ensemble reached AUC 0.583 and AUPR 0.055; per-fold AUCs ranged from 0.251 to 0.641.
  No fold or direction is selected from direct-RNA labels. Repeat the same predeclared ensemble when
  all ten checkpoints are available.

### Completion update at 2026-08-22 05:20 CST

- R2 is complete. `released_rnafm_native_10fold_summary_v1_20260822` reports
  AUC/AUPR/F1 0.778420$\pm$0.009106 / 0.614020$\pm$0.013919 /
  0.509881$\pm$0.022686 across the ten released repeated holdouts.
- R6 is complete for the released RNA-FM ensemble. On all 2,179 labelled direct-RNA records, the
  ten-checkpoint arithmetic mean gives AUC 0.600 (95\% CI 0.567--0.632), AUPR 0.058
  (0.054--0.063), F1 0.097, specificity 0.001 and ECE 0.906. No direct-RNA label was used for
  checkpoint, direction or threshold selection.
- The frozen heterogeneity audit gives individual-checkpoint AUC 0.251--0.821 and pairwise
  Spearman correlation 0.338--0.705. The highest individual value is post hoc evidence of
  instability, not a selectable model result.
- IRESfinder native and similarity-aware outputs were upgraded to complete metric panels and saved
  predictions. UTR-LM full metrics were recomputed from its existing saved predictions without
  another model run.
- `paper/data/benchmark_results.csv` and `paper/tables/unified_results.tex` now distinguish
  published references, local official-split reruns, similarity-aware evaluation and direct-RNA
  transfer, with run and file hashes for every local row.

The final RNA-FM checkpoint inventory below is also embedded in
`released_rnafm_native_10fold_summary_v1_20260822/fold_metrics.csv`:

| Fold | Bytes | SHA-256 | Load/native run |
|---:|---:|---|---|
| 0 | 398,272,556 | `3df27bfcc9a09cc68316f82635bfa14b6a51132fe9a4738943181c2aa6fc6fb2` | valid/complete |
| 1 | 398,272,556 | `238545f10718e21ed7303e7dee1658c8d947e6a1f9dec76b3da35ea43cea21ad` | valid/complete |
| 2 | 398,272,556 | `278aa51d99bacd39195485f46a4381601e4d6e701bcfe0aa5c88a990a01c8734` | valid/complete |
| 3 | 398,272,556 | `d41f2c4d5de6a41046d06092e4809210cb417244f60457afab0278857d48dd30` | valid/complete |
| 4 | 398,272,556 | `8b1c4baa9dcafb79b1017e4c67abba32ffed64644164712e557bc1cfe1ad3d3d` | valid/complete |
| 5 | 398,272,556 | `43d34099447e099a924c6788f87952e2e7bb5441d1e2de32b5bc593d36c17bab` | valid/complete |
| 6 | 398,272,556 | `ca8ce17ff9c48645babca6010d0db8234ebb5833f7f8a3823fd8cd0b9be4dc29` | valid/complete |
| 7 | 398,272,556 | `01294403baa199ce209ea4c3872fa3ba6baaa252adf4ba56d715d0a5bcb68288` | fresh file valid/complete |
| 8 | 398,272,556 | `2259bb7ce9c54a38d148691934592184cc66794b98ac318916a3921025313acc` | valid/complete |
| 9 | 398,272,556 | `962d581998fb5bdc57f9742c0b4d02cee6b08ad0a09419c5bf9ad32d3dbd5d2b` | valid/complete |

The old fold-7 file with SHA-256 `558517...81a4` remains quarantined and excluded. There are no
active checkpoint downloads, no running P1 inference jobs and no pending RNA-FM classification
folds. The only failed/pending baseline item is exact DeepCIP inference under its Python 3.8 ABI.

## Draft-scope update on 2026-08-22 (CST)

The initial manuscript evidence chain is frozen in `docs/MAIN_TEXT_SCOPE_2026-08-22.md`.
No additional classification reruns are on the critical path to the first draft. The remaining
work is intentionally separated from completed evidence:

| Priority | Item | State | Gate / reason |
|---|---|---|---|
| P0 | Final manuscript visual/table QA | active | Rebuild only from frozen, provenance-backed outputs; no new result values. |
| P1 | IRES-EA exact released baseline | blocked | Versioned predictor wrappers and model directory are absent upstream; request/download the author assets before retrying. |
| P2 | DeepCIP official inference | time-boxed blocker | Requires a verified isolated legacy ABI and intact dependency archive; no approximation. |
| P3 | IRES-DM shared-protocol legacy task | pending / supplementary | Different 174-nt reporter task; run only after task data and selection protocol are frozen. |
| P4 | Project-owned trainable model | pending research phase | Requires a new, documented architecture and held-out evaluation; RFamLlama cannot be claimed as project pretraining. |

The frozen classification reruns require 21 local weight files: one UTR-LM backbone, ten released
IRES-RNAFM folds and ten released IRES-UTRLM folds. All are complete. A separate RNA-FM base
checkpoint is unnecessary because every fine-tuned fold contains the full backbone state. The
quarantined corrupt fold-7 file is retained outside Git for audit only. Six generative IRES-DM
checkpoints remain intentionally excluded because they are not needed for classification.

## Frozen comparisons

- Paper-native repeated 90/10 holdouts: reproduction only; preserve the upstream leakage warning.
- Exact 174-nt 90%-identity component split: 31,249 train, 6,696 validation and 6,696 test
  sequences, with zero threshold identity edges crossing splits.
- Source holdout: train/validate on `55k`, test all `IRESite_exp` records.
- Direct-RNA transfer: 2,179 labelled IRES-TrAPPr sequences; no test-side tuning.

Every completed run must include the dataset hash, source commit, checkpoint hash, exact command,
package versions, device identity, seed, predictions and metrics. Failed smoke runs remain recorded
and are never silently overwritten.
