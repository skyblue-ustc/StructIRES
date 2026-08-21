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
| R1 | RNA-FM native fold 0 | Upstream code/configuration, fold 0, 10 epochs, seed 1337 | 1 x A800, expected 1--3 h after weight retrieval | AUC within 0.03 and AUPR within 0.04 of released fold 0, or a documented explanation | Log, checkpoint, fold predictions, metrics, manifest |
| R2 | RNA-FM native folds 1--9 | Same released repeated-holdout protocol | Up to 3 independent A800 jobs | Run only after R1 passes | Per-fold and aggregate reproduction statistics |
| R3 | UTR-LM released-checkpoint rerun | Exact released fold checkpoints on author-native test folds | 1 x A800, complete | Ten folds match released CSVs within floating-point tolerance | Metrics, predictions, checkpoint hashes and run manifest |
| R4 | Corrected RNA-FM/UTR-LM | Validation-selected heads/checkpoints on the frozen 90% identity component split | 3 seeds; 1--3 A800 | No test access before final evaluation | AUC/AUPR/F1, calibration, CIs, predictions |
| R5 | IRES-specific predictors | DeepCIP, IRESpy, IRESfinder using released executables where licensing permits | CPU/GPU as needed | Native smoke result matches its release | Native and frozen-split tables |
| R6 | Assay transfer | Freeze each trained scorer, then evaluate direct-RNA labels with no retuning | CPU/GPU inference | R2/R3/R5 completed | All/non-control/high-similarity strata and calibration |

## Current allocation and download state

- Cluster node 56, physical GPU 3: available after the completed UTR-LM ten-fold inference.
- Nodes 46, 47 and 48: RNA-FM checkpoint download shards only; do not schedule GPU work because
  current utilization is high.
- Nodes 62 and 64: unavailable at the latest probe; recheck before allocating.
- Never start R2 full-fold scaling before R1 produces a valid checkpoint and prediction file.

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
| RNA-FM folds 1,3,4 | `downloading` | Active single-worker resumable job on node 47; fold 1 was 351,272,960/398,272,556 bytes. |
| RNA-FM folds 2,5,8 | `downloading` | Active single-worker resumable job on node 48; fold 2 was 373,817,344/398,272,556 bytes. |
| RNA-FM folds 6,7 | `downloading` | Active non-overlapping single-worker job on node 46; fold 6 resumes before fold 7 is fetched anew. |
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

Classification reproduction requires 22 weight files: one RNA-FM backbone, one UTR-LM backbone,
ten released IRES-RNAFM folds and ten released IRES-UTRLM folds. The UTR-LM set (11 files) is
complete and reproduced. The ten IRES-RNAFM folds and official RNA-FM backbone are resumable
downloads outside Git. Six generative IRES-DM checkpoints are intentionally excluded because they
are not needed for classification reproduction.

## Frozen comparisons

- Paper-native repeated 90/10 holdouts: reproduction only; preserve the upstream leakage warning.
- Exact 174-nt 90%-identity component split: 31,249 train, 6,696 validation and 6,696 test
  sequences, with zero threshold identity edges crossing splits.
- Source holdout: train/validate on `55k`, test all `IRESite_exp` records.
- Direct-RNA transfer: 2,179 labelled IRES-TrAPPr sequences; no test-side tuning.

Every completed run must include the dataset hash, source commit, checkpoint hash, exact command,
package versions, device identity, seed, predictions and metrics. Failed smoke runs remain recorded
and are never silently overwritten.
