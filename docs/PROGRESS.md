# Project progress

Last updated: 2026-08-31 (Asia/Shanghai)

## Current phase

**P3/P4 — result consolidation and manuscript freeze.** P1 is complete: all ten released RNA-FM
checkpoints passed size, SHA-256 and `torch.load` checks and were evaluated separately on their
official test holdouts. The final native summary and ten-checkpoint direct-RNA transfer are frozen.
DeepCIP official inference remains a time-boxed environment blocker rather than a claimed result.

### Active submission-strengthening runs (2026-08-31)

- [~] The validation-clean, author-style `sequence` versus MFE-contact `contact` pair is
  running for native fold 1 on node 56 GPUs 5/6.  Both arms use the immutable v2 protocol:
  public RNA-FM t12 initialization, the same source-fold train/validation/test records,
  seed 1337, 15% within-train validation, 10-epoch maximum budget, class CE*2 + masked-LM
  CE*1, and validation-only AUPR/F1 selection.
- [~] To reduce elapsed time without changing the pre-fixed protocol, fold 3 sequence-only was
  launched on node 56 GPU 4 (which had 55 GB free and zero compute utilisation); its matched
  contact arm remains in the original GPU-6 queue.  The sequence queue now waits for folds 1 and
  3 before fixed-order, fail-fast folds 5, 6, 7, 8 and 9; the contact queue retains folds 3, 5,
  6, 7, 8 and 9.  The fold list was fixed before observing fold-1 metrics; queues stop on their
  first execution error and do not retry or overwrite a run.
- [~] A one-shot finalizer waits for both fixed queues, then independently recomputes paired
  summaries for folds 1/3/5/6/7/8/9 and aggregates them with the existing verified folds 0/2/4.
  It targets `structires_native_authorstyle_batchshuffle_multifold_0_9_v2_20260831`; it uses
  `set -e`, refuses pre-existing outputs, and leaves a log rather than retrying on any error.
- [x] Replaced the BIBE Figure 1 overview with a three-panel vector figure that separates
  the actual MFE-contact gated classifier, the matched four-objective rank selection, and the
  assay-qualified direct-RNA/MPRA evidence boundary.  The BIBE caption and Discussion now
  scope the three-fold classifier result as a modest native-fold signal rather than a
  generalization or calibration claim.
- [x] Verified the current IEEE conference draft with Tectonic on 2026-08-31:
  `paper/bibe2026/build/main.pdf` compiles successfully as a seven-page PDF.  The project
  unit-test suite (`32` tests) also passes.  Remaining TeX output consists only of
  underfull-box layout warnings; it contains no unresolved citations, missing figures, or
  compilation errors.  The definitive classifier table remains pending the active ten-fold
  finalizer and must not be replaced with provisional validation metrics.

## Completed

- [x] Migrated the active work into an independent `ires-design` repository.
- [x] Preserved RFamLlama attribution and moved the local reproduction to supplementary status.
- [x] Downloaded and text-audited the core direct and adjacent method papers.
- [x] Inspected the IRES-AI supplement and local upstream implementation.
- [x] Identified the assay mismatch between the old 174-nt DNA/lentiviral library and direct-RNA IRES-TrAPPr.
- [x] Identified Delli-Ponti structure/MFE mutation as a mandatory baseline rather than a novelty claim.
- [x] Added IRES–cargo crosstalk as an evidence-backed context constraint.
- [x] Reclassified RFamLlama and GenerRNA as optional proposal backbones.
- [x] Completed an independent S3-held-out direct-RNA computational proxy evaluation of the first
  matched design selections: 222 held-out records, AUC 0.592 and AUPR 0.497; robust-full minus
  score-only was +0.0110 (95\% bootstrap CI 0.0051--0.0176) across 30 parent-by-run units.
- [x] Ran the released ten-checkpoint UTR-LM ensemble over all 15,360 frozen candidates and added
  a budget-matched external score-only arm. Its direct-RNA proxy score was 0.4223, statistically
  indistinguishable from local RNA-LM score-only; robust-full minus UTR-LM was +0.0110 (95\% CI
  0.0024--0.0208) across the same 30 paired units.
- [x] Froze direct baselines and the three-track task hierarchy in documentation/configuration.
- [x] Retrieved, field-audited and hashed all nine IRES-TrAPPr XLSX supplements plus the supplementary PDF.
- [x] Audited the released IRES-AI split: ten repeated 90/10 holdouts, not a mutually exclusive
  ten-fold partition.
- [x] Completed a manifest-backed ten-fold lightweight pilot on reconstructed, nested folds.
- [x] Created a venue-neutral manuscript scaffold, predeclared tables, and two vector figures.
- [x] Completed the frozen v2 lightweight cross-assay transfer audit with stratified-bootstrap
  intervals and manuscript Figure 3/table outputs.
- [x] Completed the 55k-to-`IRESite_exp` source-holdout shortcut controls with locked thresholds and
  stratified-bootstrap intervals.
- [x] Exhaustively enumerated the 174-nt $\geq$90\% Hamming-identity graph, froze a 70/15/15
  component-disjoint split with zero crossing edges, and ran three-seed lightweight baselines.
- [x] Recomputed the released RNA-FM and UTR-LM ten-fold metric CSVs and audited the upstream
  checkpoint-selection control flow (`published_ireslm_release_audit_v1_20260821`).
- [x] Confirmed that both released training scripts inspect test AUPR each epoch and use it for
  checkpoint selection; native values are therefore reproduction references, not blind-test estimates.
- [x] Downloaded the released UTR-LM backbone and all ten IRES-UTRLM fold checkpoints, then reran
  all 46,780 native test-fold predictions (`released_utrlm_native_10fold_v1_20260821`).
- [x] Reproduced the released IRES-UTRLM aggregate from checkpoints: AUC 0.764938, AUPR 0.580823,
  F1 0.492983; the largest fold-level difference from the released CSV is $3.74\times10^{-6}$.
- [x] Ported the released IRESfinder Python 2.7 orchestration to Python 3 while retaining the
  original Perl feature extractor, training data and selected feature indices. The author example
  reproduced all 14 labels with a maximum probability difference of 0.000190.
- [x] Scored all 46,774 unique released IRES-LM sequences with IRESfinder and evaluated the same
  ten repeated native test folds: AUC 0.605074, AUPR 0.273844 and F1 0.346292, reproducing the
  published rounded values 0.61/0.27/0.35.
- [x] Ran frozen IRESfinder inference on all 2,300 normalized direct-RNA IRES-TrAPPr sequences.
  On the 2,179 labelled records, AUC was 0.460 (95\% CI 0.422--0.496), AUPR 0.043 and F1 0.081,
  demonstrating failed zero-shot assay transfer.
- [x] Audited IRESfinder training-set contamination: 695/722 released training sequences exactly
  overlap the IRES-LM benchmark with 100\% label agreement. Removing them changes native repeated
  means from 0.605/0.274/0.346 to 0.602/0.267/0.337 (AUC/AUPR/F1).
- [x] Downloaded and verified the released RNA-FM fold-0 checkpoint, then completed frozen native
  inference: AUC 0.777114, AUPR 0.594668 and F1 0.524311 on 4,678 test records. This passed the
  scale gate before the remaining folds were run.
- [x] Ran frozen RNA-FM fold-0 transfer on all 2,179 labelled IRES-TrAPPr sequences. Ranking retained
  weak signal (AUC 0.618, 95\% CI 0.582--0.652), but AUPR was 0.061 and the released 0.5 threshold
  predicted virtually every record positive (specificity 0.003; ECE 0.925).
- [x] Audited the released DeepCIP data. Its fixed 1,164-sequence test set is disjoint from all three
  released training subsets, but 1,089/1,164 test sequences exactly occur in the IRES-LM 55k source.
  The two assay-derived binary labels agree for only 46.7\% of these identical sequences, directly
  demonstrating that sequence overlap does not make the legacy and circRNA-screen targets equivalent.
- [x] Completed the 2026-08-22 handoff audit of Git state, active download processes, checkpoint
  sizes/hashes/loadability, GPU occupancy and formal run provenance. Fold 9 is valid and ready;
  fold 7 was proven corrupt by `torch.load`, retained under a hash-labelled quarantine name and
  assigned a non-overlapping redownload job.
- [x] Added and smoke-tested a native RNA-FM fold summarizer that recomputes AUROC, AUPRC, F1,
  accuracy, sensitivity, specificity, MCC and 10-bin ECE from saved predictions, adds per-holdout
  stratified-bootstrap intervals and preserves the repeated-holdout non-independence warning.
- [x] Time-boxed the DeepCIP environment attempt without modifying an existing environment. Both
  isolated prefixes are empty; the cached CPU archive fails CRC validation, the CUDA archive is
  truncated, and a separate public CPU-package fetch exhausted its retries on proxy CONNECT
  timeouts without producing a usable partial. Official inference remains blocked rather than
  being approximated under an incompatible Python ABI.
- [x] Completed released RNA-FM native inference for folds 0, 1, 2, 5, 6 and 9. The manifest-backed
  available-fold summary is 0.777740$\pm$0.011009 / 0.611767$\pm$0.016782 /
  0.519049$\pm$0.023373 (AUC/AUPR/F1). This immutable 6/10 snapshot is retained only to document
  execution history; the final ten-fold aggregate supersedes it.
- [x] Evaluated the frozen six-checkpoint arithmetic ensemble on direct-RNA IRES-TrAPPr labels
  without tuning. Individual AUCs varied from 0.251 to 0.641 and pairwise Spearman correlations
  were 0.440--0.705; the ensemble yielded AUC 0.583 (95\% CI 0.549--0.616), AUPR 0.055 and ECE
  0.908. The preceding five-checkpoint subset yielded only 0.432/0.041, demonstrating unstable
  out-of-assay ranking while threshold transfer fails consistently.
- [x] Completed all ten released RNA-FM native checkpoint reruns. The final repeated-holdout
  summary (`released_rnafm_native_10fold_summary_v1_20260822`) is AUC
  0.778420$\pm$0.009106, AUPR 0.614020$\pm$0.013919 and F1
  0.509881$\pm$0.022686. Accuracy, sensitivity, specificity, MCC and ECE are also recomputed from
  saved predictions; the overlapping-holdout non-independence warning is retained.
- [x] Re-downloaded the previously corrupt fold-7 checkpoint without deleting the quarantined
  file. The fresh SHA-256 is `012944...288`, differs from the corrupt `558517...81a4`, and loads
  and evaluates successfully. All ten usable RNA-FM checkpoints are now complete.
- [x] Completed the predeclared ten-checkpoint arithmetic ensemble on direct-RNA labels with no
  tuning: AUC 0.600 (95\% CI 0.567--0.632), AUPR 0.058 (0.054--0.063), F1 0.097, specificity
  0.001 and ECE 0.906. Frozen individual-checkpoint AUCs span 0.251--0.821 and pairwise Spearman
  correlations span 0.338--0.705, so checkpoint selection on direct-RNA labels is prohibited.
- [x] Upgraded IRESfinder native and similarity-aware runs to preserve predictions, full metric
  panels and 2,000-replicate stratified-bootstrap intervals. The similarity-aware result is AUC
  0.599 (0.583--0.617), AUPR 0.199 (0.190--0.211), MCC 0.121 and ECE 0.521.
- [x] Added a provenance-checked unified result ledger. `paper/data/benchmark_results.csv` and
  `paper/tables/unified_results.tex` are generated from `configs/benchmark_ledger.json`; local
  rows include run IDs plus SHA-256 hashes of their metric files and run manifests.

## Frozen scientific story

> We study whether assay-aware, ensemble-structure- and cargo-context-constrained optimization can produce computationally robust variants of bona fide full-length viral IRESes, compared with IRES-specific, score-only, structure-only, LM-guided and random baselines under matched budgets.

The paper has three separate result blocks:

1. cross-assay failure/transfer audit;
2. primary full-length seeded IRES design;
3. secondary 174-nt legacy IRES-DM reproduction.

## First-round decision gate

The completed benchmark changes the next-stage emphasis without changing the project mainline:

1. **Do not claim novelty from swapping in another RNA foundation model.** RNA-FM already matches
   its native paper result, while direct-RNA performance varies much more across checkpoints than
   native performance does.
2. **Make assay-aware checkpoint robustness the scorer contribution.** The next corrected baseline
   should train or choose heads only on validation data under similarity-aware splits, then compare
   a single head, a fixed all-checkpoint ensemble, uncertainty-aware conservative aggregation and
   a small direct-RNA calibrated head under nested, group-aware evaluation.
3. **Carry uncertainty into the design objective.** Full-length IRES optimization should improve a
   conservative aggregate across valid scorers rather than exploit one legacy checkpoint; the
   planned structure-ensemble and cargo-context constraints remain the protection against
   function-score gaming.
4. **Use explicit kill criteria.** If a validation-clean, similarity-aware scorer cannot exceed
   composition/IRESfinder controls on AUPR and calibration, freeze the paper as a benchmark and
   assay-transfer study rather than presenting unsupported generated-sequence quality claims.

The primary experimental priority is the matched seeded-design matrix.  R4 (a validation-clean
direct-RNA scorer) remains an independent functional-evaluation track, not a reason to delay
structure/energy optimization.  During the first design pass, local RNA-LM likelihood is a
proposal-plausibility objective only and is never reported as measured IRES activity.

## Readiness matrix

| Component | Status | Exit condition |
|---|---|---|
| Core literature | Complete | Source/hash manifest committed. |
| IRES-AI upstream code | Located, partially audited | Exact environment, commit, data and commands recorded. |
| IRES-TrAPPr supplement | Retrieved and hashed | Normalize controls, duplicates and assay conditions into immutable tables. |
| Assay-labelled data schema | Planned | All sources normalized without erasing assay context. |
| Random split audit | Complete | Assignment distribution and dataset hash recorded. |
| Similarity/family split | 174-nt exact Hamming view complete | Add gapped/cross-length and family-held-out views. |
| Random mutation | Ready | Existing tests pass. |
| Random screen | Planned | Budget-matched smoke run. |
| Score-only GA | Planned | Paired initialization/query-count smoke run. |
| Structure-only mutation | Planned | Published rule reproduced on a small seed set. |
| IRES-EA | Upstream available | Adapter and lineage parser pass smoke test. |
| IRES-DM released parser | Planned | Released candidates normalized with provenance. |
| IRES-DM retraining | Deferred until data gate | Native metric reproduction before formal run. |
| ViennaRNA ensemble metrics | Implemented; environment ABI repair pending | Shared-pool folding script writes MFE and pairing-profile preservation. |
| Local RNA-AR frozen probe | Blocked by version audit | Restore the custom HoPE loader without ignored `bias_raw` weights. |
| Albatross audit | Data/source identified | Frozen subset executable with revision manifest. |

## Reviewed pilot runs

| Run ID | Protocol | Result | Interpretation |
|---|---|---|---|
| `prediction_lightweight_reconstructed_nested10_v2_20260821_s42` | 10 mutually exclusive project folds; next fold validates threshold | Composition: AUC 0.710±0.009, AUPR 0.405±0.015, F1 0.436±0.016 | Strong shortcut-control signal. |
| same run | same | Hashed 3–6-mer: AUC 0.723±0.011, AUPR 0.452±0.016, F1 0.452±0.013 | Required lower baseline; not yet a cluster-split result. |
| `cross_assay_transfer_lightweight_v2_20260821` | Train/threshold on legacy assay only; evaluate 2,179 direct-RNA labels | Composition: AUC 0.579 (95\% CI 0.535--0.624), AUPR 0.058 (0.052--0.069) | Weak transfer and severe miscalibration. |
| same run | same | Hashed 3–6-mer: AUC 0.368 (0.328--0.409), AUPR 0.037 (0.035--0.040) | Legacy sequence shortcuts reverse under assay transfer. |
| `prediction_source_holdout_55k_to_iresite_exp_v1_20260821_s42` | Train/validate on 55k only; test the full `IRESite_exp` source | Composition: AUC 0.600 (0.533--0.668), AUPR 0.694 (0.635--0.761) | Source holdout weakens discrimination and threshold transfer. |
| same run | same | Hashed 3–6-mer: AUC 0.601 (0.537--0.664), AUPR 0.681 (0.626--0.738) | Mixed-source random folds overstate portability. |
| `prediction_similarity_split_hamming90_len174_v1_20260821_s42_43_44` | Exact 90\% identity components; frozen 70/15/15 split | Composition: AUC 0.634 (0.616--0.650), AUPR 0.230 (0.217--0.248) | Stronger than k-mer after near-neighbour isolation. |
| same run | same | Hashed 3–6-mer: AUC 0.578 (0.559--0.595), AUPR 0.250 (0.229--0.273) | Large decline from reconstructed mixed-source folds. |
| `published_ireslm_release_audit_v1_20260821` | Recompute upstream released ten-fold CSV; no checkpoint rerun | RNA-FM: AUC 0.779, AUPR 0.615, F1 0.506; UTR-LM: 0.765/0.581/0.493 | Released aggregates reproduce exactly, but upstream selects checkpoints on test AUPR. |
| `released_utrlm_native_10fold_v1_20260821` | Frozen released checkpoints; author-native repeated test folds; inference only | AUC 0.764938, AUPR 0.580823, F1 0.492983 | Ten-fold checkpoint execution reproduces the release to floating-point tolerance. |
| `iresfinder_released_repeated10_v3_20260822` | Original released features/training data; Python 3 adapter; author-native repeated test folds | AUC 0.605074, AUPR 0.273844, F1 0.346292, MCC 0.109, ECE 0.345 | Full predictions and per-holdout bootstrap intervals now accompany the reproduced aggregate. |
| `iresfinder_similarity_split_hamming90_len174_v2_20260822` | Released IRESfinder; frozen exact 90\% identity component-disjoint test | AUC 0.599 (0.583--0.617), AUPR 0.199 (0.190--0.211), F1 0.313 | Full metric panel and predictions; performance remains weak. |
| `iresfinder_cross_assay_transfer_v1_20260821` | Released IRESfinder; zero-shot direct-RNA IRES-TrAPPr transfer | AUC 0.460 (0.422--0.496), AUPR 0.043 (0.041--0.047), F1 0.081 | Ranking is below random and calibration fails (ECE 0.518). |
| `iresfinder_training_overlap_audit_v1_20260822` | Exact released-training versus benchmark audit | 695/722 IRESfinder training sequences overlap; overlap-removed native AUC/AUPR/F1 0.602/0.267/0.337 | Contamination is real and disclosed, but does not explain cross-assay failure. |
| `released_rnafm_native_fold0_v1_20260822` | Frozen released RNA-FM checkpoint; native fold-0 test; inference only | AUC 0.777114, AUPR 0.594668, F1 0.524311 | Scale gate passes; public checkpoint rerun is at the released-performance scale. |
| `released_rnafm_native_fold9_v1_20260822` | Frozen released RNA-FM checkpoint; native fold-9 test; inference only | AUC 0.759165, AUPR 0.593812, F1 0.471766 | Second valid checkpoint rerun; checkpoint and predictions are hashed. |
| `released_rnafm_native_available2_summary_v1_20260822` | Folds 0 and 9 only; per-fold 2,000-replicate stratified bootstrap | AUC 0.768140$\pm$0.012692, AUPR 0.594240$\pm$0.000605, F1 0.498038$\pm$0.037155 | Historical partial summary; never substitute for the final 10-fold aggregate. |
| `released_rnafm_native_available5_summary_v1_20260822` | Folds 0, 1, 2, 6 and 9; per-fold 2,000-replicate stratified bootstrap | AUC 0.775705$\pm$0.010974, AUPR 0.610537$\pm$0.018457, F1 0.517090$\pm$0.025575 | Historical partial snapshot retained for provenance. |
| `released_rnafm_cross_assay_available5_v1_20260822` | Frozen folds 0, 1, 2, 6 and 9; arithmetic probability ensemble; no direct-label tuning | AUC 0.432 (0.396--0.466), AUPR 0.041 (0.039--0.044), F1 0.097 | Native performance does not imply stable assay transfer. |
| `released_rnafm_cross_assay_available5_heterogeneity_audit_v1_20260822` | Post hoc per-fold evaluation and pairwise prediction correlation; no selection | Per-fold AUC 0.251--0.618; Spearman 0.443--0.672 | Fold 0 alone materially overstates transfer. |
| `released_rnafm_native_available6_summary_v1_20260822` | Folds 0, 1, 2, 5, 6 and 9; per-fold 2,000-replicate stratified bootstrap | AUC 0.777740$\pm$0.011009, AUPR 0.611767$\pm$0.016782, F1 0.519049$\pm$0.023373 | Historical partial snapshot retained for provenance. |
| `released_rnafm_cross_assay_available6_v1_20260822` | Frozen six-fold arithmetic probability ensemble; no direct-label tuning | AUC 0.583 (0.549--0.616), AUPR 0.055 (0.052--0.061), F1 0.097 | Ranking moves with checkpoint composition; calibration remains unusable. |
| `released_rnafm_cross_assay_available6_heterogeneity_audit_v1_20260822` | Post hoc per-fold evaluation and pairwise prediction correlation; no selection | Per-fold AUC 0.251--0.641; Spearman 0.440--0.705 | Stable native metrics coexist with unstable direct-RNA ranking. |
| `released_rnafm_cross_assay_fold0_v1_20260822` | Frozen RNA-FM fold 0; zero-shot direct-RNA transfer | AUC 0.618 (0.582--0.652), AUPR 0.061 (0.057--0.067), F1 0.097 | Rank signal survives weakly, but threshold transfer and calibration fail severely (ECE 0.925). |
| `released_rnafm_native_10fold_summary_v1_20260822` | All ten released checkpoints; official repeated holdouts; per-holdout bootstrap | AUC 0.778420$\pm$0.009106, AUPR 0.614020$\pm$0.013919, F1 0.509881$\pm$0.022686 | Final local checkpoint reproduction; upstream test-selected-checkpoint warning remains. |
| `released_rnafm_cross_assay_10fold_v1_20260822` | All ten released checkpoints; arithmetic ensemble; no direct-label tuning | AUC 0.600 (0.567--0.632), AUPR 0.058 (0.054--0.063), F1 0.097 | Weak ranking but unusable threshold transfer: specificity 0.001, ECE 0.906. |
| `released_rnafm_cross_assay_10fold_heterogeneity_audit_v1_20260822` | Frozen post hoc per-checkpoint transfer audit; no selection | Per-fold AUC 0.251--0.821; Spearman 0.338--0.705 | Native similarity does not imply stable cross-assay ranking. |
| `released_utrlm_native_10fold_summary_v1_20260822` | Full metrics recomputed from all saved UTR-LM predictions | AUC 0.764938$\pm$0.009281, AUPR 0.580823$\pm$0.016453, F1 0.492983$\pm$0.015945 | Adds sensitivity, specificity, MCC, ECE and per-holdout bootstrap intervals without rerunning the model. |
| `deepcip_data_overlap_audit_v3_20260822` | Exact released DeepCIP split and cross-benchmark audit | 1,089/1,164 DeepCIP test sequences match IRES-LM 55k; cross-assay label agreement 46.7\% | DeepCIP test is train-disjoint internally, yet label semantics are assay-specific on the same sequences. |

The attempted local 130M RNA-AR probe produced no accepted score. Standard Transformers ignored
HoPE `bias_raw` checkpoint weights; the process was stopped and the runner now fails closed when the
custom HoPE implementation is unavailable.

The first two RNA-FM fold-0 smoke attempts stopped before training: v1 exposed an unused upstream
`ptflops` import, and v2 exposed a retired upstream checkpoint URL returning HTTP 403. The v3 route
uses the official `cuhkaih/rnafm` Hugging Face release and requires SHA-256 verification before launch.

## Next actions

### 2026-08-22 to 2026-08-28

- [x] Retrieve IRES-TrAPPr Tables S1–S9 and verify sequence, condition, replicate and TE fields.
- [ ] Inventory exact files in the local IRES-AI repository and create a source/data manifest.
- [ ] Normalize old MPRA and direct-RNA records into an assay-labelled table.
- [ ] Build sequence-cluster and viral-family split candidates; check overlap/leakage.
- [x] Complete RNA-FM native fold-0 scale gate using the verified official checkpoint.
- [x] Resolve and execute the exact released UTR-LM backbone and all ten fold checkpoints.
- [x] Finish the ten RNA-FM fold checkpoint downloads and run the same frozen-checkpoint audit.
- [x] Rerun IRESfinder on all unique sequences and the released repeated test folds.

### 2026-08-29 to 2026-09-04

- [x] Run simple k-mer/GC/length baselines on the reconstructed legacy random split.
- [ ] Repeat the lightweight baselines on frozen similarity/family splits and within each assay.
- [x] Complete the ten-fold ensemble evaluation of legacy IRES-LM transfer without direct-label tuning.
- [ ] Produce the assay-shift gate report with calibration, rank correlation and top-k overlap.
- [ ] Freeze the full-length seed panel and cargo panel only after the report.

### Primary design pass (2026-08-22)

- [x] Repair the frozen seed panel's duplicate `F0` identifiers using sequence-ID suffixes.
- [x] Freeze three byte-identical shared mutation pools: 10 parents x 512 candidates x seeds 42/43/44 (15,360 candidates total).
- [x] Score all pools with the local frozen RNA-LM; outputs are explicitly proposal plausibility, not activity.
- [x] Implement matched random, score-only, structure-only and `robust_full` selectors over the same candidate pool.
- [x] Repair ViennaRNA through isolated PyPI wheel environment and fold all 15,360 candidates with ViennaRNA 2.7.2 ensemble metrics.
- [x] Run matched selectors and write the first primary design table from immutable run manifests.
- [ ] Add direct-RNA calibrated evaluation and cargo-context stress testing as independent follow-on evidence.
- [ ] Retrieve the public cargo-sequence source for the stress test: Chen et al. Table S1 is absent
  locally. Download target: `assets/papers/supplement/chen_2026_table_s1.xlsx`; official URL is
  `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41422-026-01233-9/MediaObjects/41422_2026_1233_MOESM2_ESM.xlsx`.
- [x] Resolve the public RNA-FM predictor backbone dependency and run a released-checkpoint
  compatibility audit. `RNA-FM_pretrained.pth` was retrieved from the official `cuhkaih/rnafm`
  Hugging Face release; public `RNAFM_Predictor.py`, with `token_dropout=true`, `batch_toks=4096`,
  and hash-based restoration of length-batched rows, exactly matches all 4,678 frozen fold-0
  predictions (max absolute difference $1.11\times10^{-16}$). This validates a compatibility
  implementation for the released classifier, not the unreleased versioned `vMay7_RNAFM_Predictor.py`
  or the original EA trajectory.

## Active risks

1. **Assay invalidity:** old high scores may mostly capture DNA-reporter artifacts. Mitigation: separate assay targets and make cross-assay transfer a result.
2. **Small RNA-active set:** 67 active Type IV/VI elements are insufficient for a new foundation model. Mitigation: frozen embeddings/shallow calibrated heads and mutation-focused design.
3. **No wet lab:** computational ranking cannot establish improved activity. Mitigation: conservative claims and a small, well-justified candidate shortlist for later validation.
4. **Structure-oracle limits:** ViennaRNA on long/circular contexts can disagree with in-cell structures. Mitigation: ensemble metrics, Albatross audit and cargo stress testing.
5. **Baseline fairness:** released IRES-DM output is not budget matched. Mitigation: keep it external and require a same-protocol retraining for strict claims.

## Decision log

| Date | Decision | Evidence |
|---|---|---|
| 2026-08-21 | Move 174-nt de novo from primary to legacy benchmark. | IRES-TrAPPr reports negligible direct-RNA activity for old short candidates and highlights length/assay limitations. |
| 2026-08-21 | Make full-length seeded viral IRES design primary. | IRES-TrAPPr mutational scans and Type IV/VI actives provide stronger functional anchors. |
| 2026-08-21 | Treat MFE/structure-only as a baseline. | Delli Ponti et al. already combine fold preservation, MFE and protein-interaction predictions. |
| 2026-08-21 | Require random screening and score-only GA. | PARADE shows strong short-UTR random-screen performance and illustrates predictor-optimization confounding. |
| 2026-08-21 | Add cargo-context robustness. | Experimental IRES–cargo work links crosstalk/structure consistency to restored circRNA translation. |
| 2026-08-21 | Do not treat the released random split as ordinary ten-fold CV. | Test membership repeats 0–6 times per unique sequence. |
| 2026-08-21 | Reject the first local RNA-AR probe. | Standard loader ignored custom HoPE `bias_raw` weights. |
| 2026-08-22 | Freeze the ten-checkpoint RNA-FM reproduction and transfer audit. | Native AUC is stable near 0.778, but direct-RNA per-fold AUC spans 0.251--0.821 and the final ensemble ECE is 0.906. |
| 2026-08-22 | Begin the primary matched seeded-design matrix without waiting for a legacy-classifier gate. | The design claim is computational robustness under identical mutation pools; LM likelihood is scoped to proposal plausibility and structural/energy metrics are the main first-pass endpoints. |
| 2026-08-22 | Complete first matched structure--energy design pass. | Across 30 parent-by-run units, robust-full versus score-only lowered $|\Delta$MFE by 2.669 kcal/mol and pairing-profile distance by 0.0985, while lowering RNA-LM plausibility by 0.00864 per token. Results are computational only. |
| 2026-08-22 | Prioritize an assay-aware, checkpoint-robust scorer before sequence optimization. | A single native-performing checkpoint is not a reliable cross-assay oracle; conservative aggregation must be validated before use in design. |
| 2026-08-22 | Accept public `RNAFM_Predictor.py` as a released-classifier compatibility implementation under an explicit runtime configuration. | With the official RNA-FM backbone, `token_dropout=true`, `batch_toks=4096`, and hash-based restoration after length batching, all 4,678 fold-0 frozen predictions match the established rerun (max absolute difference $1.11\times10^{-16}$). This does not establish identity to the unreleased `vMay7_RNAFM_Predictor.py` or exact IRES-EA reproduction. |
| 2026-08-22 | Add the validated released RNA-FM fold-0 score-only design baseline to the primary matched matrix. | It scored all 15,360 frozen candidates, then selected the top 50 per parent-by-run unit. Its direct-RNA S3-held-out proxy was 0.41918, indistinguishable from random mutation (0.41914), while robust-full exceeded it by 0.01416 (95\% bootstrap CI 0.00591--0.02381). It also had $|\Delta\mathrm{MFE}|=3.899\pm1.100$ kcal/mol and pairing-profile $L_1=0.161\pm0.026$. |

## Update rule

At each follow-up, update this file with: completed items, new evidence, failed assumptions, exact run IDs, blockers and the next seven-day deliverables. Do not mark a baseline complete without an executable adapter and a manifest-backed smoke run.

## StructIRES-Rank milestone (2026-08-22)

- [x] Completed a matched four-arm ablation over the verified release-compatible RNA-FM fold-0
  classifier: `rnafm_score_only`, `+energy`, `+ensemble`, and combined
  `structires`. Every arm ranked the same 512 candidates per parent and pool and retained the same
  top 50, giving 30 matched parent-by-run units and 1,500 selected candidates per arm.
- [x] Wrote immutable direct-RNA proxy results to
  `structires_rnafm_ablation_direct_rna_s3_v1_20260822`. The S3-held-out computational proxy was
  0.41918 for RNA-FM score-only, 0.42776 for energy-only, 0.42813 for ensemble-only, and 0.43347
  for combined constraints. Combined constraints exceeded score-only by 0.01429 (95\% bootstrap CI
  0.00738--0.02195), energy-only by 0.00572 (0.00271--0.00926), and ensemble-only by 0.00535
  (0.00198--0.00974). The proxy is never an optimization objective and is not an activity assay.
- [x] Generated the matching structural summary: score-only has $|\Delta\mathrm{MFE}|=3.899\pm1.100$
  kcal/mol and pairing-profile $L_1=0.161\pm0.026$; combined constraints have
  $0.575\pm0.177$ and $0.050\pm0.010$, respectively. The manifest records exact input paths and
  SHA-256 hashes.
- [x] Completed a reporter-only top-1 cargo stress test in
  `cargo_context_reporter_top1_v1_20260822`, using public NanoLuc, Firefly and mCherry contexts.
  The first-pass constrained selector had mean crosstalk ratio 0.163 and mean context consistency
  0.841. This is a computational context stress test, not a translation result.

### Current paper naming and next decision

The completed method is named **StructIRES-Rank**: a generator-agnostic constrained rank-selection
layer, not yet a newly trained neural architecture. This is the strongest defensible immediate
paper story because it has a reproduced released baseline, single-constraint ablations, matched
budgets, provenance and an independent held-out evaluator.

The route to a trainable model called **StructIRES** is now a separate gated experiment:

1. time-box recovery of the historical custom HoPE loader and checkpoint provenance;
2. if faithful loading is possible, train assay-separated sequence/structure heads with no
   direct-RNA S3 access during training;
3. otherwise train a newly initialized, explicitly documented small sequence--structure model,
   without claiming the third-party RFamLlama pretraining as our contribution;
4. compare its score-only and constraint-guided design outputs against StructIRES-Rank on the same
   frozen pools.

**Active blocker:** the historical 130M HoPE checkpoint cannot currently be loaded faithfully by
standard Transformers because its custom `bias_raw` weights are ignored. No claim about that
checkpoint's performance is permitted until the exact custom implementation is recovered or a new
model is trained from a documented architecture.

## IRES-LM ensemble and parent-anchor milestone (2026-08-22)

- [x] Completed the main release-compatible IRES-LM design baseline by averaging all ten released
  RNA-FM and all ten released UTR-LM classifier probabilities on the same frozen 15,360-candidate
  pool. The immutable score ledger is
  `/9950backfile/lant/data/ires-design-external-runs/ireslm_ensemble_shared_pool_scores_v1_20260822`.
- [x] Added a parent-derived thermodynamic ensemble-anchor metric. For each full-length seed, an
  anchor is a parent base pair with ViennaRNA ensemble probability $P\geq0.5$; candidate retention
  is the parent-probability-weighted retention of these pairs. This is a secondary-structure
  preservation constraint, not an experimentally mapped IRES core or activity measurement. The
  complete 15,360-candidate ledger is
  `parent_ensemble_anchor_metrics_v1_20260822`.
- [x] Completed a five-arm matched selection: IRES-LM score-only, energy, global ensemble, anchor,
  and their equal-rank combination, with 1,500 candidates per arm and 30 parent-by-run units. The
  combined method has $|\Delta\mathrm{MFE}|=0.666\pm0.226$ kcal/mol, pairing-profile
  $L_1=0.037\pm0.010$, and ensemble-anchor retention $0.955\pm0.015$, versus
  $4.021\pm1.254$, $0.164\pm0.027$, and $0.646\pm0.075$ for IRES-LM score-only.
- [x] The S3-held-out direct-RNA computational proxy (three-seed 3--6-mer model, 1,957 training
  records and 222 S3 test records) was never used in candidate selection. It is 0.41933 for
  IRES-LM score-only and 0.43514 for combined \textsc{StructIRES-Rank}; paired gain is 0.01582
  (95\% bootstrap CI 0.00789--0.02421). Run:
  `structires_ireslm_anchor_direct_rna_s3_v2_20260822`.
- [x] Updated the primary manuscript table, result section and figure to use the complete IRES-LM
  ablation. Local unit tests pass (32 tests); the PDF compiles with resolved in-text citations and
  a non-empty bibliography.

## IAPV direct-RNA MPRA guardrail (2026-08-22)

- [x] Converted the public IRES-TrAPPr S1 IAPV A-substitution mutational scan into a conservative
  parent-position edit-risk ledger. It contains 208 observed tiles, directly covers 169/213 IAPV
  positions, and imputes the median observed risk for the remaining positions so optimization cannot
  exploit missing coverage. Immutable run: `iapv_mpra_mutational_risk_v1_20260822`.
- [x] Added an IAPV-only same-pool guardrail ablation. It combines IRES-LM, energy, global ensemble,
  parent-anchor and lower measured-mutational-risk ranks; all arms still take 50 candidates from each
  of the same three 512-candidate pools. MPRA edit risk is $2.438\pm0.174$ for \textsc{StructIRES-Rank}
  and $1.782\pm0.087$ with the guardrail (paired difference $-0.656$, bootstrap interval
  $-0.933$ to $-0.480$). The run is `structires_ireslm_iapv_mpra_guardrail_v2_20260822`.
- [x] The guardrail carries an explicit trade-off: mean $|\Delta\mathrm{MFE}|$ increases from
  $0.641\pm0.069$ to $0.887\pm0.131$ kcal/mol. It is therefore a supplementary, transparent
  single-seed functional constraint rather than a claimed all-seed activity gain. It does not
  validate activity of generated candidates.

## Representative structural case (2026-08-22)

- [x] Added a provenance-locked illustrative IAPV case figure. It compares the rank-1 score-only
  and \textsc{StructIRES-Rank} candidates from the frozen seed-42 pool against the parent ensemble.
  Score-only has IRES-LM 0.976, $|\Delta\mathrm{MFE}|=3.70$ and anchor retention 0.682;
  \textsc{StructIRES-Rank} has IRES-LM 0.963, $|\Delta\mathrm{MFE}|=0.00$ and retention 1.000.
  This deterministic visual is explanatory only, never a candidate-activity claim. Manifest:
  `structires_iapv_case_v2_20260822`.

## Main-text / supplement and visual audit (2026-08-22)

- [x] Froze the first-draft evidence boundary in `docs/MAIN_TEXT_SCOPE_2026-08-22.md`.
  The main text keeps the assay audit, matched five-arm design ablation, S3-held-out computational
  proxy, and IAPV-only MPRA edit-risk guardrail. The complete metric ledger, detailed shortcut
  controls, cargo stress test, and legacy 174-nt task are supplementary provenance.
- [x] Rebuilt the workflow and primary ablation as repository-native vector figures using a fixed,
  colour-vision-friendly method palette. Figure 6 is now a readable 2x2 matched-ablation layout.
- [x] Replaced the earlier IAPV base-pair-probability matrix with a ViennaRNA NaviView MFE
  secondary-structure case: parent / score-only / StructIRES-Rank layouts, edited positions,
  retained or reduced parent-derived ensemble anchors, the public IAPV risk map, and its
  same-budget guardrail summary. Output: `paper/figures/fig7_iapv_secondary_structure.pdf`.
  It is a provenance-locked visual explanation, not a functional validation.
- [x] Recompiled the venue-neutral PDF after the visual update; `paper/build/main.pdf` is the
  current reviewed artifact.

## StructIRES classifier milestone (started 2026-08-23)

- [x] Defined the validation-clean classifier protocol and model ablations in
  `docs/STRUCTIRES_EXPERIMENT_PLAN_2026-08-23.md`.  The formal development protocol is the existing
  length-174 90%-identity cluster split; validation AUPR selects checkpoints and the test partition
  is not used for selection.
- [x] Built a label-free, position-aware ViennaRNA structural cache for exactly the 44,641 records
  in that split: `structires_position_profiles_hamming90_len174_v1_20260823.npz` (44,641 x 174 x 4).
  Channels are ensemble pairing probability, MFE paired state, centroid paired state and normalized
  position.  The companion manifest pins the source dataset and assignment SHA-256 values.
- [x] Audited the local RNA-FM classifier implementation against the released upstream
  `IRES_RNAFM.py`: upstream mean-pools biological token embeddings (excluding BOS/EOS/padding).
  An earlier three-epoch frozen-CLS smoke run (`structires_rnafm_sequence_only_hamming90_smoke_v1_20260823`)
  had strict-test AUPR 0.1896 and is retained only as a failed pipeline smoke, not a result.
  A last-two-layer CLS pilot was deliberately terminated after the same audit; its log remains
  preserved at `logs/structires_rnafm_last2_pilot_v1_20260823.log`.
- [x] The generic strict-split pilots were stopped after the released classifier audit established
  that they answer a different question: they initialize from the public RNA-FM prior but do not
  begin from the author's IRES-supervised classifier. Their logs remain preserved as development
  evidence; they are not reportable classifier results.

## Checkpoint-native StructIRES adapter (started 2026-08-23)

- [x] Recovered the exact released IRES-RNAFM topology from the upstream source: public RNA-FM
  t12, BOS representation, $640\rightarrow40\rightarrow2$ classification head. The original
  command fully fine-tunes this architecture with a masked-LM auxiliary loss and truncates inputs
  at 1,024 tokens.
- [x] Added `scripts/train_structires_release_adapter.py`: it loads each released fold checkpoint,
  freezes its RNA-FM and IRES classifier parameters, and adds a 21-feature label-free ViennaRNA
  MFE/ensemble structural residual with a learned gate. The residual projection is initialized to
  zero, and an explicit numerical check requires its pre-training predictions to equal the released
  sequence-only checkpoint before training can proceed.
- [x] Corrected the first launch after it exposed a missing 1,024-token truncation. This was an
  input-preprocessing error, not an asset failure; no score from that failed launch is retained.
- [x] Built the full released-benchmark, label-free position-profile cache at
  `/9950backfile/lant/data/ires-design-external-runs/structires_release_position_profiles_1024_v1_20260823`.
  It contains 46,774 unique public sequences as a memory-mappable
  $46{,}774\times1{,}024\times5$ array: ensemble pairing probability, MFE state, centroid state,
  normalized position and an explicit valid-position mask.  Only 68 source records exceed the
  released model's 1,024-token upper bound; the cache manifest pins the source SHA-256 and
  ViennaRNA 2.7.2.
- [x] Added and pushed the position-aware successor
  `scripts/train_structires_release_profile_adapter.py` (commits `c99542c`, `4679b03`).  It uses a
  masked local CNN structural encoder and zero-initialized gated residual into the frozen released
  checkpoint.  The first `fold4_v1` launch stopped during its numerical equivalence check because
  of a batch-index normalization-shape bug; it produced no metrics and remains preserved as a
  failure log.  The corrected immutable `fold4_v2` run is active.  The repair has an explicit
  scalar-and-batch normalization smoke test; all repository unit tests remain green (32/32).
- [x] Completed the corrected position-aware `fold4_v2` run and independently recomputed its
  saved 4,678 official-test predictions.  With validation-only epoch/threshold selection, the
  frozen released baseline is AUC/AUPR/F1/MCC/ECE = 0.78530/0.62797/0.54961/0.44299/0.24239;
  the masked-CNN structural residual is 0.78565/0.62834/0.54445/0.43518/0.19330.  The nonzero
  maximum probability change (0.09236) confirms that the structural path was active, but the
  AUC/AUPR gains (+0.00035/+0.00037) are too small and F1/MCC decrease.  This single fold is
  preserved as a negative/exploratory result, not promoted to the paper table or expanded to
  ten folds.  The next classifier iteration requires a matched trainable sequence-head control
  and a trainable structure-fusion head under the same validation-only protocol.
- [x] Completed matched fold-4 controls with frozen RNA-FM and the same trainable released
  classifier head: `structires_release_matched_sequence_head_fold4_v1_20260823` and
  dropout-aligned `structires_release_matched_profile_fusion_head_fold4_v2_20260823`.  The first
  fusion-head launch remains preserved without metrics because it bypassed the released head's
  training dropout; commit `281c46a` corrects that regularization mismatch in the immutable v2
  run.  The v2 prediction ledger contains 4,678 official-test records, and an independent
  recomputation exactly reproduces its AUROC, AUPR, F1, accuracy and MCC (maximum absolute
  discrepancy 0).  Released checkpoint / matched sequence-head / matched structure-fusion are,
  respectively: AUROC 0.78530 / 0.78285 / 0.78118; AUPR 0.62797 / 0.62220 / 0.61986; F1
  0.54961 / 0.54844 / 0.54672; MCC 0.44299 / 0.45068 / 0.44631; and ECE 0.24239 / 0.19630 /
  0.18103.  Thus the current position-profile fusion improves calibration relative to the
  released checkpoint but fails the predeclared discrimination criterion against both controls.
  It is a completed negative pilot, not a main-table gain and is not expanded to ten folds.
  The next iteration must first reproduce the author's last-layer/full fine-tuning and masked-LM
  auxiliary-training regime as a matched sequence-only baseline before assessing a structural
  fusion under that stronger sequence adaptation.
- [x] Completed the matched full-adaptation pair starting from the same released fold-4 checkpoint,
  with full RNA-FM adaptation, 15\% masked-token inputs, the author's auxiliary masked-LM loss
  (weight 1), and class-loss multiplier 2.  The immutable sequence-only run is
  `structires_release_fullmlm_sequence_head_fold4_v1_20260823` (node 56 GPU 1); the matching
  structural fusion run is `structires_release_fullmlm_profile_fusion_head_fold4_v2_20260823`
  (node 47 GPU 4).  An initial v1 fusion launch on node 56 GPU 0 failed with a documented CUDA
  OOM because an unrelated process already occupied 44.7 GB; its log is retained and the v2
  restart used an otherwise idle GPU without changing the experiment configuration.  The v2
  fusion run early-stopped after epoch 4 (selected epoch 1); its 4,678 official-test predictions
  also independently recompute exactly.  Relative to full-MLM sequence-only, fusion has AUROC
  0.76966 versus 0.76949 but lower AUPR (0.60596 versus 0.60810), F1 (0.52632 versus 0.53291),
  MCC (0.44526 versus 0.45201), and ECE (0.06948 versus 0.09390).  Relative to the released
  checkpoint, both have substantially lower AUROC/AUPR/F1.  This full-adaptation fusion is
  therefore a completed negative pilot: it does not pass the discriminative-improvement gate and
  will not be expanded or included as a main-table gain.
- [x] The full-adaptation sequence-only arm completed after validation-AUPR early stopping at
  epoch 5 (selected epoch 2).  Its 4,678-record official-test prediction ledger was independently
  recomputed exactly: released checkpoint versus full-MLM sequence-only is AUROC 0.78530 versus
  0.76949, AUPR 0.62797 versus 0.60810, F1 0.54961 versus 0.53291, MCC 0.44299 versus 0.45201,
  and ECE 0.24239 versus 0.09390.  Thus this stronger adaptation improves calibration and slightly
  improves threshold-dependent MCC, but degrades ranking and F1; it is a completed negative
  sequence-only control, not a primary recognition result.  The paired full-MLM fusion result is
  recorded immediately above and also fails the primary discrimination criterion.
- [x] Completed a structure-only diagnostic in
  `structires_release_structure_only_logistic_fold4_v2_20260823`.  A validation-selected,
  class-balanced logistic model receives only the 21 label-free ViennaRNA MFE/ensemble features;
  its 4,678-record prediction ledger independently recomputes exactly to AUROC 0.56508, AUPR
  0.23960, F1 0.33451 and MCC 0.05938.  This demonstrates weak independent structural signal but
  rules out treating the current global features as an adequate replacement for RNA-FM.  The
  earlier v1 is retained as a format-failure audit only: it wrote a plain CSV with a `.gz` suffix
  and is not used for any result.
- [x] Completed the validation-clean fixed-score diagnostic in
  `structires_release_score_structure_logistic_fold4_v2_20260823`.  It performs one released
  checkpoint inference pass for all fold records, then trains score-only and score-plus-structure
  logistic heads solely on the same training subset and chooses regularization by validation AUPR.
  Score-only preserves the released ranking exactly.  Although adding structure improves validation
  AUPR from 0.71585 to 0.71875, its independent test AUPR declines from 0.62797 to 0.62487;
  AUROC/F1/MCC likewise change from 0.78530/0.54961/0.44299 to
  0.78497/0.54708/0.43977.  All three prediction columns independently recompute exactly.  Thus
  the apparent validation gain does not generalize and this transparent fusion is retained as a
  negative diagnostic, not a manuscript gain.
- [x] Completed the contact-aware fold-4 pilot,
  `structires_release_mfe_contact_fusion_fold4_v1_20260823`, using the label-free sparse MFE
  contact cache (46,774 sequences; 2,425,739 base-pair edges) rather than global or per-position
  summaries.  It froze the released RNA-FM/IRES head, pooled contextual residue pairs along MFE
  contact edges, and injected a zero-initialized gated residual.  Its pretraining-equivalence check
  passed and validation selected epoch 5 (AUPR 0.73549), without using test labels.  Independent
  recomputation from `test_predictions.csv.gz` exactly matches the stored test values.  However,
  released checkpoint versus contact fusion is AUROC 0.78530 versus 0.78027, AUPR 0.62797 versus
  0.62204, F1 0.54961 versus 0.54472, and MCC 0.44299 versus 0.44165; only ECE improves from
  0.24239 to 0.10803.  It is therefore a completed negative discrimination pilot, not expanded
  and not reported as a main-table gain.
- [x] Completed matched author-style native retraining on fold 4 from the public RNA-FM base
  checkpoint.  The initial diagnostic pair, `structires_native_sequence_authorstyle_fold4_v1_20260823` and
  `structires_native_contact_authorstyle_fold4_v1_20260823`, omitted the upstream implementation's
  per-epoch shuffle of length-aware batches and its `LinearLR` schedule. It was stopped after
  sequence epoch 3 / contact epoch 2, before any test evaluation, to free its two GPUs; logs are
  preserved as a non-final diagnostic. The primary pair is
  `structires_native_sequence_authorstyle_batchshuffle_fold4_v2_20260823` and
  `structires_native_contact_authorstyle_batchshuffle_fold4_v2_20260823`, which additionally
  reproduces those training details. Both pairs use the public t12 checkpoint,
  full RNA-FM fine-tuning, BOS 640$\rightarrow$40$\rightarrow$2 classification topology,
  classification loss weight 2, 15\% masked-LM auxiliary loss weight 1, the same source fold,
  seed, internal validation split, 10-epoch budget, and untouched native test set.  They differ
  only by the sparse MFE-contact encoder and gated residual.  This is the first direct StructIRES
  sequence-only versus structure-fusion comparison that does not initialize from a released,
  upstream-test-selected IRES classifier checkpoint.  Validation selected epoch 4 in both arms;
  the independent 4,678-record native-test recomputation gives sequence-only
  AUROC/AUPR/F1/MCC/ECE = 0.77076/0.60441/0.52664/0.43319/0.18145 and contact fusion
  = 0.77191/0.60676/0.53998/0.45051/0.22011.  The paired deltas are +0.00115 AUROC,
  +0.00235 AUPR, +0.01334 F1 and +0.01732 MCC, but +0.03866 ECE.  This is a small,
  single-fold positive discrimination signal with worse calibration, not a paper-level claim.
  Immutable paired verifier output: `structires_native_authorstyle_batchshuffle_pair_fold4_v2_20260823`.
- [x] Completed the predeclared fold-0 replication pair,
  `structires_native_sequence_authorstyle_batchshuffle_fold0_v2_20260823` and
  `structires_native_contact_authorstyle_batchshuffle_fold0_v2_20260823`, with the exact v2
  protocol.  The fold-0 contact architecture was selected from the fold-4 *validation*
  trajectory only; no fold-4 test result was consulted before launching fold 0.  Fold 0:
  sequence-only is AUROC/AUPR/F1/MCC/ECE = 0.76369/0.58002/0.50284/0.41214/0.20021 and
  contact fusion = 0.76789/0.57687/0.50714/0.38796/0.23282.  Thus contact changes AUROC by
  +0.00421 and F1 by +0.00429, but AUPR by -0.00315, MCC by -0.02418 and ECE by +0.03261.
  The mixed fold-0 result does not replicate the fold-4 discrimination pattern; the two-fold
  evidence is not eligible for a positive main-table claim.  The first fold-0 monitor noticed
  `metrics.json` before the script had finished writing its prediction ledger and manifest, and
  therefore exited without an output; a provenance-complete monitor subsequently ran the same
  immutable verifier after both files existed.  Verified pair output:
  `structires_native_authorstyle_batchshuffle_pair_fold0_v2_20260823`.
- [x] Completed the additional fold-2 matched pair under the unchanged v2 protocol:
  `structires_native_sequence_authorstyle_batchshuffle_fold2_v2_20260823` and
  `structires_native_contact_authorstyle_batchshuffle_fold2_v2_20260823`.  Fold 2 was scheduled
  only after fold 4 had completed and is an independent replication, not an adaptive replacement
  of the architecture.  Its validation-selected locked-test sequence-only metrics are
  AUROC/AUPR/F1/MCC/ECE = 0.77406/0.61452/0.53254/0.43274/0.18961; contact fusion is
  0.78203/0.62018/0.55319/0.47938/0.24637.  Contact therefore changes AUROC by +0.00796,
  AUPR by +0.00566, F1 by +0.02065 and MCC by +0.04663, while worsening ECE by +0.05676.
  Verified pair output: `structires_native_authorstyle_batchshuffle_pair_fold2_v2_20260823`.
- [x] Aggregated the three independently verified matched native folds 0, 2 and 4 using
  `scripts/summarize_native_structires_multifold.py`.  The sequence-only re-train is
  AUROC/AUPR/F1/MCC/ECE = 0.76950$\pm$0.00530/0.59965$\pm$0.01774/
  0.52068$\pm$0.01572/0.42602$\pm$0.01203/0.19043$\pm$0.00941; contact fusion is
  0.77394$\pm$0.00728/0.60127$\pm$0.02217/0.53343$\pm$0.02371/
  0.43928$\pm$0.04673/0.23310$\pm$0.01313.  Foldwise mean contact--sequence deltas are
  +0.00444 AUROC, +0.00162 AUPR, +0.01276 F1 and +0.01326 MCC, with +0.04267 ECE.
  The result is now added to the provenance-ledger table as a controlled, three-fold native
  recognition ablation; folds overlap and it is not presented as independent assay validation.
  Immutable aggregate: `structires_native_authorstyle_batchshuffle_multifold_0_2_4_v2_20260823`.
- [x] Stopped the initial strict 90\%-identity frozen-backbone development jobs before any test
  evaluation.  `structires_rnafm_sequence_only_hamming90_full3seed_v1_20260823` and
  `structires_rnafm_gated_profile_hamming90_full3seed_v1_20260823` each showed near-prevalence
  validation AUPR in their early seed-42 epochs, consistent with the earlier frozen smoke underfit.
  They have no test predictions or reportable metrics and remain only as preserved development logs.
- [x] Stopped the replacement last-two-block strict development pair after its first two
  sequence-only validation epochs and first fusion epoch remained at AUPR 0.1842, 0.1761 and
  0.1761, respectively, against a 0.1658 validation positive fraction.  The jobs
  `structires_rnafm_sequence_only_hamming90_last2_dev_s42_v1_20260823` and
  `structires_rnafm_gated_profile_hamming90_last2_dev_s42_v1_20260823` have no test evaluation.
  This is evidence of insufficient adaptation capacity, not evidence against structural features.
- [x] Completed the strict development pair with the entire public RNA-FM t12 model trainable,
  seed 42, three epochs, validation-only selection and no test evaluation:
  `structires_rnafm_sequence_only_hamming90_full_dev_s42_v1_20260823` is the matched sequence
  control; `structires_rnafm_gated_profile_hamming90_full_dev_s42_v1_20260823` adds the existing
  label-free ViennaRNA ensemble-pairing/MFE-state/position profile cache and gated profile encoder.
  The frozen `70/15/15` identity-aware assignment is unchanged.  If this adaptation finds a viable
  validation signal, after which the predeclared configuration was locked and rerun across seeds.
  These development artifacts do not overwrite or pool with the native-fold result.
- [x] **Completed locked final strict testing.**  On the shared seed-42 validation split,
  full gated profile fusion reached AUPR 0.25140 / AUROC 0.59707 / F1 0.31250 at epoch 3, versus
  the matched sequence-only best validation AUPR 0.22645 / AUROC 0.57774 / F1 0.30228 at epoch 2.
  The final comparison fixes three epochs for both arms so their optimization budget is identical.
  Six runs evaluated the untouched test partition once: sequence-only seeds 42/43/44 in
  `structires_rnafm_sequence_only_hamming90_full_locked_test_s{42,43,44}_v1_20260823`, and gated
  profile fusion seeds 42/43/44 in
  `structires_rnafm_gated_profile_hamming90_full_locked_test_s{42,43,44}_v1_20260823`.  No result
  from these test runs was used to choose another architecture or training schedule.  The
  prediction-ledger verifier `scripts/summarize_strict_structires_comparison.py` independently
  recomputed every metric from the six test prediction files and confirmed paired sequence IDs and
  labels.  The final strict means are sequence-only AUC/AUPR/F1/MCC/ECE =
  0.55926$\pm$0.00417/0.18788$\pm$0.00505/0.29457$\pm$0.00179/
  0.07016$\pm$0.00511/0.16458$\pm$0.00000 and gated profile fusion =
  0.55677$\pm$0.01652/0.18290$\pm$0.00334/0.29681$\pm$0.00987/
  0.07405$\pm$0.02953/0.16458$\pm$0.00000.  Fusion minus sequence is -0.00249 AUC,
  -0.00498 AUPR, +0.00224 F1 and +0.00389 MCC.  It is therefore an explicit negative strict
  ablation, not a recognition-improvement claim.  Immutable summary:
  `structires_rnafm_strict_locked_comparison_3seed_v2_20260823`.
- [x] Stopped the complementary strict structure-only ablation after its first validation epoch.
  `structires_rnafm_structure_profile_only_hamming90_dev_s42_v1_20260823` reached AUPR 0.14883,
  below the 0.1658 validation positive fraction, with AUROC 0.45720.  It has no test evaluation;
  the preserved log shows that this fixed global profile CNN does not carry usable standalone
  signal under the strict split.  This does not invalidate an RNA-FM-contextual contact encoder.
- [x] Completed the complementary strict RNA-FM-contextual MFE-contact development ablation in
  `structires_rnafm_gated_contact_hamming90_full_dev_s42_v1_20260823`.  It uses an immutable
  44,641-record, label-free MFE base-pair cache aligned to the locked assignment
  (`structires_strict_mfe_pair_contacts_hamming90_len174_v2_20260823`), pools contextual
  RNA-FM representations across predicted paired positions, and adds a zero-initialized gated
  residual.  The full t12 backbone, three-epoch budget, seed and validation-only selection match
  the strict sequence-only development control.  Its validation AUPR/AUROC/F1 is
  0.16734/0.52766/0.28700 at epoch 1, then declines to 0.16087/0.49806/0.28321 and
  0.15761/0.48804/0.28275 at epochs 2 and 3.  The best AUPR is far below the matched
  sequence-only development best (0.22645), so no test records were constructed or evaluated and
  no multi-seed expansion was launched.  This is a preserved strict negative ablation, not a
  paper result.  An earlier `...contacts...v1...` cache included all 46,774 canonical records and
  was never used by any training run; v2 is the only strict-aligned cache.
- [ ] Running: released checkpoint fold 0 plus zero-initialized structural adapter,
  `structires_release_adapter_fold0_v1_20260823`, on node 56 GPU 1. Epoch 1 validation AUPR is
  0.7019. This is a validation-only trajectory, not a held-out result and not a paper claim.
- [x] Completed and independently re-summarized release fold 2 with 1,000 stratified bootstrap
  replicates in `structires_release_adapter_partial_summary_fold2_v1_20260823`. Under thresholds
  selected solely on the fold-train validation subset, the released checkpoint has AUROC 0.7889,
  AUPR 0.6361 and F1 0.5626; the structure adapter has 0.7923, 0.6392 and 0.5644, respectively.
  Its ECE improves from 0.2267 to 0.1959. This is an audited *single-fold signal*, not a ten-fold
  conclusion or a manuscript headline. The JSON metrics exactly match a prediction-file recompute.
- [x] Partial three-fold audit (folds 2, 5 and 7) in
  `structires_release_adapter_partial_summary_folds2_5_7_v1_20260823`: mean AUROC is 0.7832 for
  the checkpoint and 0.7835 for the adapter; mean AUPR is 0.6215 and 0.6217; mean F1 is 0.5478 and
  0.5481; mean MCC is 0.4568 and 0.4637. Mean ECE decreases from 0.2302 to 0.1855. This mixed,
  incomplete result supports a calibration/specificity signal but does **not** justify a claim of
  material recognition improvement until the complete ten-fold audit is available.
- [x] Completed and independently checked the complete global-feature ten-fold audit in
  `structires_release_adapter_10fold_summary_v1_20260823`.  It contains exactly folds 0--9 once
  each; its aggregate means reproduce from the hashed prediction-ledger rows.  The frozen
  checkpoint versus the 21-feature global structure residual is AUROC 0.77899 versus 0.77734,
  AUPR 0.61455 versus 0.61351, F1 0.54343 versus 0.54182, MCC 0.45135 versus 0.45176, and ECE
  0.22855 versus 0.17764 (mean $\pm$ sample standard deviations are in the aggregate CSV).
  Therefore the global feature adapter has a reproducible calibration improvement but no
  discrimination improvement; it is retained as a negative/calibration ablation and is excluded
  from the primary recognition claim and manuscript main table.
- [x] Clarified a provenance-sensitive output name in the legacy global-adapter runner.  Its
  `adapter_validation_selected` JSON field contains **official-test** metrics; “validation” means
  that the epoch and F1 threshold were selected using only the upstream-train validation subset.
  It is not a validation-set score and must be reported, if at all, as
  “official test, validation-selected.”  Existing immutable run artifacts retain the original key;
  this clarification prevents an ambiguous label from entering the manuscript.
