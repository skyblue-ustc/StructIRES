# Project progress

Last updated: 2026-08-22 (Asia/Shanghai)

## Current phase

**P3/P4 — result consolidation and manuscript freeze.** P1 is complete: all ten released RNA-FM
checkpoints passed size, SHA-256 and `torch.load` checks and were evaluated separately on their
official test holdouts. The final native summary and ten-checkpoint direct-RNA transfer are frozen.
DeepCIP official inference remains a time-boxed environment blocker rather than a claimed result.

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

## Update rule

At each follow-up, update this file with: completed items, new evidence, failed assumptions, exact run IDs, blockers and the next seven-day deliverables. Do not mark a baseline complete without an executable adapter and a manifest-backed smoke run.
