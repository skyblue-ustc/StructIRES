# Project progress

Last updated: 2026-08-21 (Asia/Shanghai)

## Current phase

**P0 — route freeze and data gate.** No large model training is authorized until the assay-shift gate is complete.

## Completed

- [x] Migrated the active work into an independent `ires-design` repository.
- [x] Preserved RFamLlama attribution and moved the local reproduction to supplementary status.
- [x] Downloaded and text-audited the core direct and adjacent method papers.
- [x] Inspected the IRES-AI supplement and local upstream implementation.
- [x] Identified the assay mismatch between the old 174-nt DNA/lentiviral library and direct-RNA IRES-TrAPPr.
- [x] Identified Delli-Ponti structure/MFE mutation as a mandatory baseline rather than a novelty claim.
- [x] Added IRES–cargo crosstalk as an evidence-backed context constraint.
- [x] Reclassified RFamLlama and GenerRNA as optional proposal backbones.
- [x] Froze direct baselines and the three-track task hierarchy in documentation/configuration.
- [x] Retrieved, field-audited and hashed all nine IRES-TrAPPr XLSX supplements plus the supplementary PDF.
- [x] Audited the released IRES-AI split: ten repeated 90/10 holdouts, not a mutually exclusive
  ten-fold partition.
- [x] Completed a manifest-backed ten-fold lightweight pilot on reconstructed, nested folds.
- [x] Created a venue-neutral manuscript scaffold, predeclared tables, and two vector figures.

## Frozen scientific story

> We study whether assay-aware, ensemble-structure- and cargo-context-constrained optimization can produce computationally robust variants of bona fide full-length viral IRESes, compared with IRES-specific, score-only, structure-only, LM-guided and random baselines under matched budgets.

The paper has three separate result blocks:

1. cross-assay failure/transfer audit;
2. primary full-length seeded IRES design;
3. secondary 174-nt legacy IRES-DM reproduction.

## Readiness matrix

| Component | Status | Exit condition |
|---|---|---|
| Core literature | Complete | Source/hash manifest committed. |
| IRES-AI upstream code | Located, partially audited | Exact environment, commit, data and commands recorded. |
| IRES-TrAPPr supplement | Retrieved and hashed | Normalize controls, duplicates and assay conditions into immutable tables. |
| Assay-labelled data schema | Planned | All sources normalized without erasing assay context. |
| Random split audit | Complete | Assignment distribution and dataset hash recorded. |
| Similarity/family split | Planned | Immutable train/validation/test manifest. |
| Random mutation | Ready | Existing tests pass. |
| Random screen | Planned | Budget-matched smoke run. |
| Score-only GA | Planned | Paired initialization/query-count smoke run. |
| Structure-only mutation | Planned | Published rule reproduced on a small seed set. |
| IRES-EA | Upstream available | Adapter and lineage parser pass smoke test. |
| IRES-DM released parser | Planned | Released candidates normalized with provenance. |
| IRES-DM retraining | Deferred until data gate | Native metric reproduction before formal run. |
| ViennaRNA ensemble/context metrics | Planned | Version-pinned unit tests and reference examples. |
| Local RNA-AR frozen probe | Blocked by version audit | Restore the custom HoPE loader without ignored `bias_raw` weights. |
| Albatross audit | Data/source identified | Frozen subset executable with revision manifest. |

## Reviewed pilot runs

| Run ID | Protocol | Result | Interpretation |
|---|---|---|---|
| `prediction_lightweight_reconstructed_nested10_v2_20260821_s42` | 10 mutually exclusive project folds; next fold validates threshold | Composition: AUC 0.710±0.009, AUPR 0.405±0.015, F1 0.436±0.016 | Strong shortcut-control signal. |
| same run | same | Hashed 3–6-mer: AUC 0.723±0.011, AUPR 0.452±0.016, F1 0.452±0.013 | Required lower baseline; not yet a cluster-split result. |

The attempted local 130M RNA-AR probe produced no accepted score. Standard Transformers ignored
HoPE `bias_raw` checkpoint weights; the process was stopped and the runner now fails closed when the
custom HoPE implementation is unavailable.

## Next actions

### 2026-08-22 to 2026-08-28

- [x] Retrieve IRES-TrAPPr Tables S1–S9 and verify sequence, condition, replicate and TE fields.
- [ ] Inventory exact files in the local IRES-AI repository and create a source/data manifest.
- [ ] Normalize old MPRA and direct-RNA records into an assay-labelled table.
- [ ] Build sequence-cluster and viral-family split candidates; check overlap/leakage.

### 2026-08-29 to 2026-09-04

- [x] Run simple k-mer/GC/length baselines on the reconstructed legacy random split.
- [ ] Repeat the lightweight baselines on frozen similarity/family splits and within each assay.
- [ ] Evaluate legacy IRES-LM transfer to direct-RNA labels without tuning on the test split.
- [ ] Produce the assay-shift gate report with calibration, rank correlation and top-k overlap.
- [ ] Freeze the full-length seed panel and cargo panel only after the report.

### After the data gate

- [ ] Implement and smoke-test random, score-only GA, structure-only and IRES-EA baselines first.
- [ ] Implement ViennaRNA ensemble and cargo-context metrics.
- [ ] Add the proposed `robust_full` objective only after all required baselines run.

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

## Update rule

At each follow-up, update this file with: completed items, new evidence, failed assumptions, exact run IDs, blockers and the next seven-day deliverables. Do not mark a baseline complete without an executable adapter and a manifest-backed smoke run.
