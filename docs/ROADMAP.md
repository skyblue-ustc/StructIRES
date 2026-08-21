# Implementation roadmap

## P0: route freeze and data gate — 2026-08-21 to 2026-09-04

- [x] Independent Git repository scaffold.
- [x] Baseline registry and role/fairness labels.
- [x] Canonical candidate JSONL schema.
- [x] Random-mutation baseline and low-cost FASTA audit.
- [x] Frozen smoke/de novo/mutation configuration files.
- [x] Primary-source literature and baseline audit.
- [x] Assay-labelled evaluator registry.
- [x] Recover and hash all nine IRES-TrAPPr supplementary tables.
- [ ] Build immutable DNA-reporter, direct-RNA, Albatross and cargo-context manifests.
- [ ] Freeze similarity/family splits and publish the assay-shift gate report.

## P1: strict baseline smoke gate — 2026-09-05 to 2026-09-25

- [ ] IRES-DM released-output parser and manifest.
- [ ] IRES-DM same-protocol reproduction smoke run.
- [ ] IRES-EA command adapter and lineage parser.
- [ ] Budget-matched random-screen and score-only GA baselines.
- [ ] Delli-Ponti-style structure-only mutation baseline.
- [ ] LM-likelihood-guided mutation baseline.
- [ ] NSGA-II shared search implementation.
- [ ] Optional RFamLlama/GenerRNA proposal adapters only after direct baselines pass.

## P2: evaluators and primary method — 2026-09-26 to 2026-10-23

- [ ] Reproduce the legacy IRES-LM scorer and its native split metrics.
- [ ] Train calibrated shallow/frozen-embedding direct-RNA scorers with family holdouts.
- [ ] Freeze LAMAR as a secondary, non-optimization evaluator.
- [ ] ViennaRNA MFE/BPP/ensemble features with version capture.
- [ ] IRES Crosstalk Ratio and Structure Consistency across the frozen cargo panel.
- [ ] Albatross structure audit on the frozen evaluation subset.
- [ ] Novelty, diversity, and applicability-domain metrics.
- [ ] Per-seed aggregation and paired confidence intervals.

## P3: formal experiments — 2026-10-24 to 2026-11-30

- [ ] Task 0 cross-assay audit with frozen figures and tables.
- [ ] Task 1 type-balanced full-length mutation benchmark, three random seeds.
- [ ] Nested `score_only/mfe/ensemble/context/robust_full` ablation.
- [ ] Task 2 IRES-DM legacy de novo reference/reproduction table.
- [ ] Worst-case cargo robustness, uncertainty and diversity analyses.

## P4: paper package — 2026-12-01 to 2027-01-15

- [ ] Frozen paper tables, figures, and candidate manifest.
- [ ] Internal methods/results draft and related-work audit.
- [ ] Patentability/public-disclosure review before public release or preprint.
- [ ] Submission-ready manuscript and reproducibility package.

## Release gate

- [ ] Patent/public-disclosure order confirmed.
- [ ] No secrets, personal paths, weights, raw licensed data, or copied papers.
- [ ] Every third-party adapter has citation and license notes.
- [ ] Reproduction commands pass in a clean environment.
- [ ] Repository visibility may change from private to public only after authorization.
