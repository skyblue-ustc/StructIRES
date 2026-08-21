# Implementation roadmap

## P0: repository and data contract

- [x] Independent Git repository scaffold.
- [x] Baseline registry and role/fairness labels.
- [x] Canonical candidate JSONL schema.
- [x] Random-mutation baseline and low-cost FASTA audit.
- [x] Frozen smoke/de novo/mutation configuration files.
- [ ] Immutable MPRA data manifest and similarity-cluster split.
- [ ] Locked optimization/evaluation scorer registry.

## P1: executable baselines

- [ ] Official RFamLlama generation/scoring adapter.
- [ ] GenerRNA generation/scoring adapter.
- [ ] IRES-DM released-output parser and manifest.
- [ ] IRES-EA command adapter and lineage parser.
- [ ] LM-likelihood-guided mutation baseline.
- [ ] NSGA-II shared search implementation.

## P2: evaluation

- [ ] Cross-fitted hurdle MPRA scorer.
- [ ] Public IRES-AI and DeepIRES locked evaluators.
- [ ] ViennaRNA MFE/BPP/ensemble features with version capture.
- [ ] Novelty, diversity, and applicability-domain metrics.
- [ ] Per-seed aggregation and paired confidence intervals.

## P3: formal runs

- [ ] Five-baseline smoke gate.
- [ ] Task A three-seed candidate pools and nested ablation.
- [ ] Task B ten-seed mutation experiment.
- [ ] IRES-DM external-reference table.
- [ ] Frozen paper tables, figures, and candidate manifest.

## Release gate

- [ ] Patent/public-disclosure order confirmed.
- [ ] No secrets, personal paths, weights, raw licensed data, or copied papers.
- [ ] Every third-party adapter has citation and license notes.
- [ ] Reproduction commands pass in a clean environment.
- [ ] Repository visibility may change from private to public only after authorization.

