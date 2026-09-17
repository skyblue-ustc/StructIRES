# IRES Design

Reproducible, baseline-first experiments for assay- and context-aware computational design of internal ribosome entry site (IRES) RNA sequences.

> Status: active research repository. The current claims are computational candidate enrichment under specified reporter/MPRA settings; no generated sequence is claimed to have improved biological activity without experimental validation.

## BIBE 2026 reproducibility snapshot

The minimal, sequence-free artifacts used for the BIBE 2026 manuscript are in
[`release/bibe2026`](release/bibe2026/README.md). The snapshot contains the
locked three-fold recognition metrics, aggregate candidate-selection results,
provenance hashes, and the exact model contract. It intentionally excludes raw
benchmark data, pretrained weights, checkpoints, generated candidate
sequences, W&B files, and cluster logs.

Validate the committed snapshot with:

```bash
python scripts/verify_bibe2026_release.py
```

## Scope

This repository has one main story: optimize experimentally supported, full-length viral IRESes while resisting single-oracle score hacking and preserving structural function in downstream cargo contexts.

Three result tracks are kept separate:

- **Task 0 — assay-shift audit:** test whether models learned from the 2016 DNA/lentiviral reporter data transfer to the 2026 RNA-based IRES-TrAPPr assay.
- **Task 1 — primary seeded design:** optimize a frozen panel of full-length viral IRESes under matched edit and oracle-query budgets.
- **Task 2 — legacy de novo benchmark:** reproduce 174-nt IRES-DM-style design under its original reporter context; this is not the primary biological claim.

The proposed layer is deliberately generator-agnostic:

```text
full-length IRES seed -> assay-labelled function scorers
                      -> ensemble structure + IRES-cargo context objectives
                      -> standard NSGA-II/Pareto search
                      -> independent, uncertainty-aware evaluation
```

## Baselines

| ID | Role | Main comparison |
|---|---|---|
| `ires_ea` | exact IRES mutation method | direct Task 1 baseline |
| `structure_only_mutation` | published fold/MFE mutation rule | structure-only Task 1 baseline |
| `score_only_ga` | predictor-guided GA | primary paired control |
| `lm_likelihood_mutation` | general LM mutation ranking | naturalness/fitness-prior control |
| `random_mutation` | non-neural lower bound | budget-matched Task 1 control |
| `ires_dm` | released IRES diffusion outputs | external Task 2 reference |
| `ires_dm_retrained` | same-protocol IRES-DM retraining | strict Task 2 baseline |
| `random_screen` | large random pool plus top-k | strong Task 2 lower bound |
| `nsga2` | standard multi-objective search | shared optimizer, not the novelty |

RFamLlama and GenerRNA remain optional proposal/likelihood ablations. They are generic RNA generators, not same-task IRES activity-design baselines.

RFamLlama is the work of Sun, Li, and Deng (2024). This project does not claim RFamLlama or its Rfam-conditioned pretraining method as an original contribution. See [third-party provenance](THIRD_PARTY.md).

## Repository layout

```text
configs/                 frozen baseline and experiment contracts
src/ires_design/         reusable registry, schemas, adapters, and audits
tests/                   dependency-light unit tests
docs/                    scope, protocol, migration, and roadmap
assets/manifests/        small provenance manifests only
runs/                    local outputs; ignored by Git
paper/                   venue-neutral manuscript, tables, and vector figures
```

Model weights, datasets, generated libraries, external repositories, and experiment logs are never committed. They live under ignored `assets/` and `runs/` paths and are referenced by manifests.

## Quick start

Python 3.10 or newer is required.

Route all temporary files and framework caches to the project's personal
storage before running commands.  Do not use the compute nodes' shared
`/tmp`:

```bash
source scripts/ires_project_env.sh
```

The portable default scratch root is `$HOME/.cache/ires-design`. On a cluster,
set `IRES_SCRATCH_ROOT` to a personal high-capacity location before sourcing
the script.

```bash
python -m pip install -e .
ires-design baselines list
ires-design baselines check
ires-design validate-experiment configs/experiments/smoke.json
python -m unittest discover -s tests -v
```

Audit any candidate FASTA before running expensive scorers:

```bash
ires-design audit-fasta candidates.fasta --expected-length 174
```

Normalize an external FASTA or CSV into the shared JSONL candidate schema:

```bash
ires-design normalize-output input.fasta output.jsonl \
  --method official_rfamllama --task de_novo --seed 42
```

Read the [literature and baseline audit](docs/LITERATURE_AUDIT_2026-08-21.md), [baseline setup](docs/BASELINES.md), and [frozen experiment protocol](docs/EXPERIMENT_PROTOCOL.md) before adding model-specific code.

Run the lightweight identification controls on reconstructed nested folds:

```bash
python scripts/run_prediction_benchmark.py \
  --dataset /path/to/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv.zip \
  --output-dir runs/prediction_pilot_s42 \
  --models composition,kmer --protocol reconstructed_nested10 --seed 42
```

The released random-split semantics, reconstructed-fold policy, and first reviewed pilot are in
[the split audit](docs/DATA_SPLIT_AUDIT_2026-08-21.md). The manuscript workspace is documented in
[paper/README.md](paper/README.md).

## Reproducibility contract

Every formal run records:

- dataset and split manifest hashes;
- model/checkpoint source and hash;
- baseline ID and adapter version;
- seed, candidate count, decoding/search parameters, and oracle-query budget;
- scorer versions and whether each scorer is used for optimization or evaluation;
- raw candidates, validity/survival counts, and complete per-sequence results.

The optimization scorer is never presented as the sole final judge. Every functional score carries its assay context. IRES-DM released outputs are not described as a strict budget-matched comparison unless the model is retrained under the shared protocol.

## Publication and patent note

Keep the GitHub repository **private until the intended patent application has been filed and the disclosure order has been confirmed with the university/patent representative**. Public code, issues, releases, and preprints can constitute public disclosure. This note is project hygiene, not legal advice.

## License

The new project code is released under Apache License 2.0. External repositories, datasets, checkpoints, and model weights retain their own licenses and are not redistributed here.
