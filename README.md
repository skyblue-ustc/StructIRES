# IRES Design

Reproducible, baseline-first experiments for computational design of internal ribosome entry site (IRES) RNA sequences.

> Status: active research repository. The current claims are computational candidate enrichment under specified reporter/MPRA settings; no generated sequence is claimed to have improved biological activity without experimental validation.

## Scope

This repository has one main story: evaluate established RNA/IRES design backbones under a shared protocol, then measure whether data-calibrated structure and energy constraints improve candidate quality without sacrificing predicted function, novelty, or diversity.

Two tasks are kept separate:

- **Task A — de novo design:** generate batches of 174-nt IRES candidates without a natural seed.
- **Task B — mutation design:** optimize frozen natural or measured IRES seeds under a fixed edit and oracle-query budget.

The proposed layer is deliberately generator-agnostic:

```text
candidate source -> locked function scorers -> structure/energy objectives
                 -> NSGA-II/Pareto search -> diversity-aware selection
                 -> independent evaluation
```

## Baselines

| ID | Role | Main comparison |
|---|---|---|
| `official_rfamllama` | public RNA LM backbone | paired Raw-to-Full ablation |
| `generna` | independent public RNA LM backbone | paired Raw-to-Full ablation |
| `ires_ea` | IRES-specific mutation design | Task B baseline |
| `ires_dm` | IRES-specific diffusion design | released-output external reference |
| `lm_likelihood_mutation` | general LM mutation ranking | Task B baseline |
| `random_mutation` | non-neural lower bound | Task B baseline |
| `nsga2` | standard multi-objective search | shared optimizer/control |
| `natural_iresbase` | experimentally supported reference | distribution anchor, not a trainable model |

RFamLlama is the work of Sun, Li, and Deng (2024). This project does not claim RFamLlama or its Rfam-conditioned pretraining method as an original contribution. See [third-party provenance](THIRD_PARTY.md).

## Repository layout

```text
configs/                 frozen baseline and experiment contracts
src/ires_design/         reusable registry, schemas, adapters, and audits
tests/                   dependency-light unit tests
docs/                    scope, protocol, migration, and roadmap
assets/manifests/        small provenance manifests only
runs/                    local outputs; ignored by Git
```

Model weights, datasets, generated libraries, external repositories, and experiment logs are never committed. They live under ignored `assets/` and `runs/` paths and are referenced by manifests.

## Quick start

Python 3.10 or newer is required.

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

See [baseline setup](docs/BASELINES.md) and the [frozen experiment protocol](docs/EXPERIMENT_PROTOCOL.md) before adding model-specific code.

## Reproducibility contract

Every formal run records:

- dataset and split manifest hashes;
- model/checkpoint source and hash;
- baseline ID and adapter version;
- seed, candidate count, decoding/search parameters, and oracle-query budget;
- scorer versions and whether each scorer is used for optimization or evaluation;
- raw candidates, validity/survival counts, and complete per-sequence results.

The optimization scorer is never presented as the sole final judge. IRES-DM released outputs are not described as a strict budget-matched comparison unless the model is retrained under the shared protocol.

## Publication and patent note

Keep the GitHub repository **private until the intended patent application has been filed and the disclosure order has been confirmed with the university/patent representative**. Public code, issues, releases, and preprints can constitute public disclosure. This note is project hygiene, not legal advice.

## License

The new project code is released under Apache License 2.0. External repositories, datasets, checkpoints, and model weights retain their own licenses and are not redistributed here.

