# Baseline setup

`configs/baselines.json` is the single source of truth. Every table, run manifest, and adapter uses the canonical ID from that registry.

## Roles

### Direct IRES baselines

- `ires_ea`: exact mutation-design baseline from IRES-AI. It is strict only when seeds, editable positions, edit limit, population and oracle-query budget are matched.
- `ires_dm`: released exact-task de novo reference. It is always external because its generation/training budget is not controlled here.
- `ires_dm_retrained`: same-data and same-protocol IRES-DM reproduction. This is required before any strict de novo superiority claim.
- `structure_only_mutation`: independent implementation of the Delli-Ponti structure-identity, paired-fraction and MFE rule.

### Required search controls

- `random_mutation`: non-neural seeded lower bound, executable now.
- `random_screen`: large random proposal pool followed by top-k scoring. PARADE shows why this is a strong control for short UTRs.
- `lm_likelihood_mutation`: GARNET-style mutation ranking by RNA-LM likelihood difference.
- `score_only_ga`: primary attack/control condition. It maximizes the same frozen function score as the proposed method without structure, context or uncertainty objectives.
- `nsga2`: shared standard optimizer. The contribution is the objective formulation and evidence, not NSGA-II.

### Optional proposal backbones

`official_rfamllama` and `generna` are generic RNA generators/likelihood models. They can enter proposal or naturalness ablations, but never the headline same-task IRES baseline table.

`local_rfam_ar_reproduction` is historical supplementary evidence only. It enters a formal table only after checkpoint, data, tokenizer, code-source and license manifests are complete.

### References and evaluators

`natural_iresbase` is a provenance-labelled distribution reference, not a generated method and not a uniformly validated gold standard. Evaluators live in `configs/evaluators.json`, separate from candidate-generating baselines.

## Asset conventions

The registry resolves an asset in this order:

1. the baseline-specific environment variable;
2. `default_asset_path` relative to repository root.

Example:

```bash
export IRES_RFAMLLAMA_MODEL=/path/to/RFamLlama-base
export IRES_GENERNA_MODEL=/path/to/GenerRNA
export IRES_EA_ROOT=/path/to/IRES_Prediction_Design
export IRES_DM_ROOT=/path/to/IRES_Prediction_Design
export IRES_DM_RELEASED=/path/to/ires_dm_release
export IRESBASE_FASTA=/path/to/All_IRES.fa
ires-design baselines check
```

`adapter_pending` means that the scientific role is registered but the model-specific execution adapter is not yet implemented. Such a baseline must not appear as completed in a result table.

## Required provenance

For each external baseline record:

- paper and source URL;
- source commit/tag and download date;
- code and model-weight licenses;
- checkpoint/file hash;
- tokenizer and sequence convention;
- exact command and hardware;
- whether it was zero-shot, adapted, retrained, or only re-evaluated.

For each function scorer, also record the assay context (`DNA/lentiviral`, `circRNA plasmid`, `direct RNA`, or other), source overlap and whether it was exposed during optimization.
