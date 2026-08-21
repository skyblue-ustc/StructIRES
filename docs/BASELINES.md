# Baseline setup

`configs/baselines.json` is the single source of truth. Every table, run manifest, and adapter uses the canonical ID from that registry.

## Roles

### Paired public backbones

`official_rfamllama` and `generna` receive the same outer candidate/search protocol. Their most important result is the paired change from `function_only` to `full`, not an unqualified cross-model ranking.

### IRES-specific methods

- `ires_ea`: strict Task B baseline when the upstream pipeline, seed set, edit constraints, and budget are aligned.
- `ires_dm`: external Task A reference until a same-data retraining is complete.

### Standard controls

- `random_mutation`: built in and executable now.
- `lm_likelihood_mutation`: reproduces the general strategy of ranking functional mutations with an RNA LM.
- `nsga2`: shared multi-objective optimizer. Its algorithm definition must not change between backbones.
- `natural_iresbase`: distribution reference, never a generated method.

### Supplement only

`local_rfam_ar_reproduction` is a rough local reproduction of RFamLlama-style pretraining. It enters a formal table only after checkpoint, data, tokenizer, code-source, and license manifests are complete.

## Asset conventions

The registry resolves an asset in this order:

1. the baseline-specific environment variable;
2. `default_asset_path` relative to repository root.

Example:

```bash
export IRES_RFAMLLAMA_MODEL=/path/to/RFamLlama-base
export IRES_GENERNA_MODEL=/path/to/GenerRNA
export IRES_EA_ROOT=/path/to/IRES_Prediction_Design
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

