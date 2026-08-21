# Frozen experiment protocol

## Shared candidate schema

All methods emit JSONL matching `CandidateRecord`:

- `candidate_id`
- `sequence` in canonical RNA alphabet A/C/G/U
- `method` using a registry baseline ID or registered project variant
- `task`: `de_novo` or `mutation`
- `seed`
- `parent_id` for mutation experiments
- `source_path`
- free-form `metadata`, including edit positions or upstream IDs

External DNA-like A/C/G/T outputs are normalized to RNA by replacing T with U. Original files remain immutable.

## Data split

Every sequence record carries `assay_context`, `source_study`, `construct_context`, `cell_type`, `length`, `family/type`, and evidence level where available. DNA/lentiviral and direct-RNA measurements are never pooled into an unlabeled target.

Each source uses similarity-cluster splitting rather than row-random splitting. The default contract is 70/15/15 train/validation/test with source, family/type and activity-floor stratification. A family-held-out view is mandatory for full-length viral IRESes. Test assignments are locked before generation.

No test label or test-fitted feature is allowed in:

- model adaptation;
- objective calibration;
- candidate mutation/selection;
- early stopping or variant choice.

## Nested variants

1. `raw`: unoptimized samples, mutation pool or official released outputs.
2. `score_only`: maximize the frozen optimization-side function scorer.
3. `mfe`: add only an MFE/paired-fraction rule, matching the simplest published structure baseline.
4. `ensemble`: add base-pair-probability/ensemble preservation relative to the seed or consensus.
5. `context`: add IRES–cargo crosstalk and context-specific structure consistency.
6. `robust_full`: add conservative multi-scorer aggregation, applicability/uncertainty, novelty and batch diversity.

The variants are nested. Changing the decoder, initial population, or budget between variants invalidates the paired ablation.

## Structural interpretation

The primary seeded task preserves a seed/consensus structural ensemble. Record MFE, partition-function, base-pair-probability and ensemble-defect metrics separately. MFE alone is never described as structural fidelity.

For each frozen cargo, fold the IRES and context using a pinned convention and compute at least:

- IRES bases paired to cargo divided by IRES-domain length (IRES Crosstalk Ratio);
- IRES positions consistent with the reference/seed ensemble (Structure Consistency);
- per-domain pairing-profile distance.

The secondary 174-nt de novo task has no universal target fold. It uses only distributional structure diagnostics and cannot inherit the full-length seed-preservation claim.

## Metric hierarchy

1. Cross-assay function: direct-RNA held-out rank/calibration, old-reporter score and their disagreement, reported separately.
2. Structural robustness: ensemble defect/probability, base-pairing-profile distance, domain preservation and Albatross audit.
3. Cargo context: Crosstalk Ratio and Structure Consistency for every cargo plus worst-case/mean performance.
4. Validity and anti-shortcut: length, alphabet, GC, homopolymer, low complexity, k-mer drift and applicability survival.
5. Novelty/diversity: edit distance, nearest-train identity, unique rate, pairwise distance and cluster coverage.
6. Efficiency: candidate calls, oracle calls, wall time and peak memory.

## Success criterion

The project may claim improved computational robustness only if `robust_full` versus `score_only`:

- improves ensemble preservation and cargo-context robustness across the frozen seed/cargo panel;
- does not degrade direct-RNA held-out evaluation beyond a predeclared non-inferiority margin;
- preserves candidate diversity;
- shows directionally consistent paired effects across random seeds 42, 43 and 44 and across biological IRES seeds;
- reports all raw and surviving counts.

If MFE alone matches `robust_full`, remove the unsupported complexity. If structural/context metrics improve but direct-RNA evaluation declines, report the trade-off rather than claiming higher IRES quality. No computational result is an experimental activity claim.
