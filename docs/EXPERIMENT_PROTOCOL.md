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

The main MPRA dataset must use similarity-cluster splitting rather than row-random splitting. The intended contract is 70/15/15 train/validation/test with source and activity-floor stratification. The test assignment is locked before generation.

No test label or test-fitted feature is allowed in:

- model adaptation;
- objective calibration;
- candidate mutation/selection;
- early stopping or variant choice.

## Nested variants

1. `raw`: unoptimized model samples or official released outputs.
2. `function_only`: maximize the frozen optimization-side function scorer.
3. `mfe`: add distance to the high-IRES MFE-per-nt distribution.
4. `structure`: add ensemble and base-pairing distribution objectives.
5. `full`: add uncertainty/applicability, novelty, and batch diversity.

The variants are nested. Changing the decoder, initial population, or budget between variants invalidates the paired ablation.

## Structural interpretation

De novo IRES design does not have a single universal target fold. Therefore Task A uses distributional structure objectives calibrated on measured high-IRES references. Target-specific ensemble defect, Boltzmann probability, or base-pair distance is used only when Task B explicitly preserves a seed or consensus structure.

## Metric hierarchy

1. Independent function: locked MPRA above-floor probability, conditional active score/rank, and public IRES scorer agreement.
2. Structural applicability: MFE-per-nt reference distance, ensemble diversity, base-pairing/profile distance, and pass rate.
3. Validity and anti-shortcut: length, alphabet, GC, homopolymer, low complexity, k-mer drift, applicability survival, and scorer disagreement.
4. Novelty/diversity: nearest-train identity, unique rate, pairwise distance, and cluster coverage.
5. Efficiency: generator calls, oracle calls, wall time, and peak memory.

## Success criterion

The project may claim improved candidate quality only if `full` versus `function_only`:

- improves structural applicability on at least two public backbones;
- does not degrade independent functional evaluation;
- preserves candidate diversity;
- shows directionally consistent results across seeds 42, 43, and 44;
- reports all raw and surviving counts.

If MFE alone matches the full method, remove the unsupported complexity. If structure metrics improve but every independent function scorer declines, report the trade-off rather than claiming higher IRES quality.

