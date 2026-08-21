# Project scope and claim boundary

## Scientific question

Can assay-aware, ensemble-structure- and cargo-context-constrained optimization produce computationally more robust variants of experimentally supported viral IRESes than score-only and structure-only methods under the same mutation and oracle-query budget?

## Task 0: assay-shift audit

- Input: assay-labelled DNA/lentiviral MPRA data and RNA-based IRES-TrAPPr data.
- Output: frozen similarity/family splits, calibration results, transfer metrics and failure examples.
- Purpose: determine whether the old 174-nt oracle can safely guide any later design step.

## Task 1: primary seeded design

- Input: a frozen, type-balanced set of full-length experimentally supported viral IRES seeds, initially including IAPV, HCV, CrPV and SV-A.
- Output: mutants and complete parent/edit lineage.
- Baselines: uniform random mutation, LM-likelihood-guided mutation, Delli-Ponti-style structure-only mutation, score-only GA and IRES-EA.
- Primary comparison: `robust_full` versus `score_only_ga` with identical seeds, initial populations, edit limits, candidate counts and oracle-query budgets.

## Task 2: secondary legacy de novo design

- Input: 174-nt, seed-free design under the 2016 DNA/lentiviral reporter context.
- Direct baseline: IRES-DM, with released output labelled external and same-data retraining labelled strict.
- Required controls: random screening and score-only GA.
- Optional proposals: RFamLlama and GenerRNA, which are generic RNA models rather than same-task IRES methods.
- Claim boundary: reporter-context candidate enrichment only.

## Proposed layer

The project does not claim a new foundation model or a new general-purpose optimizer. It combines:

1. explicit assay provenance and a mandatory cross-assay audit;
2. conservative aggregation of independently trained, assay-labelled function scorers;
3. seed/consensus structural-ensemble preservation beyond MFE;
4. IRES–cargo crosstalk and context-specific structure consistency;
5. standard NSGA-II/Pareto search with score-only and structure-only controls;
6. uncertainty, applicability, novelty and diversity-aware batch selection;
7. independent functional and structural evaluation.

## Claims that are not allowed

- `We propose RFamLlama` or `our RFamLlama`.
- First RNA LM, first IRES generator, first structure-aware RNA generation, or first RNA DPO.
- MFE or structure preservation alone as the central novelty.
- IRES-DM released-output results as a strict same-data comparison.
- A DNA/lentiviral reporter score as assay-independent IRES activity.
- A proxy-score increase as experimentally verified IRES activity.
- Universal translation improvement beyond the reporter and assay context.
