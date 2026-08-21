# Project scope and claim boundary

## Scientific question

Can structure- and energy-aware multi-objective constraints improve computational IRES candidate enrichment across more than one public RNA generator, compared with function-only optimization under the same candidate and oracle-query budget?

## Task A: de novo candidate design

- Input: desired length 174 nt, no natural seed.
- Output: a batch of canonical candidate records plus the complete raw candidate pool.
- Controlled backbones: Official RFamLlama and GenerRNA.
- External reference: released IRES-DM sequences.
- Primary comparison: paired `full` versus `function_only` within each controlled backbone.

## Task B: mutation design

- Input: a frozen set of at least ten IRES seeds, including VCIP, EMCV, and CVB3.
- Output: mutants and complete parent/edit lineage.
- Baselines: uniform random mutation, LM-likelihood-guided mutation, NSGA-II, and IRES-EA.
- Fairness controls: same seed, maximum edit distance, candidate count, and oracle-query budget.

## Proposed layer

The project does not claim a new foundation model or a new general-purpose optimizer. It combines:

1. a locked, cross-fitted hurdle function model for floor-heavy MPRA measurements;
2. function-only optimization as the attack/control condition;
3. structure and energy objectives calibrated to measured high-IRES reference distributions;
4. standard NSGA-II/Pareto search;
5. uncertainty, novelty, and diversity-aware batch selection;
6. independent functional and structural evaluation.

## Claims that are not allowed

- `We propose RFamLlama` or `our RFamLlama`.
- First RNA LM, first IRES generator, first structure-aware RNA generation, or first RNA DPO.
- IRES-DM released-output results as a strict same-data comparison.
- A proxy-score increase as experimentally verified IRES activity.
- Universal translation improvement beyond the reporter and assay context.

