# StructIRES mainline — locked working plan

## Research question

Given a candidate or experimentally supported parent IRES segment, can a
sequence scorer and an explicit RNA secondary-structure objective jointly
prioritize **computationally robust, minimally edited variants**?  We do not
claim measured translation improvement without a new matched experimental
assay.

## Method identity

`StructIRES` has two connected components.

1. **StructIRES-Classifier** scores a candidate IRES segment from a sequence
   representation plus RNA thermodynamic/ensemble features.  Recognition is
   evaluated separately from design on locked labels.
2. **StructIRES-EA** is a budget-matched iterative mutation search.  It uses
   the frozen classifier score together with explicit energy and parent-
   relative ensemble preservation to select the next generation.

`Rank` is the one-generation special case of the second component.  It is not
the complete method claim.

## Explicit objectives

For a parent sequence \(x_0\) and a candidate \(x\), the design fitness is

\[
F(x) = S_{\mathrm{StructIRES}}(x)
 - \lambda_E |\Delta \mathrm{MFE}(x,x_0)|
 - \lambda_P D_{\mathrm{BPP}}(x,x_0)
 + \lambda_A R_{\mathrm{anchor}}(x,x_0)
 - \lambda_M R_{\mathrm{MPRA}}(x).
\]

- \(S_{\mathrm{StructIRES}}\): validation-locked IRES score.
- \(|\Delta\mathrm{MFE}|\): global thermodynamic deviation.
- \(D_{\mathrm{BPP}}\): ensemble pairing-profile displacement.
- \(R_{\mathrm{anchor}}\): retention of high-confidence *parent-derived*
  ensemble pairs.  These are not experimentally mapped universal motifs.
- \(R_{\mathrm{MPRA}}\): optional public mutational-risk guardrail, used only
  where a matched assay map exists; it is not a label for generated candidates.

Edit distance is a fixed search-budget control, not a biological objective.

## Evidence and gates

| Block | Comparison | Status / decision rule |
|---|---|---|
| Recognition reference | released RNA-FM, UTR-LM, IRESfinder, published IRES-LM/DeepCIP/IRESpy | complete; native release protocol remains reproduction-only |
| Recognition gate | sequence-only, structure-only, sequence+structure under frozen 90%-identity 70/15/15 split | active; validation selects threshold only |
| Foundation fusion | frozen foundation-model representation + thermodynamic adapter | run only if the lightweight feature audit establishes an interpretable baseline and exact feature provenance |
| Design baseline | score-only iterative EA | required; same parents, mutation operator, population, generations, edit ceiling and oracle budget |
| Proposed design | StructIRES-EA | required paired comparison against score-only EA |
| Structural ablation | score-only, +MFE, +BPP ensemble, +anchors, +MPRA guardrail where applicable | required; nested objectives only |
| Independent proxy | direct-RNA IRES-TrAPPr held-out scoring | evaluator only; never use its labels for mutation selection or checkpoint choice |

## Locked reporting boundaries

- The 174-nt legacy classification set and full-length parent-relative design
  panel are distinct tasks; their results never share a headline claim.
- The released IRES-LM scripts select checkpoints with test AUPR.  Their
  native results are faithful reproductions, not blind estimates.
- A classifier is not called improved unless its held-out metrics and run
  manifest exist.  A design candidate is not called active without a matched
  experimental measurement.
- If structural features fail to improve classification, retain them only as
  design constraints and report that distinction clearly.

## Paper progression

1. Establish comparable IRES recognition references and the assay-shift
   boundary.
2. Introduce the validation-clean StructIRES scorer and ablation table.
3. Show score-only EA versus StructIRES-EA with matched design budgets.
4. Show energy/BPP/anchor retention, MPRA risk where available, and an
   illustrative secondary-structure case.
5. Describe direct-RNA evaluation as an independent computational proxy, not
   experimental confirmation of designed variants.
