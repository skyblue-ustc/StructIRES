# StructIRES initial-draft audit — 2026-08-22

## Scope

This audit checks the deliverables for the first reviewable computational manuscript. It does not
claim experimental IRES activity for generated candidates.

| Requirement | Evidence | Status | Boundary |
|---|---|---|---|
| Complete released IRES-LM baseline | `ireslm_ensemble_shared_pool_scores_v1_20260822/run_manifest.json`; 10 RNA-FM + 10 UTR-LM scores for 15,360 frozen candidates | complete | Released classifiers are a legacy sequence prior, not a direct-RNA activity oracle. |
| Same-budget design ablation | `structires_ireslm_anchor_selection_v1_20260822/selected.csv`; 5 methods × 1,500 selected candidates, 30 parent-by-run units | complete | All arms rank the same 512 candidates per parent and pool. |
| Interpretable structural constraints | `parent_ensemble_anchor_metrics_v1_20260822/run_manifest.json`; MFE, global pairing profile, and parent-derived $P\geq0.5$ anchor metrics | complete | Thermodynamic ensemble features, not experimentally mapped IRES motifs. |
| Direct-RNA functional evaluation | `structires_ireslm_anchor_direct_rna_s3_v2_20260822/run_manifest.json`; S3 held-out 3--6-mer proxy | complete | Computational evaluator only; it was not used in selection. |
| MPRA-derived functional constraint | `iapv_mpra_mutational_risk_v1_20260822/run_manifest.json`; `structires_ireslm_iapv_mpra_guardrail_v2_20260822/iapv_guardrail_manifest.json` | complete, IAPV-only | Public parent mutation evidence protects sensitive edit positions; it does not measure new candidate activity. |
| Traceable manuscript figures/tables | `paper/figures/fig1`--`fig4`, `fig6_structires_ireslm_ablation`, `fig7_iapv_case_v2`; `paper/tables/design_ablation.tex`, `iapv_mpra_guardrail.tex` | complete | Figure 7 is a deterministically selected explanatory case, not a performance estimate. |
| Reviewable build | `paper/build/main.pdf`; Tectonic build and 32 unit tests pass on 2026-08-22 | complete | Venue wrapper is intentionally not frozen. |
| Bibliography and citation resolution | `paper/references.bib`; non-empty `paper/build/main.bbl` | complete | The bibliography is a compact draft set and should expand with venue-specific related-work edits. |

## Remaining before external submission

1. Select a concrete venue and apply its current author/template requirements.
2. Replace anonymous author/contact placeholders and resolve patent/disclosure language with the
   institution before public release.
3. Add a supplementary archive listing every external run manifest and selected-candidate manifest.
4. If stronger biological claims are required, obtain a genuinely independent validation dataset or
   prospective measurements; do not relabel the current proxy or guardrail as validation.

## Reviewer-safe headline

`StructIRES-Rank` is a computational, same-budget constrained rank-selection layer. Relative to
IRES-LM score-only selection, it preserves parent thermodynamic structure and improves a held-out
direct-RNA computational proxy. For IAPV, public direct-RNA MPRA mutational evidence additionally
reduces selection of edits at experimentally sensitive parent positions. These results establish
computational candidate prioritization and constraint trade-offs, not improved translation of new
candidates.
