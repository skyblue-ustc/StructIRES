# Main-text scope and remaining-work audit

Last reviewed: 2026-08-22 (Asia/Shanghai)

## Decision

The initial paper should make one computational claim only: **under a matched,
full-length seeded-design budget, StructIRES-Rank selects candidates with better
thermodynamic and ensemble-structural preservation than score-only selection, while
its post-selection direct-RNA proxy and an IAPV-specific MPRA edit-risk guardrail
provide independent, assay-qualified checks.**  It does not claim measured activity
of new candidates.

## Main text: frozen evidence

| Evidence block | Result retained in main text | Artifact / run | Why it belongs in the paper |
|---|---|---|---|
| Assay audit | Released RNA-FM, UTR-LM and IRESfinder native reruns; direct-RNA transfer is weak/miscalibrated | `paper/data/benchmark_results.csv`; external release-rerun manifests | Establishes why a legacy score is a proposal term rather than a final activity oracle. |
| Matched constraint ablation | Five arms, same 512 candidates per parent-by-pool unit; StructIRES-Rank has the lowest pairing-profile distance (0.037 +/- 0.010) and highest anchor retention (0.955 +/- 0.015) | `structires_ireslm_anchor_selection_v1_20260822` | Main method evidence; same budget and shared candidates prevent a query-budget explanation. |
| Independent proxy | S3-held-out direct-RNA proxy: StructIRES-Rank 0.435 vs IRES-LM score-only 0.419; paired difference +0.0158, bootstrap interval 0.0079--0.0242 | `structires_ireslm_anchor_direct_rna_s3_v2_20260822` | Independent post-selection computational evaluation. It is not an activity assay. |
| MPRA guardrail | IAPV-only public mutational map reduces edit risk 2.438 +/- 0.174 to 1.782 +/- 0.087, with an explicit MFE trade-off | `structires_ireslm_iapv_mpra_guardrail_v2_20260822` | Demonstrates how direct-RNA experimental information can be encoded without relabelling generated candidates. |
| Structural case | Deterministic IAPV / seed-42 rank-1 score-only versus StructIRES-Rank example | `structires_iapv_case_v2_20260822` | Makes the averaged preservation effect visually inspectable; explicitly labelled illustrative. |

## Move to supplement / appendix

| Material | Reason |
|---|---|
| Complete unified metric ledger with all published and rerun rows | Essential provenance, but too wide and detailed for the main narrative. |
| Reconstructed-fold shortcut controls and source-holdout details | Supports the assay audit; retain a concise visual summary in main text only. |
| IRESfinder training overlap / detailed component split | Important audit detail, not the central design contribution. |
| Full RNA-FM checkpoint heterogeneity plot and per-fold outputs | Keeps the no-post-hoc-selection safeguard auditable. |
| Reporter-only cargo-context stress test | Computational stress test only; not central direct-RNA evidence. |
| Legacy 174-nt IRES-DM task | Different assay and task; retain as a legacy appendix unless a shared-protocol rerun is completed. |

## Not complete: do not imply otherwise

| Item | State | Required action before it can become a main comparison |
|---|---|---|
| IRES-EA official reproduction | Blocked: released repository lacks the versioned RNA-FM / UTR-LM predictor wrappers and model directory it calls | Obtain author-released missing assets, hash them, then rerun under the matched protocol. |
| IRES-DM de-novo / 174-nt shared-protocol baseline | Not run | Freeze task-specific data and retrain/reproduce without borrowing a reported number. |
| DeepCIP official inference | Time-boxed environment / archive-compatibility blocker | Use an isolated, verified legacy environment and released checkpoint, or leave as a data-overlap audit only. |
| New project-owned RNA foundation-model pretraining | Not a validated contribution | Complete data manifest, architecture, pretraining and held-out evaluation before naming it as a contribution. |
| Experimental confirmation of generated candidates | Not performed | No claim of increased IRES activity is permitted without appropriate independent measurement. |

## Figure/table contract for this draft

Main text should contain four figures and three compact tables:

1. **Figure 1 — StructIRES-Rank overview:** evidence, same-budget candidate pool, rank objectives, assay-qualified evaluation.
2. **Figure 2 — Assay audit:** concise native-to-direct-RNA transfer summary; details in supplement.
3. **Figure 3 — Matched constraint ablation:** energy, ensemble, anchor and held-out proxy outcomes.
4. **Figure 4 — IAPV structural / MPRA case:** parent, score-only and constrained MFE structures plus the position-risk guardrail.
5. **Table 1:** data and assay roles.
6. **Table 2:** compact released-baseline / transfer summary.
7. **Table 3:** matched StructIRES-Rank ablation; IAPV guardrail may be a small adjacent table or a figure callout.

The wide unified ledger, similarity controls and repeated-holdout details stay accessible as
supplementary provenance rather than being discarded.
