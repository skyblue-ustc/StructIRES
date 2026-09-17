# StructIRES × IEEE BIBE 2026: submission-readiness checklist

Last checked: 2026-08-31.  Target venue: the 26th IEEE International
Conference on Bioinformatics and Bioengineering (BIBE 2026), Shanghai,
27--29 November 2026.

The official conference call currently requests original PDF submissions in
IEEE double-column, single-spaced, 10-point format and describes submissions
as ``up to 8 pages''.  The same site also contains a legacy-looking statement
about five regular pages plus one paid extra page; therefore, **do not assume
that seven pages is billable or acceptable until the organizers confirm the
current rule**.  Its current stated full-paper deadline is **31 October 2026**.
Recheck the official CFP and submission system immediately before upload:

- https://bibeconference.com/
- https://easychair.org/cfp/BIBE2026

## Current manuscript state

- [x] IEEE conference template compiles locally with Tectonic.
- [x] Current PDF is seven pages: `paper/bibe2026/build/main.pdf`.
- [ ] Confirm the final page-charge/page-limit rule; prepare a five- or
  six-page compact build if required.
- [x] Clear task definition: legacy-reporter IRES-like classification and
  parent-relative constrained local variant selection are separated.
- [x] Main rank-selection table is based on fixed, matched candidate pools.
- [x] Structure case, rank ablation, classifier overview, study overview, and
  paired classifier result figures are in the manuscript.
- [x] Claims explicitly exclude experimental validation of new candidates.
- [~] Classification main table currently contains the independently verified
  three-fold matched re-training; fixed-order completion of the remaining
  native folds is running before any ten-fold wording is added.

## Evidence required before a submission build

1. **Recognition benchmark.** Freeze a result ledger containing public
   baseline reruns and the completed fixed-fold sequence-only/contact-fusion
   comparison.  Update the aggregate only from independently recomputed
   prediction ledgers.  Report AUROC, AUPR, F1, MCC, and ECE; do not omit the
   higher ECE of contact fusion.
2. **Design benchmark.** Retain the existing shared-pool five-arm ablation:
   score-only, energy, ensemble, anchor, and combined rank.  All values must
   link to the immutable run directory and generator script.
3. **Assay boundary.** Keep the direct-RNA 3--6-mer result as a held-out
   computational proxy and IAPV as a parent-specific edit-risk guardrail.
   Neither is a new-candidate activity measurement.
4. **Submission hygiene.** Replace placeholder author information, confirm
   affiliations and acknowledgements, remove internal filesystem paths from
   the final PDF, and run the conference PDF checker if supplied.
5. **Reproducibility release.** Include scripts, compact result ledgers,
   figure-generation commands and environment notes; exclude raw data,
   checkpoints, caches, credentials, and large external run artifacts.

## Decision rule for the next classifier result

- If the pre-fixed missing native folds retain a non-negative mean change in
  AUROC/AUPR/F1/MCC, report the completed native protocol as a controlled
  legacy-benchmark improvement and keep the identity-aware audit as its
  explicit limitation.
- If the completed mean is mixed or negative, keep the classifier as a
  transparent ablation and make constrained full-length design the paper's
  headline contribution.  Do not select a subset of folds after observing
  results.

## Current local build command

```bash
cd /9950backfile/lant/ires-design/paper/bibe2026
conda run -n ires-tex tectonic -X compile main.tex \
  --outdir build --keep-logs --keep-intermediates
```
