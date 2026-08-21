# Manuscript workspace

`main.tex` is a venue-neutral scientific draft. Its section files are the source of truth; a future
IEEE or OUP wrapper should import them instead of duplicating text.

Current figure/table contract:

1. `fig1_overview`: assay provenance and technical route.
2. `fig2_prediction_pilot`: dataset shortcut audit and reconstructed-fold lightweight baseline.
3. `fig3_assay_shift`: cross-assay transfer, calibration, and failure analysis.
4. `fig4_pareto`: function, ensemble preservation, and cargo-crosstalk Pareto front.
5. `fig5_ablation`: paired score-only/MFE/ensemble/context/robust-full effects.
6. `fig6_validity`: applicability, diversity, efficiency, and representative structures.

Tables must identify the assay, split, tuning data, and whether a method is reproduced or copied as
a published reference. Placeholder cells use `--`; they are not results.

Generate the overview figure with:

```bash
python scripts/make_overview_figure.py
```

Generate the reviewed prediction pilot figure with:

```bash
python \
  scripts/make_prediction_pilot_figure.py \
  --run-dir runs/prediction_lightweight_reconstructed_nested10_v2_20260821_s42 \
  --output paper/figures/fig2_prediction_pilot.pdf
```

The host currently has no `pdflatex`, `latexmk`, or `tectonic`; compilation must be checked in a
TeX-enabled environment before any venue wrapper is frozen.
