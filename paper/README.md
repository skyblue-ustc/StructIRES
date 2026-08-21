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

Summarize completed released RNA-FM native folds without pooling overlapping test memberships
(the example below shows the first two-fold operational snapshot; formal run IDs remain immutable):

```bash
python scripts/summarize_rnafm_native_folds.py \
  --run-dir /path/to/released_rnafm_native_fold0_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold9_v1_20260822 \
  --output-dir /path/to/released_rnafm_native_available2_summary_v1_20260822 \
  --bootstrap-replicates 2000 \
  --seed 1337
```

Generate the overview figure with:

```bash
python scripts/make_overview_figure.py
```

Generate the reviewed prediction pilot figure with:

```bash
python \
  scripts/make_prediction_pilot_figure.py \
  --run-dir runs/prediction_lightweight_reconstructed_nested10_v2_20260821_s42 \
  --cluster-run runs/prediction_similarity_split_hamming90_len174_v1_20260821_s42_43_44 \
  --iresfinder-native-run /path/to/iresfinder_released_repeated10_v2_20260821 \
  --iresfinder-cluster-run /path/to/iresfinder_similarity_split_hamming90_len174_v1_20260821 \
  --iresfinder-overlap-run /path/to/iresfinder_training_overlap_audit_v1_20260822 \
  --output paper/figures/fig2_prediction_pilot.pdf
```

Generate the frozen lightweight assay-shift statistics, table, and figure with:

```bash
python scripts/make_assay_shift_figure.py \
  --transfer-run runs/cross_assay_transfer_lightweight_v2_20260821 \
  --overlap-run runs/cross_assay_overlap_v2_20260821 \
  --legacy-run runs/prediction_lightweight_reconstructed_nested10_v2_20260821_s42 \
  --output paper/figures/fig3_assay_shift.pdf \
  --table-output paper/tables/assay_shift.tex
```

Run the reviewed 55k-to-`IRESite_exp` source-holdout control with:

```bash
python scripts/run_source_holdout_benchmark.py \
  --dataset /path/to/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv.zip \
  --output-dir runs/prediction_source_holdout_55k_to_iresite_exp_v1_20260821_s42 \
  --train-source 55k \
  --test-source IRESite_exp \
  --models composition,kmer \
  --seed 42
```

The host currently has no `pdflatex`, `latexmk`, or `tectonic`; compilation must be checked in a
TeX-enabled environment before any venue wrapper is frozen.
