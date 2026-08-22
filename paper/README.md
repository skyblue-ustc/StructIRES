# Manuscript workspace

`main.tex` is a venue-neutral scientific draft. Its section files are the source of truth; a future
IEEE or OUP wrapper should import them instead of duplicating text.

Current reviewed figure/table contract:

1. `fig1_overview`: assay-qualified evidence, shared candidate pool, StructIRES-Rank objectives,
   and evaluation route.
2. `fig2_prediction_pilot`: dataset shortcut audit and reconstructed-fold lightweight baseline.
3. `fig3_assay_shift`: cross-assay transfer, calibration, and failure analysis.
4. `fig4_checkpoint_stability`: final ten-checkpoint native/transfer heterogeneity audit.
5. `fig6_structires_ireslm_ablation`: matched five-arm IRES-LM, energy, ensemble, anchor and
   combined-constraint ablation.
6. `fig7_iapv_secondary_structure`: provenance-locked IAPV parent / score-only / StructIRES
   ViennaRNA secondary-structure case plus the public MPRA edit-risk guardrail.
7. `iapv_mpra_guardrail`: IAPV-only public direct-RNA mutational-scan guardrail table.

Tables must identify the assay, split, tuning data, and whether a method is reproduced or copied as
a published reference. Placeholder cells use `--`; they are not results.

## Build the reviewed draft

The current server has a working `ires-tex` environment. From this directory:

```bash
mkdir -p build
conda run -n ires-tex tectonic -X compile main.tex \
  --outdir build --keep-logs --keep-intermediates
```

The expected artifact is `build/main.pdf`. It is a venue-neutral internal-review draft; author,
contact, disclosure and final venue-wrapper fields intentionally remain pending.

## Main StructIRES evidence chain

The primary manuscript claim is computational only and is backed by immutable external run outputs:

1. `ireslm_ensemble_shared_pool_scores_v1_20260822`: all 10 RNA-FM and 10 UTR-LM released
   classifiers score the same 15,360 frozen candidates.
2. `parent_ensemble_anchor_metrics_v1_20260822`: ViennaRNA 2.7.2 parent-derived $P\geq0.5$
   ensemble-anchor retention for every candidate.
3. `structires_ireslm_anchor_selection_v1_20260822`: matched five-arm top-50 selection from every
   512-candidate parent-by-run pool.
4. `structires_ireslm_anchor_direct_rna_s3_v2_20260822`: independent S3-held-out direct-RNA
   computational proxy evaluation; it is not a candidate activity assay.
5. `iapv_mpra_mutational_risk_v1_20260822` and
   `structires_ireslm_iapv_mpra_guardrail_v2_20260822`: IAPV-only public direct-RNA mutational-scan
   edit-risk guardrail and its same-budget ablation.

Every run directory contains a manifest with paths, hashes and claim scope. These external assets
are intentionally not committed to Git.

Summarize all released RNA-FM native folds without pooling overlapping test memberships:

```bash
python scripts/summarize_rnafm_native_folds.py \
  --run-dir /path/to/released_rnafm_native_fold0_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold1_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold2_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold3_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold4_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold5_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold6_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold7_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold8_v1_20260822 \
  --run-dir /path/to/released_rnafm_native_fold9_v1_20260822 \
  --output-dir /path/to/released_rnafm_native_10fold_summary_v1_20260822 \
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
  --iresfinder-native-run /path/to/iresfinder_released_repeated10_v3_20260822 \
  --iresfinder-cluster-run /path/to/iresfinder_similarity_split_hamming90_len174_v2_20260822 \
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

Audit cross-fold heterogeneity after a multi-checkpoint direct-RNA transfer run:

```bash
PYTHONPATH=scripts python scripts/audit_rnafm_cross_assay_heterogeneity.py \
  --transfer-run /path/to/released_rnafm_cross_assay_10fold_v1_20260822 \
  --output-dir /path/to/released_rnafm_cross_assay_10fold_heterogeneity_audit_v1_20260822 \
  --bootstrap-replicates 2000 \
  --seed 42
```

Generate the final checkpoint-stability figure with:

```bash
python scripts/make_rnafm_checkpoint_audit_figure.py \
  --native-summary /path/to/released_rnafm_native_10fold_summary_v1_20260822 \
  --transfer-audit /path/to/released_rnafm_cross_assay_10fold_heterogeneity_audit_v1_20260822 \
  --output paper/figures/fig4_checkpoint_stability.pdf \
  --png-output paper/figures/fig4_checkpoint_stability.png
```

Regenerate the provenance-checked complete metric ledger and LaTeX table with:

```bash
python scripts/build_unified_result_ledger.py \
  --config configs/benchmark_ledger.json \
  --external-run-root /path/to/ires-design-external-runs \
  --output-csv paper/data/benchmark_results.csv \
  --output-tex paper/tables/unified_results.tex
```
