#!/usr/bin/env bash
# Finalize paper artifacts only after the fixed ten-fold native aggregate exists.
# The aggregate is produced by the independent paired verifier; this script
# never reads checkpoints or training-time validation metrics.

set -euo pipefail

root=/9950backfile/lant/ires-design
external=/9950backfile/lant/data/ires-design-external-runs
aggregate_dir="$external/structires_native_authorstyle_batchshuffle_multifold_0_9_v2_20260831"
aggregate="$aggregate_dir/aggregate_metrics.csv"
summary="$aggregate_dir/summary.json"
python=/9950backfile/lant/miniconda3/envs/rfamllama/bin/python
conda=/9950backfile/lant/miniconda3/bin/conda

while [[ ! -f "$aggregate" || ! -f "$summary" ]]; do
  sleep 60
done

# Guard against a partial or wrongly addressed aggregate before any manuscript
# artifact is touched.
"$python" - "$summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("n_folds") != 10 or summary.get("folds") != list(range(10)):
    raise SystemExit(f"refusing non-ten-fold aggregate: {summary.get('folds')}")
if summary.get("test_metrics_independently_recomputed") is not True:
    raise SystemExit("aggregate is not independently recomputed")
PY

"$python" "$root/scripts/render_native_classifier_rows.py" \
  --input "$aggregate" \
  --output "$root/paper/bibe2026/tables/recognition_native_rows.tex" \
  --protocol-label "10-fold validation-clean native folds"

"$python" "$root/scripts/make_native_classifier_result_figure.py" \
  --input "$aggregate_dir/per_fold_metrics.csv" \
  --output "$root/paper/figures/fig2_native_classifier_results"

(
  cd "$root/paper/bibe2026"
  mkdir -p build
  "$conda" run -n ires-tex tectonic -X compile main.tex --outdir build --keep-logs --keep-intermediates
)

PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/ires-design-pyc \
PYTHONPATH="$root/src" "$python" -m unittest discover -s "$root/tests" -q

echo "$(date -Is) finalized ten-fold StructIRES paper artifacts from $aggregate" >&2
