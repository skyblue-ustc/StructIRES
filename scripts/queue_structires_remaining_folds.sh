#!/usr/bin/env bash
# Continue a fixed StructIRES native-fold schedule without overwriting outputs.
#
# This launcher is deliberately conservative: an existing output directory is
# treated as an externally launched fixed fold and is skipped; a missing or
# failed run then makes the final paired summary fail under `set -e`.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <sequence-wait-pid> <contact-wait-pid> <remote-host> <remote-pids-csv>" >&2
  exit 2
fi

sequence_wait_pid="$1"
contact_wait_pid="$2"
remote_host="$3"
remote_pids_csv="$4"

root=/9950backfile/lant/ires-design
external=/9950backfile/lant/data/ires-design-external-runs
python=/9950backfile/lant/miniconda3/envs/rfamllama/bin/python
dataset=/9950backfile/lant/RFamLlama_legacy/IRES_Prediction_Design/Data/v2_dataset_with_unified_stratified_shuffle_train_test_split.csv.zip
rnafm_base=/9950backfile/lant/data/ires-design-external-assets/rnafm_torch_cache/hub/checkpoints/RNA-FM_pretrained.pth
upstream_fm=/9950backfile/lant/RFamLlama_legacy/IRES_Prediction_Design/Script
contact_dir="$external/structires_release_mfe_pair_contacts_1024_v1_20260823"

common=(
  --dataset "$dataset"
  --rnafm-base "$rnafm_base"
  --upstream-fm-dir "$upstream_fm"
  --seed 1337
  --validation-fraction .15
  --epochs 10
  --early-stopping-patience 3
  --tokens-per-batch 4096
  --truncate-num 1024
  --lr 1e-4
  --dropout .5
  --mask-prob .15
  --classification-loss-weight 2
  --mlm-loss-weight 1
  --device cuda
)

wait_for_local_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}

wait_for_remote_pids() {
  local pid
  IFS=',' read -r -a pids <<< "$remote_pids_csv"
  while :; do
    local running=0
    for pid in "${pids[@]}"; do
      if ssh -o BatchMode=yes -o StrictHostKeyChecking=no "lant@$remote_host" "kill -0 $pid 2>/dev/null"; then
        running=1
        break
      fi
    done
    [[ "$running" -eq 0 ]] && return 0
    sleep 60
  done
}

run_sequence_queue() {
  wait_for_local_pid "$sequence_wait_pid"
  for fold in 5 6 7 8 9; do
    local output="$external/structires_native_sequence_authorstyle_batchshuffle_fold${fold}_v2_20260831"
    [[ -e "$output" ]] && continue
    CUDA_VISIBLE_DEVICES=5 "$python" "$root/scripts/train_structires_native_contact_fusion.py" \
      --variant sequence --fold "$fold" --output-dir "$output" "${common[@]}"
  done
}

run_contact_queue() {
  wait_for_local_pid "$contact_wait_pid"
  for fold in 3 5 6 7 8 9; do
    local output="$external/structires_native_contact_authorstyle_batchshuffle_fold${fold}_v2_20260831"
    [[ -e "$output" ]] && continue
    CUDA_VISIBLE_DEVICES=6 "$python" "$root/scripts/train_structires_native_contact_fusion.py" \
      --variant contact --contact-dir "$contact_dir" --fold "$fold" --output-dir "$output" "${common[@]}"
  done
}

run_sequence_queue > /tmp/structires_native_sequence_queue_skip_existing_20260831.log 2>&1 &
sequence_queue_pid=$!
run_contact_queue > /tmp/structires_native_contact_queue_skip_existing_20260831.log 2>&1 &
contact_queue_pid=$!

wait "$sequence_queue_pid"
wait "$contact_queue_pid"
wait_for_remote_pids

for fold in 1 3 5 6 7 8 9; do
  sequence_run="$external/structires_native_sequence_authorstyle_batchshuffle_fold${fold}_v2_20260831"
  contact_run="$external/structires_native_contact_authorstyle_batchshuffle_fold${fold}_v2_20260831"
  pair_output="$external/structires_native_authorstyle_batchshuffle_pair_fold${fold}_v2_20260831"
  [[ ! -e "$pair_output" ]]
  "$python" "$root/scripts/summarize_native_structires_pair.py" \
    --sequence-run "$sequence_run" --contact-run "$contact_run" --output-dir "$pair_output"
done

"$python" "$root/scripts/summarize_native_structires_multifold.py" \
  --pair-dirs \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold0_v2_20260823" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold1_v2_20260831" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold2_v2_20260823" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold3_v2_20260831" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold4_v2_20260823" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold5_v2_20260831" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold6_v2_20260831" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold7_v2_20260831" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold8_v2_20260831" \
  "$external/structires_native_authorstyle_batchshuffle_pair_fold9_v2_20260831" \
  --output-dir "$external/structires_native_authorstyle_batchshuffle_multifold_0_9_v2_20260831"
