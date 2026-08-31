#!/usr/bin/env bash
#
# Train and evaluate every Shelly scene with line primitives.
#
# The Shelly images must be downloaded and extracted to
# datasets/shelly_data_release/ separately (see README.md).  The seed point
# clouds come from the Fuzzy dataset and are fetched below.
#
# Usage:
#   bash scripts/run_shelly.sh                 # results/<default>_<timestamp>/
#   bash scripts/run_shelly.sh my_run_name     # results/my_run_name/
#
# Output:
#   results/<run>/
#     summary.json          averaged PSNR/SSIM/LPIPS and per-scene timing
#     <scene>/
#       options.json  timing.json  loss.txt  test_mae.txt
#       shelly_<scene>_07000.npz  shelly_<scene>_50000.npz
#       eval_50000/     metrics.json, per_view.json, gt/, renders/
#
# GPU selection: the renderer (Vulkan) and torch (CUDA) must be pinned to the
# same physical GPU.  On a multi-GPU machine the two enumerations can differ,
# so set both explicitly:
#
#   export FUZZYDR_DEVICE_INDEX=1   # Vulkan physical-device index
#   export CUDA_VISIBLE_DEVICES=0   # torch sees that GPU as cuda:0
#
# A mismatched pair fails at the first backward pass with
# "cudaImportExternalSemaphore failed: invalid device ordinal".
# Both may be left unset on a single-GPU machine.
#
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENES=(fernvase horse khady kitten pug woolly)
EVAL_ITERS=(50000)

python scripts/fetch_data.py train || exit 1

RUN_NAME="${1:-shelly_lines_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="results/${RUN_NAME}"
# Never overwrite an existing run: refuse if the target exists and is non-empty.
if [ -d "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${OUT_DIR} already exists and is non-empty - refusing to overwrite."
    echo "       Pass a different run name: bash scripts/run_shelly.sh <run_name>"
    exit 1
fi
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Run: ${RUN_NAME}   (line primitives)"
echo "Output: ${OUT_DIR}"
echo "Scenes: ${SCENES[*]}"
echo "Device: torch cuda:0 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: all GPUs visible>})"
echo "============================================================"

t_total0=$(date +%s)

for scene in "${SCENES[@]}"; do
    SCENE_DIR="${OUT_DIR}/${scene}"
    mkdir -p "${SCENE_DIR}"

    echo ""
    echo "============================================================"
    echo "  Training: ${scene}"
    echo "============================================================"

    python scripts/train_shelly_lines.py \
        --scene "${scene}" \
        --device cuda --gpu_id 0 \
        --out_dir "${OUT_DIR}"

    for eval_it in "${EVAL_ITERS[@]}"; do
        CKPT="${SCENE_DIR}/shelly_${scene}_$(printf '%05d' ${eval_it}).npz"
        EVAL_DIR="${SCENE_DIR}/eval_${eval_it}"

        if [ ! -f "${CKPT}" ]; then
            echo "WARNING: ${CKPT} not found, skipping eval at iter ${eval_it}"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "  Eval: ${scene} @ iter ${eval_it}"
        echo "------------------------------------------------------------"

        python scripts/eval.py \
            --ckpt "${CKPT}" \
            --device cuda --gpu_id 0 \
            --eval_dir "${EVAL_DIR}"
    done
done

# ---- Summary: per-scene timing + metrics into one JSON --------------------
echo ""
echo "Collecting summary ..."

python3 -c "
import json, os

out_dir = '${OUT_DIR}'
scenes = '${SCENES[*]}'.split()
eval_iters = [$(IFS=,; echo "${EVAL_ITERS[*]}")]
summary = {}

for scene in scenes:
    scene_dir = os.path.join(out_dir, scene)
    entry = {}

    timing_path = os.path.join(scene_dir, 'timing.json')
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            entry['timing'] = json.load(f)

    for it in eval_iters:
        metrics_path = os.path.join(scene_dir, f'eval_{it}', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                entry[f'metrics_{it}'] = json.load(f)

    summary[scene] = entry

for it in eval_iters:
    key = f'metrics_{it}'
    vals = [s[key] for s in summary.values() if key in s]
    if vals:
        avg = {m: sum(v[m] for v in vals) / len(vals) for m in vals[0]}
        summary[f'average_{it}'] = avg
        print(f'  @{it}: PSNR={avg[\"PSNR\"]:.4f}  SSIM={avg[\"SSIM\"]:.4f}  LPIPS={avg[\"LPIPS\"]:.4f}  ({len(vals)} scenes)')

summary_path = os.path.join(out_dir, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved: {summary_path}')
"

t_total1=$(date +%s)
elapsed=$(( t_total1 - t_total0 ))

echo ""
echo "============================================================"
echo "All done in $((elapsed / 60))m $((elapsed % 60))s.  Results: ${OUT_DIR}"
echo "============================================================"
