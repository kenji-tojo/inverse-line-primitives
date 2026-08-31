#!/usr/bin/env bash
#
# Train every scene of the Fuzzy capture dataset with line primitives.
#
# Each scene is optimized over the train and test views combined (--useall);
# the reported PSNR/SSIM is still measured on the held-out test split.
#
# A training video is rendered from one hero camera per scene, selected from
# views/ - the teaser view for cactus1, flowers and kiwi, and <scene>_view1
# (or _view2) from views/gallery/ for the rest.
#
# The training data is fetched from the Fuzzy dataset (4.3 GB) before any
# training starts, so a download failure surfaces immediately rather than
# hours into the run.  Files already on disk are skipped.
#
# Usage:
#   bash scripts/run_fuzzy.sh                          # results/<default>_<timestamp>/
#   bash scripts/run_fuzzy.sh my_run_name              # results/my_run_name/
#   bash scripts/run_fuzzy.sh my_run_name flowers kiwi # only the named scenes
#
# Output:
#   results/<run>/
#     <scene>/
#       options.json  loss.txt
#       fuzzy_<scene>.mp4          optimization video from the hero view
#       fuzzy_<scene>_50000.npz    final checkpoint
#       eval_50000/                metrics.json  train/test/combined metrics
#                                  per_view.json
#                                  renders/, gt_masked/  every train+test view
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[ -n "${FUZZYDR_DEVICE_INDEX:-}" ] && export FUZZYDR_DEVICE_INDEX

# ---- Scene list -----------------------------------------------------------
DEFAULT_SCENES=(cactus1 cactus2 dinosaur flowers fur kiwi tawashi textiles)

python scripts/fetch_data.py train || exit 1

RUN_NAME="${1:-fuzzy_lines_$(date +%Y%m%d_%H%M%S)}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then
    SCENES=("$@")
else
    SCENES=("${DEFAULT_SCENES[@]}")
fi

OUT_DIR="results/${RUN_NAME}"
# Never overwrite an existing run: refuse if the target exists and is non-empty.
if [ -d "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${OUT_DIR} already exists and is non-empty - refusing to overwrite."
    echo "       Pass a different run name: bash scripts/run_fuzzy.sh <run_name>"
    exit 1
fi
mkdir -p "${OUT_DIR}"

# ---- Video hero camera per scene ------------------------------------------
TEASER_DIR="views/teaser"
VIEWS_DIR="views/gallery"
declare -A VIEW_JSON=(
    [cactus1]="${TEASER_DIR}/cactus1.json"
    [cactus2]="${VIEWS_DIR}/cactus2_view1.json"
    [dinosaur]="${VIEWS_DIR}/dinosaur_view2.json"
    [flowers]="${TEASER_DIR}/flowers.json"
    [fur]="${VIEWS_DIR}/fur_view1.json"
    [kiwi]="${TEASER_DIR}/kiwi.json"
    [tawashi]="${VIEWS_DIR}/tawashi_view2.json"
    [textiles]="${VIEWS_DIR}/textiles_view1.json"
)

echo "============================================================"
echo "Run: ${RUN_NAME}   (line primitives)"
echo "Output: ${OUT_DIR}"
echo "Scenes: ${SCENES[*]}"
echo "Device: torch cuda:0 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, FUZZYDR_DEVICE_INDEX=${FUZZYDR_DEVICE_INDEX:-<auto>})"
echo "============================================================"

t_total0=$(date +%s)
declare -a OK_SCENES=()
declare -a FAIL_SCENES=()

for scene in "${SCENES[@]}"; do
    VIEW="${VIEW_JSON[$scene]:-}"

    echo ""
    echo "============================================================"
    echo "  Training: ${scene}    view: ${VIEW:-<none>}"
    echo "============================================================"

    VIEW_ARGS=()
    if [ -n "${VIEW}" ]; then
        VIEW_ARGS=(--view_json "${VIEW}")
    else
        echo "  WARN: no hero view mapped for ${scene}; using default --video_view 0"
    fi

    python scripts/train_fuzzy.py \
        --scene "${scene}" \
        --device cuda \
        --useall \
        --out_dir "${OUT_DIR}" \
        "${VIEW_ARGS[@]}"
    rc=$?

    if [ "${rc}" -eq 0 ]; then
        echo "  OK   ${scene}  (rc=0)"
        OK_SCENES+=("${scene}")
    else
        echo "  FAIL ${scene}  (rc=${rc})"
        FAIL_SCENES+=("${scene}")
    fi
done

t_total1=$(date +%s)
elapsed=$(( t_total1 - t_total0 ))

echo ""
echo "============================================================"
echo "All done in $((elapsed / 60))m $((elapsed % 60))s.  Results: ${OUT_DIR}"
echo "OK   (${#OK_SCENES[@]}): ${OK_SCENES[*]:-none}"
echo "FAIL (${#FAIL_SCENES[@]}): ${FAIL_SCENES[*]:-none}"
echo "============================================================"

[ "${#FAIL_SCENES[@]}" -eq 0 ]
