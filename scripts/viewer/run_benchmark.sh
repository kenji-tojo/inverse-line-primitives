#!/usr/bin/env bash
#
# Benchmark every Shelly scene in every antialiasing mode.
#
# Requires the Vulkan SDK environment:
#   source ~/VulkanSDK/<version>/setup-env.sh
#
# Test cameras and ground-truth images come from
#   datasets/shelly_data_release/<scene>/transforms_test.json
#
# The released checkpoints (7.3 GB) are fetched from the Fuzzy dataset before
# the sweep starts.
#
# Usage:
#   bash scripts/viewer/run_benchmark.sh              # results/benchmark/<timestamp>/
#   bash scripts/viewer/run_benchmark.sh my_run_name  # results/benchmark/my_run_name/
#
# Output:
#   results/benchmark/<run>/
#     summary.json          per-mode averages and per-scene detail
#     <scene>/<aa>/
#       bench.json  bench.log
#       renders/    gt/
#
# Each (scene, mode) runs in its own process so a scene-specific driver state
# cannot carry into the rest of the sweep.
#
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# macOS clears DYLD_* when it executes /bin/bash, so the loader path set by
# setup-env.sh does not reach this script.  VULKAN_SDK is an ordinary variable
# and does survive, so re-derive the path from it.
if [ "$(uname)" = "Darwin" ] && [ -n "${VULKAN_SDK:-}" ]; then
    export DYLD_LIBRARY_PATH="${VULKAN_SDK}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi

SCENES=(fernvase horse khady kitten pug woolly)
AA_MODES=(none gaussian_msaa hw_msaa_2x hw_msaa_4x)
CKPT_DIR="datasets/fuzzy_dataset/checkpoints/shelly"

python scripts/fetch_data.py checkpoints || exit 1

RUN_NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="results/benchmark/${RUN_NAME}"
# Never overwrite an existing run: refuse if the target exists and is non-empty.
if [ -d "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${OUT_DIR} already exists and is non-empty - refusing to overwrite."
    echo "       Pass a different run name: bash scripts/viewer/run_benchmark.sh <run_name>"
    exit 1
fi
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Run: ${RUN_NAME}   (render speed)"
echo "Output: ${OUT_DIR}"
echo "Scenes: ${SCENES[*]}"
echo "AA modes: ${AA_MODES[*]}"
echo "============================================================"

t_total0=$(date +%s)

for aa in "${AA_MODES[@]}"; do
    for scene in "${SCENES[@]}"; do
        CKPT="${CKPT_DIR}/shelly_${scene}.npz"
        SCENE_DIR="${OUT_DIR}/${scene}/${aa}"

        if [ ! -f "${CKPT}" ]; then
            echo "WARNING: ${CKPT} not found, skipping ${scene}"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "  Bench: ${scene}  aa=${aa}"
        echo "------------------------------------------------------------"

        mkdir -p "${SCENE_DIR}"
        python scripts/viewer/benchmark.py "${CKPT}" \
            --aa "${aa}" \
            --out_dir "${SCENE_DIR}" \
            2>&1 | tee "${SCENE_DIR}/bench.log"
    done
done

# ---- Summary: per-mode averages across scenes into one JSON ----------------
echo ""
echo "Collecting summary ..."

if python3 -c "
import json, os

out_dir = '${OUT_DIR}'
scenes = '${SCENES[*]}'.split()
aa_modes = '${AA_MODES[*]}'.split()
keys = ['fps_mean', 'fps_median', 'frame_ms_mean']
summary = {}

for aa in aa_modes:
    per_scene = {}
    for scene in scenes:
        path = os.path.join(out_dir, scene, aa, 'bench.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rec = json.load(f)
        entry = {k: rec['summary_full'][k] for k in keys}
        entry['n_lines_kept'] = rec['n_lines_kept']
        entry['line_topology'] = rec['line_topology']
        entry['n_strips'] = rec['n_strips']
        if 'summary_graphics_only' in rec:
            entry['graphics_only_fps_mean'] = rec['summary_graphics_only']['fps_mean']
        per_scene[scene] = entry
    if per_scene:
        avg = {k: sum(v[k] for v in per_scene.values()) / len(per_scene) for k in keys}
        summary[aa] = {'per_scene': per_scene, 'average': avg}

label, mean, median, ms = 'aa mode', 'fps.mean', 'fps.median', 'ms.mean'
print(f'  {label:<16}{mean:>10}{median:>12}{ms:>10}   scenes')
for aa in aa_modes:
    if aa not in summary:
        continue
    a = summary[aa]['average']
    n = len(summary[aa]['per_scene'])
    fm, fmed, fms = a['fps_mean'], a['fps_median'], a['frame_ms_mean']
    print(f'  {aa:<16}{fm:>10.2f}{fmed:>12.2f}{fms:>10.2f}{n:>9}')

if not summary:
    raise SystemExit(1)

summary_path = os.path.join(out_dir, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved: {summary_path}')
"
then
    :
else
    echo ""
    echo "ERROR: no bench.json was produced - every run failed."
    echo "       See ${OUT_DIR}/<scene>/<aa>/bench.log"
    exit 1
fi

t_total1=$(date +%s)
elapsed=$(( t_total1 - t_total0 ))

echo ""
echo "============================================================"
echo "All done in $((elapsed / 60))m $((elapsed % 60))s.  Results: ${OUT_DIR}"
echo "============================================================"
