#!/usr/bin/env python3
"""Measure the viewer's Vulkan render path on a trained line checkpoint.

Usage
-----
    source ~/VulkanSDK/<version>/setup-env.sh
    python scripts/viewer/benchmark.py <path/to/ckpt.npz> --aa hw_msaa_4x

Renders the scene's test cameras offscreen and reports frame timings. One
antialiasing mode per invocation: each (scene, mode) is measured in its own
process so a scene-specific driver state cannot carry into the rest of a
sweep. A sweep is a shell loop:

    for aa in hw_msaa_4x gaussian_msaa; do
        python scripts/viewer/benchmark.py "$CKPT" --aa "$aa"
    done

Cameras and ground-truth images come from the dataset's
``transforms_test.json`` at native image resolution, read with the same
``utils.load_nerf_synthetic`` the training scripts use. ``--dataset`` points
at the directory holding ``<scene>/transforms_test.json``; the scene name is
read from the checkpoint.

Lines are pruned by opacity before rendering: a line is kept iff
``sigmoid(line_opacity_logit) >= --line_thresh`` (default 0.5). Low-opacity
primitives that survive into the draw call leak soft pixels. Faces are not
part of this path; the released checkpoints hold none.

The surviving segments are drawn as LINE_STRIP runs by default: the segment
list is rebaked into maximal walks joined by primitive-restart sentinels,
which roughly halves vertex-shader invocations on polyline data. Pass
``--line_topology list`` to draw the segments as stored.

The graphics-only pass excludes SH evaluation, so its difference against the
full pass is the shading cost. Pass ``--no_graphics_only`` to skip it.

Results land under ``results/benchmark/<scene>/<aa>/``:

    bench.json          per-view and aggregate frame timings
    renders/            one PNG per rendered view
    gt/                 the matching ground-truth view

``bench.json`` is the canonical output; the printed summary mirrors it.
``renders/`` and ``gt/`` follow the layout ``scripts/eval.py`` writes, so
image-quality metrics can be computed from them afterwards -- this script
measures speed only. Renders are captured on an untimed frame after each
view's measurement, so writing them does not move the timings. Pass
``--no_screenshots`` to skip both image sets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import fuzzydr_viewer
from fuzzydr_viewer import lines_to_strips

_REPO_ROOT = Path(__file__).resolve().parents[2]

# utils lives alongside the training scripts, one level up.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from utils import load_nerf_synthetic  # noqa: E402

# z-planes matching the Shelly training runs.
Z_NEAR = 0.01
Z_FAR = 100.0

# The published sweep used hw_msaa_4x and gaussian_msaa; the other two are
# the same modes the viewer panel offers.
AA_MODES = ["hw_msaa_4x", "gaussian_msaa", "hw_msaa_2x", "none"]


def recover_sh_coeffs(colors: np.ndarray) -> np.ndarray:
    """Convert checkpoint colors [N, 48] (SH stored as [N, 16, 3]) into
    the viewer's [N, 3, 16] layout."""
    if colors.ndim != 2 or colors.shape[1] != 48:
        raise ValueError(f"expected colors.shape == (N, 48); got {colors.shape}")
    N = colors.shape[0]
    return colors.reshape(N, 16, 3).transpose(0, 2, 1).copy().astype(np.float32)


def pack_vert_attrs(verts: np.ndarray) -> np.ndarray:
    """Build [N, 7] vert_attrs (pos + radius=1 placeholder + zero rgb).

    The radius slot is read only by the quad-line pipeline; Bresenham draws
    ignore it. RGB is overwritten by the SH compute pass.
    """
    N = verts.shape[0]
    out = np.zeros((N, 7), dtype=np.float32)
    out[:, :3] = verts.astype(np.float32)
    out[:, 3] = 1.0
    return out


def build_bench_json(times_measured: np.ndarray,
                     times_graphics_only: np.ndarray | None,
                     scene: str, aa: str, ckpt_path: Path,
                     W: int, H: int,
                     warmup: int, measure: int,
                     wall_sec: float,
                     line_thresh: float, n_lines_kept: int,
                     line_topology: str, n_strips: int) -> dict:
    """Machine-readable record of one (scene, aa) bench run."""
    record: dict = {
        "scene":        scene,
        "aa_mode":      aa,
        "ckpt":         str(ckpt_path),
        "width":        int(W),
        "height":       int(H),
        "warmup":       int(warmup),
        "measure":      int(measure),
        "wall_sec":     float(wall_sec),
        "line_thresh":  float(line_thresh),
        "n_lines_kept": int(n_lines_kept),
        "line_topology": line_topology,
        "n_strips":     int(n_strips),
        "summary_full": summarize(times_measured),
        "per_view_full": {
            "fps":     (1.0 / times_measured.mean(axis=1)).astype(float).tolist(),
            "mean_ms": (times_measured.mean(axis=1) * 1e3).astype(float).tolist(),
            "std_ms":  (times_measured.std(axis=1) * 1e3).astype(float).tolist(),
        },
    }
    if times_graphics_only is not None:
        record["summary_graphics_only"] = summarize(times_graphics_only)
        record["per_view_graphics_only"] = {
            "fps":     (1.0 / times_graphics_only.mean(axis=1)).astype(float).tolist(),
            "mean_ms": (times_graphics_only.mean(axis=1) * 1e3).astype(float).tolist(),
            "std_ms":  (times_graphics_only.std(axis=1) * 1e3).astype(float).tolist(),
        }
    return record


def save_images(images: np.ndarray, out_dir: Path) -> None:
    """Write one uint8 PNG per view as ``view_NNN.png``."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    V = images.shape[0]
    width = max(len(str(V - 1)), 3)
    mode = "RGBA" if images.shape[-1] == 4 else "RGB"
    for i in range(V):
        Image.fromarray(images[i], mode=mode).save(
            out_dir / f"view_{i:0{width}d}.png"
        )


def summarize(times_measured: np.ndarray) -> dict:
    """times_measured : float32 [V, measure] -> dict of summary stats.

    Caller is responsible for slicing off the warmup frames.
    """
    flat = times_measured.reshape(-1)
    per_view_mean = times_measured.mean(axis=1)
    fps_per_view = 1.0 / per_view_mean
    return {
        "num_views":       int(times_measured.shape[0]),
        "frames_per_view": int(times_measured.shape[1]),
        "fps_mean":        float(fps_per_view.mean()),
        "fps_median":      float(np.median(fps_per_view)),
        "fps_min":         float(fps_per_view.min()),
        "fps_max":         float(fps_per_view.max()),
        "fps_p5":          float(np.percentile(fps_per_view, 5)),
        "fps_p95":         float(np.percentile(fps_per_view, 95)),
        "frame_ms_mean":   float(flat.mean() * 1e3),
        "frame_ms_median": float(np.median(flat) * 1e3),
    }


def print_summary(label: str, times_measured: np.ndarray) -> None:
    s = summarize(times_measured)
    print(f"== {label} ==")
    print(f"  fps.mean        : {s['fps_mean']:10.2f}")
    print(f"  fps.median      : {s['fps_median']:10.2f}")
    print(f"  fps.min         : {s['fps_min']:10.2f}")
    print(f"  fps.max         : {s['fps_max']:10.2f}")
    print(f"  fps.p5          : {s['fps_p5']:10.2f}")
    print(f"  fps.p95         : {s['fps_p95']:10.2f}")
    print(f"  frame_ms.mean   : {s['frame_ms_mean']:8.3f}")
    print(f"  frame_ms.median : {s['frame_ms_median']:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="path to a checkpoint .npz")
    ap.add_argument("--dataset", type=Path,
                    default=_REPO_ROOT / "datasets" / "shelly_data_release",
                    help="directory holding <scene>/transforms_test.json "
                         "(default: datasets/shelly_data_release)")
    ap.add_argument("--scene", type=str, default=None,
                    help="dataset subdirectory; default follows the "
                         "checkpoint's scene field")
    ap.add_argument("--aa", type=str, default="hw_msaa_4x", choices=AA_MODES,
                    help="antialiasing mode (default: hw_msaa_4x)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=30)
    ap.add_argument("--no_graphics_only", action="store_true",
                    help="skip graphics-only timing (excludes SH evaluation)")
    ap.add_argument("--no_screenshots", action="store_true",
                    help="skip the per-view PNG capture")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="where bench.json, renders/ and gt/ are written "
                         "(default: results/benchmark/<scene>/<aa>)")
    ap.add_argument("--line_thresh", type=float, default=0.5,
                    help="keep lines with sigmoid(line_opacity_logit) >= this "
                         "(default: 0.5)")
    ap.add_argument("--line_topology", type=str, default="strip",
                    choices=["strip", "list"],
                    help="index buffer topology: 'strip' rebakes the segment "
                         "list into LINE_STRIP runs, roughly halving vertex-"
                         "shader invocations; 'list' draws the segments as "
                         "they are stored (default: strip)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    ck = np.load(args.ckpt, allow_pickle=False)
    meta = json.loads(ck["meta_json"].item()) if "meta_json" in ck.files else {}
    if meta.get("color_mode") != "sh" or meta.get("color_channels") != 48:
        raise SystemExit(f"expected a degree-3 SH checkpoint; got "
                         f"{meta.get('color_mode')!r} with "
                         f"{meta.get('color_channels')} colour channels")

    scene = args.scene or meta.get("scene")
    if not scene:
        raise SystemExit("checkpoint records no scene; pass --scene")
    scene_root = args.dataset / scene
    if not (scene_root / "transforms_test.json").exists():
        raise SystemExit(f"no transforms_test.json under {scene_root}")

    vert_attrs = pack_vert_attrs(ck["verts"].astype(np.float32))
    sh_coeffs = recover_sh_coeffs(ck["colors"].astype(np.float32))

    lines = np.ascontiguousarray(ck["lines"], dtype=np.uint32).reshape(-1, 2)
    lop = ck["line_opacity_logit"].astype(np.float32)
    if lop.size == lines.shape[0]:
        # sigmoid(x) >= t <=> x >= log(t / (1 - t)); avoids a per-element sigmoid.
        t = float(np.clip(args.line_thresh, 1e-6, 1.0 - 1e-6))
        lines = lines[lop >= float(np.log(t / (1.0 - t)))]
    if lines.shape[0] == 0:
        raise SystemExit("every line was pruned; lower --line_thresh")
    n_lines_kept = int(lines.shape[0])

    if args.line_topology == "strip":
        lines_payload, n_strips = lines_to_strips(lines)
    else:
        lines_payload, n_strips = lines, 0

    gt_images, mvps, eyes_t, W, H = load_nerf_synthetic(
        str(scene_root), "test", Z_NEAR, Z_FAR,
        dataset=meta.get("dataset", "shelly"))
    viewprojs = np.stack([m.numpy() for m in mvps], axis=0).astype(np.float32)
    eyes = np.stack([e.numpy() for e in eyes_t], axis=0).astype(np.float32)
    measure_graphics_only = not args.no_graphics_only
    capture_screenshots = not args.no_screenshots
    out_dir = args.out_dir or (
        _REPO_ROOT / "results" / "benchmark" / scene / args.aa)

    print(f"[bench] {scene}  aa={args.aa}  {W}x{H}  "
          f"views={viewprojs.shape[0]}  lines={n_lines_kept}  "
          f"topology={args.line_topology}"
          f"{f' strips={n_strips}' if n_strips else ''}  "
          f"warmup={args.warmup} measure={args.measure} "
          f"graphics_only={measure_graphics_only} "
          f"screenshots={capture_screenshots}",
          flush=True)

    t_start = time.time()
    result = fuzzydr_viewer.benchmark(
        vert_attrs, viewprojs, eyes,
        lines=lines_payload,
        sh_coeffs=sh_coeffs,
        width=W, height=H,
        warmup=args.warmup, measure=args.measure,
        measure_nosh=measure_graphics_only,   # backend name for the same pass
        capture_screenshots=capture_screenshots,
        aa_mode=args.aa,
        line_topology=args.line_topology,
    )
    wall = time.time() - t_start

    times_measured = result["times"][:, args.warmup:]
    times_graphics_only = (result["times_nosh"][:, args.warmup:]
                         if result["times_nosh"] is not None else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    record = build_bench_json(
        times_measured, times_graphics_only,
        scene, args.aa, args.ckpt, W, H,
        args.warmup, args.measure, wall,
        args.line_thresh, n_lines_kept,
        args.line_topology, n_strips,
    )
    bench_json = out_dir / "bench.json"
    bench_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print_summary(f"{scene} / {args.aa}", times_measured)
    if times_graphics_only is not None:
        print_summary(f"{scene} / {args.aa} / graphics-only", times_graphics_only)
    if result["images"] is not None:
        save_images(result["images"], out_dir / "renders")
        save_images((gt_images.numpy() * 255).astype(np.uint8), out_dir / "gt")
    print(f"  -> {bench_json}  ({wall:.1f}s)")


if __name__ == "__main__":
    main()
