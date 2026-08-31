#!/usr/bin/env python3
"""Render test views from a trained checkpoint and compute PSNR, SSIM and LPIPS.

Supports point, triangle and line primitives; the type is read from the
checkpoint.

Usage:
    python scripts/baselines/eval.py --ckpt results/shelly_points_<timestamp>/khady/shelly_khady_50000.npz \
                   --eval_dir results/shelly_points_<timestamp>/khady/eval_50000

Output:
    <eval_dir>/
        gt/             ground-truth test images (PNG)
        renders/        rendered test images (PNG)
        metrics.json    aggregated {PSNR, SSIM, LPIPS}
        per_view.json   per-image {PSNR, SSIM, LPIPS}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from math import exp
from PIL import Image

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuzzydr
from eval import (
    _box_downsample_rgba,
    ssim_metric,
    psnr_metric,
    lpips_vgg,
)
from utils import (
    load_nerf_synthetic,
    nerf_frame_filenames,
    load_colmap_dataset,
    load_checkpoint,
    SH_BAND_SIZES,
)


# ---------------------------------------------------------------------------
# Pixel reconstruction filter
# ---------------------------------------------------------------------------
# Default: fuzzydr.msaa_downsample_rgba (Gaussian, sigma=0.5).
# When --box_aa is set this is overridden to a 2x2 box average.


import math as _math_native

_NATIVE_AA_SIGMA = 0.5
_NATIVE_AA_KERNEL_CACHE: dict = {}


def _get_native_gaussian_kernel(sigma: float, device, dtype=torch.float32):
    key = (float(sigma), str(device), dtype)
    if key in _NATIVE_AA_KERNEL_CACHE:
        return _NATIVE_AA_KERNEL_CACHE[key]
    ks = max(3, 2 * int(_math_native.ceil(2.0 * float(sigma))) + 1)
    half = ks // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    g = torch.exp(-x * x / (2.0 * float(sigma) * float(sigma)))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d[None, None].expand(3, 1, ks, ks).contiguous()
    _NATIVE_AA_KERNEL_CACHE[key] = kernel
    return kernel


def _gaussian_native_rgba(img: torch.Tensor) -> torch.Tensor:
    """Isotropic Gaussian AA on a natively-rendered RGBA image (no downsample).

    Sigma is in output pixels and is read from ``_NATIVE_AA_SIGMA``.
    Returns [H, W, 3]; alpha is dropped.
    """
    kernel = _get_native_gaussian_kernel(_NATIVE_AA_SIGMA, img.device, img.dtype)
    ks = kernel.shape[-1]
    rgb = img[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
    rgb = F.conv2d(rgb, kernel, padding=ks // 2, groups=3)
    return rgb.squeeze(0).permute(1, 2, 0).contiguous()


# Module-level dispatchers.  Reassigned in main() based on --box_aa /
# --native_render so the three render_test_view_* helpers below pick up
# the variant without threading flags through their signatures.
_downsample_rgba = lambda img: fuzzydr.msaa_downsample_rgba(img)
_render_scale    = 2      # rasterizer renders at (W x scale, H x scale)


# ===========================================================================
# Rendering
# ===========================================================================

def render_test_view_lines(
    verts, sh_coeffs, faces, face_opacity, lines, line_opacity,
    mvp, eye, width, height, radius=None, bresen=True,
):
    """Render a single test view of line primitives, returning [H, W, 3] float32 RGB.

    With ``bresen`` the lines are 1-px Bresenham segments and the radius slot
    is ignored, so a dummy value is packed.  Otherwise they are camera-facing
    quads and ``radius`` gives their world-space width.
    """
    if bresen or radius is None:
        rad_slot = torch.ones(int(verts.shape[0]), dtype=torch.float32, device=verts.device)
    else:
        rad_slot = radius
    va = fuzzydr.eval_sh_attrs(verts, sh_coeffs, rad_slot, campos=eye)

    return _downsample_rgba(fuzzydr.rasterize(
        va, viewproj=mvp, campos=eye,
        faces=faces, face_opacity=face_opacity,
        lines=lines, line_opacity=line_opacity,
        width=width * _render_scale, height=height * _render_scale,
        tau=0.5, seed=0, white_bg=True,
        bresen_lines=bresen,
    )).clamp(0, 1).contiguous()


def render_test_view_triangles(
    verts, sh_coeffs, faces, face_opacity,
    mvp, eye, width, height,
):
    """Render a single test view (triangles only, no lines), returning [H, W, 3] float32 RGB."""
    radius = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    va = fuzzydr.eval_sh_attrs(verts, sh_coeffs, radius, campos=eye)

    return _downsample_rgba(fuzzydr.rasterize(
        va, viewproj=mvp, campos=eye,
        faces=faces, face_opacity=face_opacity,
        width=width * _render_scale, height=height * _render_scale,
        tau=0.5, seed=0, white_bg=True,
    )).clamp(0, 1).contiguous()


def render_test_view_points(
    verts, sh_coeffs, points, point_opacity,
    mvp, eye, width, height,
):
    """Render a single test view (points), returning [H, W, 3] float32 RGB."""
    radius = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    va = fuzzydr.eval_sh_attrs(verts, sh_coeffs, radius, campos=eye)

    return _downsample_rgba(fuzzydr.rasterize_points(
        va, viewproj=mvp, campos=eye,
        points=points, point_opacity=point_opacity,
        width=width * _render_scale, height=height * _render_scale,
        tau=0.5, seed=0, white_bg=True,
    )).clamp(0, 1).contiguous()


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Render and evaluate a trained checkpoint.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .npz checkpoint")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--eval_dir", type=str, default=None,
                    help="Override output directory for eval results "
                         "(default: <ckpt_dir>/eval/)")
    ap.add_argument("--box_aa", action="store_true",
                    help="Use a 2x2 box average for pixel reconstruction "
                         "instead of the default Gaussian filter.")
    ap.add_argument("--native_render", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Render at native (W, H) with a 3x3 box AA filter "
                         "instead of 2x supersampling.  Default: read from the "
                         "checkpoint meta.")
    args = ap.parse_args()

    if args.box_aa and args.native_render:
        raise SystemExit("--box_aa and --native_render are mutually exclusive.")
    if args.box_aa:
        global _downsample_rgba
        _downsample_rgba = _box_downsample_rgba
        print("[eval] AA filter = 2x2 box average on 2x supersample (--box_aa).")
    else:
        print("[eval] AA filter = Gaussian (msaa_downsample_rgba, sigma=0.5).")

    device = torch.device(args.device, args.gpu_id) if args.device == "cuda" else torch.device("cpu")
    torch.set_default_dtype(torch.float32)

    # ---- Load checkpoint --------------------------------------------------
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt)
    meta = ckpt["meta"]

    scene = meta["scene"]
    dataset = meta["dataset"]
    sh_degree = meta.get("sh_degree", 3)
    # Auto-detect --native_render from checkpoint meta unless explicitly set.
    meta_native = bool(meta.get("native_render", False))
    if args.native_render is None:
        args.native_render = meta_native
    elif args.native_render != meta_native:
        print(f"  WARNING: --native_render={args.native_render} differs from "
              f"checkpoint meta ({meta_native}); using CLI value.")
    if args.native_render:
        global _render_scale, _NATIVE_AA_SIGMA
        # Take sigma from the checkpoint meta, defaulting to 0.5.
        _NATIVE_AA_SIGMA = float(meta.get("native_aa_sigma", 0.5))
        if not args.box_aa:
            _downsample_rgba = _gaussian_native_rgba
        _render_scale = 1
        print(f"[eval] Native render: rasterize at (W, H); "
              f"Gaussian AA sigma={_NATIVE_AA_SIGMA:.3f} output px "
              f"(--native_render).")
    num_sh_coeffs = sum(SH_BAND_SIZES[:sh_degree + 1])

    verts = torch.from_numpy(ckpt["verts"]).to(device)

    # SH coefficients: stored interleaved [N, num_sh*3] in checkpoint
    colors_raw = torch.from_numpy(ckpt["colors"]).to(device)  # [N, 48]
    sh_coeffs = colors_raw.view(-1, 16, 3).permute(0, 2, 1)[..., :num_sh_coeffs].contiguous()

    # Detect primitive type: points, triangles, or lines
    n_lines = int(ckpt["lines"].shape[0]) if "lines" in ckpt else 0
    n_faces = int(ckpt["faces"].shape[0]) if "faces" in ckpt else 0
    is_points = (
        meta.get("primitive") == "points"
        or "n_points" in meta
        or (n_lines == 0 and n_faces == 0)
    )
    is_triangles = (n_faces > 0 and n_lines == 0 and not is_points)

    if is_points:
        N = int(verts.shape[0])
        points = torch.arange(N, dtype=torch.int32, device=device).view(torch.uint32)
        # Prefer point_opacity_logit; fall back to face_opacity_logit.
        if "point_opacity_logit" in ckpt and ckpt["point_opacity_logit"].shape[0] == N:
            point_opacity = torch.sigmoid(torch.from_numpy(ckpt["point_opacity_logit"]).to(device))
        elif "face_opacity_logit" in ckpt and ckpt["face_opacity_logit"].shape[0] == N:
            point_opacity = torch.sigmoid(torch.from_numpy(ckpt["face_opacity_logit"]).to(device))
        else:
            point_opacity = torch.ones(N, dtype=torch.float32, device=device)
        print(f"  [points] scene={scene}  dataset={dataset}  points={N}  sh_degree={sh_degree}")
    elif is_triangles:
        faces = torch.from_numpy(ckpt["faces"]).to(device)
        face_opacity = torch.sigmoid(torch.from_numpy(ckpt["face_opacity_logit"]).to(device))
        print(f"  [triangles] scene={scene}  dataset={dataset}  verts={verts.shape[0]}  "
              f"faces={n_faces}  sh_degree={sh_degree}")
    else:
        faces = torch.from_numpy(ckpt["faces"]).to(device)
        lines = torch.from_numpy(ckpt["lines"]).to(device)
        face_opacity = torch.sigmoid(torch.from_numpy(ckpt["face_opacity_logit"]).to(device))
        line_opacity = torch.sigmoid(torch.from_numpy(ckpt["line_opacity_logit"]).to(device))
        # Checkpoints without the flag predate it and are Bresenham.
        bresen = bool(meta.get("bresen_lines", True))
        radius = (torch.from_numpy(ckpt["radius"]).to(device)
                  if not bresen and "radius" in ckpt and ckpt["radius"].shape[0] else None)
        print(f"  [lines] scene={scene}  dataset={dataset}  verts={verts.shape[0]}  "
              f"faces={n_faces}  lines={n_lines}  sh_degree={sh_degree}  "
              f"bresen={bresen}")

    # ---- Load test data ---------------------------------------------------
    if dataset == "360_v2":
        scene_root = os.path.join("datasets", "360_v2", scene)
        resolution = meta.get("resolution", 4)
        print(f"Loading {args.split} split from {scene_root} (resolution {resolution}) ...")
        images_cpu, mvps_cpu, eyes_cpu, W, H, _ = load_colmap_dataset(
            scene_root, args.split, z_near=0.01, z_far=100.0,
            resolution=resolution,
        )
        fnames = None
    else:
        scene_root_dict = {
            "nerf_synthetic": os.path.join("datasets", "nerf_synthetic"),
            "shelly": os.path.join("datasets", "shelly_data_release"),
        }
        scene_root = os.path.join(scene_root_dict[dataset], scene)
        print(f"Loading {args.split} split from {scene_root} ...")
        images_cpu, mvps_cpu, eyes_cpu, W, H = load_nerf_synthetic(
            scene_root, args.split, z_near=0.01, z_far=100.0, dataset=dataset,
        )
        fnames = nerf_frame_filenames(scene_root, args.split, dataset=dataset)
    N_views = int(images_cpu.shape[0])
    print(f"  {N_views} views, resolution {W}x{H}")

    # Saved images keep their dataset filenames; fall back to the view index
    # when the split does not carry per-frame names.
    if fnames is None or len(fnames) != N_views:
        fnames = [f"{vi:05d}.png" for vi in range(N_views)]

    mvps = [m.to(device).contiguous() for m in mvps_cpu]
    eyes = [e.to(device).contiguous() for e in eyes_cpu]

    # ---- Output dirs ------------------------------------------------------
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    eval_dir = args.eval_dir if args.eval_dir is not None else os.path.join(ckpt_dir, "eval")
    gt_dir = os.path.join(eval_dir, "gt")
    render_dir = os.path.join(eval_dir, "renders")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    # ---- Render and save images -------------------------------------------
    print("Rendering and saving images ...")
    fuzzydr.init()

    psnrs, ssims, lpipss = [], [], []

    t0 = time.time()
    try:
        with torch.no_grad():
            for vi in range(N_views):
                if is_points:
                    pred = render_test_view_points(
                        verts, sh_coeffs, points, point_opacity,
                        mvps[vi], eyes[vi], W, H,
                    )
                elif is_triangles:
                    pred = render_test_view_triangles(
                        verts, sh_coeffs, faces, face_opacity,
                        mvps[vi], eyes[vi], W, H,
                    )
                else:
                    pred = render_test_view_lines(
                        verts, sh_coeffs,
                        faces, face_opacity, lines, line_opacity,
                        mvps[vi], eyes[vi], W, H,
                        radius=radius, bresen=bresen,
                    )
                gt = images_cpu[vi].to(device)

                # Save as PNG
                fname = fnames[vi]
                pred_u8 = (pred.cpu().numpy() * 255).astype(np.uint8)
                gt_u8 = (gt.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(pred_u8).save(os.path.join(render_dir, fname))
                Image.fromarray(gt_u8).save(os.path.join(gt_dir, fname))

                # Compute metrics in [1, 3, H, W] format
                pred_4d = pred.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, H, W]
                gt_4d = gt.permute(2, 0, 1).unsqueeze(0).contiguous()      # [1, 3, H, W]

                psnrs.append(psnr_metric(pred_4d, gt_4d).item())
                ssims.append(ssim_metric(pred_4d, gt_4d).item())
                lpipss.append(lpips_vgg(pred_4d, gt_4d).item())

                if (vi + 1) % 10 == 0 or vi == N_views - 1:
                    print(f"  [{vi+1}/{N_views}]  PSNR={psnrs[-1]:.2f}  "
                          f"SSIM={ssims[-1]:.4f}  LPIPS={lpipss[-1]:.4f}")

    finally:
        fuzzydr.shutdown()

    elapsed = time.time() - t0

    # ---- Aggregate and save metrics ---------------------------------------
    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_lpips = float(np.mean(lpipss))

    print(f"\n{'='*60}")
    print(f"  PSNR : {mean_psnr:.6f}")
    print(f"  SSIM : {mean_ssim:.6f}")
    print(f"  LPIPS: {mean_lpips:.6f}")
    print(f"{'='*60}")
    print(f"  ({N_views} views, {elapsed:.1f}s)")

    results = {
        "PSNR": mean_psnr,
        "SSIM": mean_ssim,
        "LPIPS": mean_lpips,
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    per_view = {
        "PSNR": {fn: p for fn, p in zip(fnames, psnrs)},
        "SSIM": {fn: s for fn, s in zip(fnames, ssims)},
        "LPIPS": {fn: l for fn, l in zip(fnames, lpipss)},
    }
    with open(os.path.join(eval_dir, "per_view.json"), "w") as f:
        json.dump(per_view, f, indent=2)

    print(f"\nSaved to {eval_dir}/")
    print(f"  renders/  ({N_views} PNGs)")
    print(f"  gt/       ({N_views} PNGs)")
    print(f"  metrics.json")
    print(f"  per_view.json")


if __name__ == "__main__":
    main()
