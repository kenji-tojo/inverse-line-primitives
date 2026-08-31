#!/usr/bin/env python3
"""Render test views from a trained checkpoint and compute PSNR, SSIM and LPIPS.

Line primitives are rendered as 1-px Bresenham segments.

Usage:
    python scripts/eval.py --ckpt results/shelly_lines_<timestamp>/khady/shelly_khady_50000.npz \
                   --eval_dir results/shelly_lines_<timestamp>/khady/eval_50000

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

import fuzzydr
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

def _box_downsample_rgba(img: torch.Tensor) -> torch.Tensor:
    """2x2 box-average downsample of a 2x supersampled RGBA image.

    Drop-in replacement for ``fuzzydr.msaa_downsample_rgba``: takes
    float32 ``[H, W, 4]``, returns float32 ``[H/2, W/2, 3]``.  Alpha is
    ignored (rasterizer pads it to 1).
    """
    rgb = img[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
    rgb = F.avg_pool2d(rgb, kernel_size=2, stride=2)
    return rgb.squeeze(0).permute(1, 2, 0).contiguous()


# Module-level dispatcher.  Reassigned in main() when --box_aa is set, so the
# render helper below picks up the variant without threading a flag through
# its signature.
_downsample_rgba = lambda img: fuzzydr.msaa_downsample_rgba(img)
_render_scale    = 2      # rasterizer renders at (W x scale, H x scale)


# ===========================================================================
# Metrics
# ===========================================================================

# ---- SSIM ----------------------------------------------------------------

def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    w1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    w2d = w1d.mm(w1d.t()).float().unsqueeze(0).unsqueeze(0)
    return w2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_metric(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM over an 11x11 Gaussian window (sigma 1.5).  Input: [1, C, H, W]."""
    channel = img1.size(-3)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ---- PSNR ----------------------------------------------------------------

def psnr_metric(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """PSNR in dB, averaged per image.  Input: [1, C, H, W] tensors."""
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ---- LPIPS ---------------------------------------------------------------

_lpips_fn = None


def lpips_vgg(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """LPIPS with a VGG backbone.  Input: [1, C, H, W]."""
    global _lpips_fn
    if _lpips_fn is None:
        import lpips as lpips_lib
        _lpips_fn = lpips_lib.LPIPS(net="vgg").to(img1.device)
    with torch.no_grad():
        return _lpips_fn(img1, img2).squeeze()


# ===========================================================================
# Rendering
# ===========================================================================

def render_test_view_lines(
    verts, sh_coeffs, faces, face_opacity, lines, line_opacity,
    mvp, eye, width, height,
):
    """Render a single test view of line primitives, returning [H, W, 3] float32 RGB.

    Lines render as 1-px Bresenham segments, which ignore the per-vertex
    radius slot entirely.  The value packed below is a dummy placeholder
    only, NOT a trained or meaningful width.
    """
    rad_slot = torch.ones(int(verts.shape[0]), dtype=torch.float32, device=verts.device)
    va = fuzzydr.eval_sh_attrs(verts, sh_coeffs, rad_slot, campos=eye)

    return _downsample_rgba(fuzzydr.rasterize(
        va, viewproj=mvp, campos=eye,
        faces=faces, face_opacity=face_opacity,
        lines=lines, line_opacity=line_opacity,
        width=width * _render_scale, height=height * _render_scale,
        tau=0.5, seed=0, white_bg=True,
        bresen_lines=True,
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
    args = ap.parse_args()

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
    num_sh_coeffs = sum(SH_BAND_SIZES[:sh_degree + 1])

    verts = torch.from_numpy(ckpt["verts"]).to(device)

    # SH coefficients: stored interleaved [N, num_sh*3] in checkpoint
    colors_raw = torch.from_numpy(ckpt["colors"]).to(device)  # [N, 48]
    sh_coeffs = colors_raw.view(-1, 16, 3).permute(0, 2, 1)[..., :num_sh_coeffs].contiguous()

    n_lines = int(ckpt["lines"].shape[0]) if "lines" in ckpt else 0
    n_faces = int(ckpt["faces"].shape[0]) if "faces" in ckpt else 0

    faces = torch.from_numpy(ckpt["faces"]).to(device)
    lines = torch.from_numpy(ckpt["lines"]).to(device)
    face_opacity = torch.sigmoid(torch.from_numpy(ckpt["face_opacity_logit"]).to(device))
    line_opacity = torch.sigmoid(torch.from_numpy(ckpt["line_opacity_logit"]).to(device))
    print(f"  [lines] scene={scene}  dataset={dataset}  verts={verts.shape[0]}  "
          f"faces={n_faces}  lines={n_lines}  sh_degree={sh_degree}")

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
                pred = render_test_view_lines(
                    verts, sh_coeffs,
                    faces, face_opacity, lines, line_opacity,
                    mvps[vi], eyes[vi], W, H,
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
