#!/usr/bin/env python3
"""Point-primitive training on the Shelly scenes.

Point baseline for comparison against the line primitives: same spherical
harmonics, learning-rate schedule, loss, and checkpoint schedule.  The point
budget is fixed throughout training - instead of splitting, each adaptive
step prunes transparent points and duplicates surviving ones into the freed
slots with a small position jitter (0.1 x mean KNN-1 distance by default).

Usage:
    python scripts/baselines/train_shelly_points.py --scene khady
    python scripts/baselines/train_shelly_points.py --scene khady --iters 50000 --num_points 2000000

Run every scene:
    bash scripts/baselines/run_shelly_points.sh

Output:
    <out_dir>/<scene>/
        options.json  timing.json  loss.txt  test_mae.txt
        shelly_<scene>_07000.npz  shelly_<scene>_50000.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

try:
    from pytorch_msssim import ssim
    use_ssim = True
except ImportError:
    print("[torch] WARNING: pytorch_msssim not found, SSIM loss disabled.")
    use_ssim = False


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuzzydr
import fetch_data
from utils import (
    spatial_lr_scale,
    select_device,
    load_ply_positions,
    load_nerf_synthetic,
    load_colmap_dataset,
    SH_BAND_SIZES,
    SH_NUM_COEFFS,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Point primitive training.")

    # Data
    ap.add_argument("--scene", type=str, default="khady")
    # Only "shelly" is used in the paper.  The nerf_synthetic and 360_v2
    # loaders are kept for future extensions, but their data is not shipped, so
    # they need an explicit --scene_root.
    ap.add_argument("--dataset", type=str, default="shelly",
                    choices=["nerf_synthetic", "shelly", "360_v2"])
    ap.add_argument("--scene_root", type=str, default=None,
                    help="Directory holding the scene. Defaults to "
                         "datasets/shelly_data_release/<scene>; required for "
                         "the nerf_synthetic and 360_v2 datasets.")
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--resolution", type=int, default=1,
                    help="Image downsample factor for COLMAP datasets (1, 2, 4, 8)")

    # Render
    ap.add_argument("--sh_degree", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--sh_upgrade_every", type=int, default=1000)

    # Device
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)

    # Training
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)

    # Point-cloud init
    ap.add_argument("--num_points", type=int, default=2_000_000)
    ap.add_argument("--init_samples", type=str, default=None,
                    help="Path to pre-sampled PLY (auto-resolved for shelly)")
    ap.add_argument("--init_mesh_band", type=float, default=0.1)
    ap.add_argument("--no_fetch", dest="fetch", action="store_false", default=True,
                    help="Do not download missing seed points from the "
                         "Fuzzy dataset; fail instead.")

    # Learning rates
    ap.add_argument("--lr_verts", type=float, default=2.5e-5,
                    help="Initial position LR (multiplied by spatial_lr_scale if --spatial_lr_scale)")
    ap.add_argument("--lr_verts_final", type=float, default=5.0e-6,
                    help="Final position LR")
    ap.add_argument("--spatial_lr_scale", action=argparse.BooleanOptionalAction, default=True,
                    help="Scale the position learning rate by camera extent")
    ap.add_argument("--lr_sh_dc", type=float, default=2.5e-3)
    ap.add_argument("--lr_sh_rest", type=float, default=1.25e-4)
    ap.add_argument("--lr_pop", type=float, default=0.05)

    # Adaptive (prune + duplicate)
    ap.add_argument("--remesh_start", type=int, default=500)
    ap.add_argument("--remesh_every", type=int, default=100)
    ap.add_argument("--remesh_end", type=int, default=35_000)
    ap.add_argument("--prune_thresh", type=float, default=0.05)
    ap.add_argument("--prune_thresh_final", type=float, default=0.5,
                    help="Target prune threshold at end of ramp window.")
    ap.add_argument("--prune_ramp_start", type=int, default=25_000,
                    help="Iter at which prune_thresh begins ramping linearly "
                         "from prune_thresh to prune_thresh_final, finishing "
                         "at remesh_end.")
    ap.add_argument("--dup_jitter_lr_frac", type=float, default=0.1,
                    help="Duplicate position jitter std = frac * current verts LR "
                         "(scales with spatial_lr_scale and the LR schedule)")

    # Camera
    ap.add_argument("--z_near", type=float, default=0.01)
    ap.add_argument("--z_far", type=float, default=100.0)

    # Checkpoints & output
    ap.add_argument("--ckpt_iters", nargs="+", type=int, default=[7000, 50000])
    ap.add_argument("--time_iters", nargs="+", type=int, default=[50000])
    ap.add_argument("--out_dir", type=str, default="results/shelly_points")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--no_test_mae", action="store_true",
                    help="Disable periodic test-set MAE evaluation")

    args = ap.parse_args()

    # Auto-resolve pre-sampled PLY for shelly.
    if args.dataset == "shelly" and args.init_samples is None:
        args.init_samples = os.path.join(
            "datasets", "fuzzy_dataset", "coarse", "neus2_shelly_dtu_15000steps",
            f"shelly_{args.scene}", "band_samples",
            f"band{float(args.init_mesh_band):.3f}_n{int(args.num_points)}.ply",
        )

    if args.init_samples is not None:
        fetch_data.ensure_seed_ply(args.init_samples, enabled=args.fetch)

    # SH setup.
    num_sh_coeffs = sum(SH_BAND_SIZES[:args.sh_degree + 1])
    num_sh_dc = 1
    num_sh_rest = num_sh_coeffs - num_sh_dc
    sh_upgrade_every = int(args.sh_upgrade_every)
    active_sh_degree = 0 if sh_upgrade_every > 0 else args.sh_degree

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device, args.gpu_id)
    print(f"[torch] device={device}  cuda_available={torch.cuda.is_available()}")

    # Output directory.
    out_dir = os.path.join(args.out_dir, args.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Save options.
    opts_path = os.path.join(out_dir, "options.json")
    with open(opts_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Options saved: {opts_path}")

    iters = int(args.iters)
    remesh_start = int(args.remesh_start)
    remesh_every = int(args.remesh_every)
    remesh_end = int(args.remesh_end)
    prune_thresh = float(args.prune_thresh)
    eps = float(args.eps)

    ckpt_iters = set(args.ckpt_iters)
    time_iters = sorted(args.time_iters)
    timing: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    sfm_pts = None

    if args.scene_root is not None:
        scene_root = args.scene_root
    elif args.dataset == "shelly":
        scene_root = os.path.join("datasets", "shelly_data_release", args.scene)
    else:
        raise SystemExit(
            f"--scene_root is required for --dataset {args.dataset} "
            f"(only shelly has a default location).")

    if args.dataset == "360_v2":
        images_cpu, mvps_cpu, eyes_cpu, width, height, sfm_pts = load_colmap_dataset(
            scene_root, args.split,
            float(args.z_near), float(args.z_far),
            resolution=int(args.resolution),
        )
    else:
        images_cpu, mvps_cpu, eyes_cpu, width, height = load_nerf_synthetic(
            scene_root, args.split,
            float(args.z_near), float(args.z_far),
            dataset=args.dataset,
        )
    num_views = int(images_cpu.shape[0])

    # Test split for periodic MAE.
    do_test_mae = not args.no_test_mae
    if do_test_mae:
        if args.dataset == "360_v2":
            test_images_cpu, test_mvps_cpu, test_eyes_cpu, test_W, test_H, _ = load_colmap_dataset(
                scene_root, "test",
                float(args.z_near), float(args.z_far),
                resolution=int(args.resolution),
            )
        else:
            test_images_cpu, test_mvps_cpu, test_eyes_cpu, test_W, test_H = load_nerf_synthetic(
                scene_root, "test",
                float(args.z_near), float(args.z_far),
                dataset=args.dataset,
            )
        num_test_views = int(test_images_cpu.shape[0])
        print(f"[test] {num_test_views} test views for periodic MAE eval")
        _log_pts = np.logspace(np.log10(100), np.log10(iters), num=20).astype(int)
        mae_eval_iters = sorted(set(np.clip(_log_pts, 1, iters)))
        mae_log: list[tuple[int, float]] = []

    # Position LR (optionally scaled by camera spatial extent).
    if args.spatial_lr_scale:
        slr_scale = spatial_lr_scale(eyes_cpu)
        lr_verts_init = float(args.lr_verts) * slr_scale
        lr_verts_final = float(args.lr_verts_final) * slr_scale
        print(f"[camera] spatial_lr_scale={slr_scale:.4f}  "
              f"lr_verts={lr_verts_init:.6f} (base {args.lr_verts} x {slr_scale:.2f})")
    else:
        lr_verts_init = float(args.lr_verts)
        lr_verts_final = float(args.lr_verts_final)

    # ------------------------------------------------------------------
    # Build point cloud
    # ------------------------------------------------------------------
    if sfm_pts is not None:
        # COLMAP: use SfM points directly (may be fewer than --num_points).
        pts_cpu = torch.as_tensor(sfm_pts, dtype=torch.float32).contiguous()
        N = int(pts_cpu.shape[0])
        print(f"[init] Points from SfM: {N} points")
    elif args.init_samples is not None:
        pts_np = load_ply_positions(args.init_samples)
        M = int(pts_np.shape[0])
        if M != int(args.num_points):
            raise ValueError(
                f"PLY point count ({M}) != --num_points ({args.num_points}). "
                f"Pre-sample the exact count with "
                f"`python datasets/fuzzy_dataset/coarse/band_sampling.py --band {args.init_mesh_band} "
                f"--n_points {args.num_points} --mesh <coarse.obj> --out <out.ply>`."
            )
        pts_cpu = torch.as_tensor(pts_np, dtype=torch.float32).contiguous()
        N = M
        print(f"[init] Points from PLY: {args.init_samples}  ({N})")
    else:
        raise ValueError("Must provide --init_samples (or use --dataset shelly for auto-resolve).")

    # ------------------------------------------------------------------
    # Move to device
    # ------------------------------------------------------------------
    images = images_cpu.to(device=device).contiguous()
    mvps = [m.to(device=device).contiguous() for m in mvps_cpu]
    eyes = [e.to(device=device).contiguous() for e in eyes_cpu]
    if do_test_mae:
        test_images = test_images_cpu.to(device=device).contiguous()
        test_mvps = [m.to(device=device).contiguous() for m in test_mvps_cpu]
        test_eyes = [e.to(device=device).contiguous() for e in test_eyes_cpu]
    points_cur = torch.arange(N, dtype=torch.int32, device=device).view(torch.uint32).contiguous()
    radius_zero = torch.zeros(N, dtype=torch.float32, device=device)

    # Trainables.
    verts = torch.nn.Parameter(
        pts_cpu.to(device).clone().detach().contiguous().requires_grad_(True))
    sh_dc = torch.nn.Parameter(
        torch.zeros((N, 3, num_sh_dc), dtype=torch.float32, device=device).contiguous().requires_grad_(True))
    sh_rest = (torch.nn.Parameter(
        torch.zeros((N, 3, num_sh_rest), dtype=torch.float32, device=device).contiguous().requires_grad_(True))
        if num_sh_rest > 0 else None)

    def get_sh():
        if sh_rest is not None:
            sh_all = torch.cat([sh_dc, sh_rest], dim=-1)
        else:
            sh_all = sh_dc
        num_active = sum(SH_BAND_SIZES[:active_sh_degree + 1])
        return sh_all[..., :num_active]

    _init_opacity_logit = float(torch.tensor(0.1).logit())
    point_opacity_logit = torch.nn.Parameter(
        torch.full((N,), _init_opacity_logit, dtype=torch.float32, device=device).contiguous().requires_grad_(True))

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    opt_verts = fuzzydr.optimize.VectorAdam([verts], lr=lr_verts_init)
    sh_groups = [{"params": [sh_dc], "lr": float(args.lr_sh_dc)}]
    if sh_rest is not None:
        sh_groups.append({"params": [sh_rest], "lr": float(args.lr_sh_rest)})
    opt_other = torch.optim.Adam(sh_groups + [
        {"params": [point_opacity_logit], "lr": float(args.lr_pop)},
    ], foreach=False)

    def _zero_adam_state(param, idx):
        """Zero Adam/VectorAdam state at the given slots for *param*."""
        for opt in (opt_verts, opt_other):
            st = opt.state.get(param)
            if st is None:
                continue
            for key in ("exp_avg", "exp_avg_sq", "g1", "g2"):
                if key in st:
                    st[key][idx] = 0
            return

    needs_topology_update = True

    # ------------------------------------------------------------------
    # Checkpoint helper (dedicated point-baseline format).
    # Fields: verts [N,3], colors [N,48] (SH padded to degree-3),
    #         point_opacity_logit [N], meta_json (primitive="points").
    # ------------------------------------------------------------------
    def save_ckpt(it: int):
        with torch.no_grad():
            _N = int(verts.shape[0])
            _sh = (torch.cat([sh_dc, sh_rest], dim=-1) if sh_rest is not None else sh_dc).detach().cpu()
            if num_sh_coeffs < SH_NUM_COEFFS:
                _sh = torch.cat([_sh, torch.zeros(_N, 3, SH_NUM_COEFFS - num_sh_coeffs)], dim=-1)
            colors = _sh.permute(0, 2, 1).reshape(_N, SH_NUM_COEFFS * 3).numpy().astype(np.float32)
            meta = {
                "format_version": 1,
                "primitive": "points",
                "color_mode": "sh",
                "sh_degree": 3,
                "scene": args.scene, "split": args.split,
                "dataset": args.dataset, "iters": it,
                "prune_thresh_train": prune_thresh,
                "resolution": int(args.resolution),
                "n_points": _N,
            }
            ckpt_path = os.path.join(out_dir, f"{args.dataset}_{args.scene}_{it:05d}.npz")
            np.savez(
                ckpt_path,
                verts=verts.detach().cpu().numpy().astype(np.float32),
                colors=colors,
                point_opacity_logit=point_opacity_logit.detach().cpu().numpy().astype(np.float32),
                meta_json=np.array(json.dumps(meta)),
            )
            print(f"[ckpt] Saved -> {ckpt_path}  (points={_N})")

    print(f"[remesh] prune_thresh ramp: {prune_thresh:.3f} -> {float(args.prune_thresh_final):.3f} "
          f"over iters {int(args.prune_ramp_start)}-{remesh_end} (linear)")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    fuzzydr.init()
    try:
        decay_gamma = (lr_verts_final / lr_verts_init) ** (1.0 / max(iters, 1))
        sched_verts = torch.optim.lr_scheduler.ExponentialLR(opt_verts, gamma=decay_gamma)
        print(f"[lr] exp ramp: {lr_verts_init:.6g} -> {lr_verts_final:.6g} over {iters} iters  (gamma={decay_gamma:.8f})")

        losses: list[float] = []
        t_eval_total = 0.0
        t_train_start = time.time()

        pbar = tqdm(range(1, iters + 1), desc=f"Training [{args.scene}]", ncols=120)
        view_order = torch.randperm(num_views).tolist()

        for it in pbar:
            # Progressive SH.
            if sh_upgrade_every > 0 and it % sh_upgrade_every == 0:
                if active_sh_degree < args.sh_degree:
                    active_sh_degree += 1
                    pbar.write(f"[SH] degree -> {active_sh_degree} at iter {it}")

            if (it - 1) % num_views == 0 and it != 1:
                view_order = torch.randperm(num_views).tolist()
            view_idx = view_order[(it - 1) % num_views]

            opt_verts.zero_grad(set_to_none=True)
            opt_other.zero_grad(set_to_none=True)

            # Render.
            mvp = mvps[view_idx]
            eye = eyes[view_idx]
            va = fuzzydr.eval_sh_attrs(verts, get_sh(), radius_zero, campos=eye)
            point_opacity = torch.sigmoid(point_opacity_logit)

            rgba = fuzzydr.rasterize_points(
                va,
                viewproj=mvp, campos=eye,
                points=points_cur if needs_topology_update else points_cur.shape[0],
                point_opacity=point_opacity,
                width=width * 2, height=height * 2,
                tau=-1.0, seed=int(it),
                white_bg=True,
            )
            rgba = fuzzydr.msaa_downsample_rgba(rgba)
            needs_topology_update = False
            img = rgba[..., :3].contiguous()

            gt = images[view_idx]
            diff = torch.abs(img - gt)

            # Loss.
            l1_loss = diff.mean()
            if use_ssim:
                ssim_loss = 0.5 * (1.0 - ssim(
                    img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0))
                loss = 0.8 * l1_loss + 0.2 * ssim_loss
            else:
                loss = l1_loss

            error = diff.mean(dim=-1).detach().contiguous()
            aux = fuzzydr.opacity_mask_aux_loss(
                point_opacity=point_opacity,
                error=fuzzydr.upsample2x2_scalar(error),
                eps=eps,
            )

            (loss + aux).backward()

            opt_verts.step()
            opt_other.step()

            sched_verts.step()

            # ----------------------------------------------------------
            # Adaptive: prune transparent + random-duplicate to refill
            # ----------------------------------------------------------
            if it >= remesh_start and (it % remesh_every == 0) and it <= remesh_end:
                with torch.no_grad():
                    # Linear ramp of prune threshold over the tail of the
                    # remesh window: prune_thresh -> prune_thresh_final
                    # between prune_ramp_start and remesh_end.
                    if it <= args.prune_ramp_start:
                        cur_prune = prune_thresh
                    elif it >= remesh_end:
                        cur_prune = float(args.prune_thresh_final)
                    else:
                        t = (it - args.prune_ramp_start) / max(
                            1, remesh_end - args.prune_ramp_start)
                        cur_prune = prune_thresh + t * (
                            float(args.prune_thresh_final) - prune_thresh)

                    opacity = torch.sigmoid(point_opacity_logit.detach())
                    alive = opacity >= cur_prune
                    alive_idx = alive.nonzero(as_tuple=False).squeeze(1)
                    dead_idx = (~alive).nonzero(as_tuple=False).squeeze(1)
                    n_dead = int(dead_idx.numel())
                    n_alive = int(alive_idx.numel())

                    if n_dead > 0 and n_alive > 0:
                        # Uniform-random parents, with replacement, and the
                        # opacity is copied from the parents.  Another ablation
                        # compares against the MCMC primitive relocation
                        # strategy that considers primitive opacity.
                        sample = torch.randint(0, n_alive, (n_dead,), device=device)
                        src_idx = alive_idx[sample]

                        cur_lr_verts = float(opt_verts.param_groups[0]["lr"])
                        dup_jitter_std = float(args.dup_jitter_lr_frac) * cur_lr_verts
                        jitter = torch.randn(n_dead, 3, device=device) * dup_jitter_std
                        verts.data[dead_idx] = verts.data[src_idx] + jitter

                        sh_dc.data[dead_idx] = sh_dc.data[src_idx]
                        if sh_rest is not None:
                            sh_rest.data[dead_idx] = sh_rest.data[src_idx]
                        point_opacity_logit.data[dead_idx] = point_opacity_logit.data[src_idx]

                        _zero_adam_state(verts, dead_idx)
                        _zero_adam_state(sh_dc, dead_idx)
                        if sh_rest is not None:
                            _zero_adam_state(sh_rest, dead_idx)
                        _zero_adam_state(point_opacity_logit, dead_idx)

                        needs_topology_update = True
                        pbar.write(
                            f"[remesh] it={it}  alive={n_alive}  dead->dup={n_dead}  "
                            f"total={N}  jitter_std={dup_jitter_std:.3e}  "
                            f"prune_thresh={cur_prune:.4f}")

            # Logging.
            loss_val = float(loss.detach().cpu())
            del loss
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.3e}", P=N)

            if it in time_iters:
                elapsed = time.time() - t_train_start - t_eval_total
                timing[f"iter_{it}"] = round(elapsed, 2)
                pbar.write(f"[time] iter {it}: {elapsed:.1f}s")

            if it in ckpt_iters:
                save_ckpt(it)

            # Periodic test-set MAE (log-spaced schedule).
            if do_test_mae and it in mae_eval_iters:
                t_eval_start = time.time()
                with torch.no_grad():
                    mae_sum = 0.0
                    for ti in range(num_test_views):
                        va = fuzzydr.eval_sh_attrs(verts, get_sh(), radius_zero, campos=test_eyes[ti])
                        rgba = fuzzydr.rasterize_points(
                            va, viewproj=test_mvps[ti], campos=test_eyes[ti],
                            points=points_cur,
                            point_opacity=torch.sigmoid(point_opacity_logit),
                            width=test_W * 2, height=test_H * 2,
                            tau=0.5, seed=0, white_bg=True,
                        )
                        rgba = fuzzydr.msaa_downsample_rgba(rgba)
                        pred = rgba[..., :3].clamp(0, 1)
                        mae_sum += torch.abs(pred - test_images[ti]).mean().item()
                    needs_topology_update = True
                    mae_val = mae_sum / num_test_views
                mae_log.append((it, mae_val))
                t_eval_total += time.time() - t_eval_start
                pbar.write(f"[test_mae] iter {it}: {mae_val:.6f}")

        # Save final timing.
        timing["total"] = round(time.time() - t_train_start - t_eval_total, 2)
        timing_path = os.path.join(out_dir, "timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"Timing saved: {timing_path}")

        # Save loss curve.
        loss_path = os.path.join(out_dir, "loss.txt")
        with open(loss_path, "w") as f:
            for i, l in enumerate(losses, start=1):
                f.write(f"{i}\t{l}\n")
        print(f"Loss saved: {loss_path}")

        # Save test MAE log.
        if do_test_mae and mae_log:
            mae_path = os.path.join(out_dir, "test_mae.txt")
            with open(mae_path, "w") as f:
                f.write("iter\tmae\n")
                for it_m, mae_m in mae_log:
                    f.write(f"{it_m}\t{mae_m:.8f}\n")
            print(f"Test MAE saved: {mae_path}")

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
