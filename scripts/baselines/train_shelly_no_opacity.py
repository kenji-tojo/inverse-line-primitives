#!/usr/bin/env python3
"""no-opacity ablation: line opacity is fixed to 1 and never optimized.

Ablates the opacity term of the differentiable rendering formulation.  Every
line is fully opaque for the whole run, which removes three things relative to
the full method (scripts/train_shelly_lines.py):

  1. There is no opacity parameter, no optimizer entry for it, and no
     opacity-mask auxiliary loss.
  2. Rasterization uses a deterministic tau=0.5 throughout, whereas the full
     method trains with tau=-1.0, which draws a random threshold to get a
     gradient through opacity.
  3. discrete_updates.relining runs with opacity_threshold=-1.0, so lines are
     never pruned by opacity.  Pruning by length, endpoint snapping, and
     longest-edge splitting to the vertex budget are unchanged.

Everything else - seeding, SH schedule, learning rates, Dirichlet weight,
remesh window, and checkpoint schedule - matches the full method.

Usage:
    python scripts/baselines/train_shelly_no_opacity.py --scene khady

Run every scene:
    bash scripts/baselines/run_shelly_no_opacity.sh

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

# Nominal line radius recorded in the checkpoint, in world units
# (5e-4 = 1 mm diameter).  THIS IS A DUMMY VALUE: training renders lines as
# 1-px Bresenham segments, which have no width, so no radius is ever
# optimized or used by the renderer.  It exists only so downstream tools that
# expect a radius have a sensible number to draw with.
LINE_RADIUS = 5e-4

# Opacity logit written to the checkpoint.  Training uses a constant opacity of
# 1, but the checkpoint format stores a logit that eval.py passes through a
# sigmoid, so store the logit whose sigmoid is 1 - 1e-6.  The 1e-6 shortfall is
# below the renderer's blending precision.
SAVED_OPACITY_LOGIT = float(torch.tensor(1.0 - 1e-6).logit())

try:
    from pytorch_msssim import ssim
    use_ssim = True
except ImportError:
    print("[torch] WARNING: pytorch_msssim not found, SSIM loss disabled.")
    use_ssim = False


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuzzydr
import discrete_updates
import fetch_data
from utils import (
    spatial_lr_scale,
    select_device,
    make_line_primitives_from_ply,
    knn1_dists,
    load_nerf_synthetic,
    load_colmap_dataset,
    SH_BAND_SIZES,
    SH_NUM_COEFFS,
    save_checkpoint,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Line primitive training.")

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

    # Primitive init
    ap.add_argument("--num_lines", type=int, default=1_000_000)
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
    # No --lr_lop: opacity is fixed to 1 and never enters the optimizer.

    # Remeshing.  The full method's opacity-pruning options (--prune_thresh,
    # --prune_thresh_final, --prune_ramp_start) are absent: with opacity fixed
    # there is nothing to prune by.
    ap.add_argument("--remesh_start", type=int, default=500)
    ap.add_argument("--remesh_every", type=int, default=100)
    ap.add_argument("--remesh_end", type=int, default=35_000)
    ap.add_argument("--length_threshold", type=float, default=-1.0,
                    help="Short-edge collapse threshold (default: init_line_length/2; 0=off)")
    ap.add_argument("--snap_endpoints", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target_num_verts", type=int, default=2_000_000)

    # Regularization (only Dirichlet is on by default)
    ap.add_argument("--lambda_dirichlet", type=float, default=1e-3)

    # Camera
    ap.add_argument("--z_near", type=float, default=0.01)
    ap.add_argument("--z_far", type=float, default=100.0)

    # Checkpoints & output
    ap.add_argument("--ckpt_iters", nargs="+", type=int, default=[7000, 50000],
                    help="Iterations at which to save checkpoints")
    ap.add_argument("--time_iters", nargs="+", type=int, default=[50000],
                    help="Iterations at which to log cumulative training time")
    ap.add_argument("--out_dir", type=str, default="results/shelly_no_opacity")
    ap.add_argument("--no_test_mae", action="store_true",
                    help="Disable periodic test-set MAE evaluation")

    args = ap.parse_args()

    # Auto-resolve pre-sampled PLY for shelly.
    if args.dataset == "shelly" and args.init_samples is None:
        args.init_samples = os.path.join(
            "datasets", "fuzzy_dataset", "coarse", "neus2_shelly_dtu_15000steps",
            f"shelly_{args.scene}", "band_samples",
            f"band{float(args.init_mesh_band):.3f}_n{int(args.num_lines)}.ply",
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
    target_verts = int(args.target_num_verts)

    ckpt_iters = set(args.ckpt_iters)
    time_iters = sorted(args.time_iters)
    timing: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Load data + build line primitives
    # ------------------------------------------------------------------
    sfm_pts = None  # SfM points for COLMAP datasets

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

    # Load test split for periodic MAE evaluation.
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
        # Log-spaced eval schedule (evenly spaced in log space).
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

    if sfm_pts is not None:
        # COLMAP dataset: initialize line primitives directly from SfM points.
        num_sfm = sfm_pts.shape[0]
        num_lines = num_sfm  # one line per SfM point
        midpoints = torch.as_tensor(sfm_pts, dtype=torch.float32)
        nn_dists = knn1_dists(midpoints).clamp_min(1e-7)
        init_line_length = float(nn_dists.mean())
        dirs = torch.randn((num_lines, 3), dtype=torch.float32)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8) * (init_line_length * 0.5)
        line_verts_cpu = torch.stack([midpoints - dirs, midpoints + dirs], dim=1).reshape(num_lines * 2, 3).contiguous()
        base = torch.arange(num_lines, dtype=torch.int64) * 2
        line_edges_cpu = torch.stack([base, base + 1], dim=1).to(torch.uint32).contiguous()
        print(f"[init] Lines from SfM: {num_sfm} points -> {num_lines} lines, init_line_length={init_line_length:.6f} (mean KNN)")
    elif args.init_samples is not None:
        line_verts_cpu, line_edges_cpu, init_line_length = make_line_primitives_from_ply(
            int(args.num_lines), args.init_samples,
        )
        print(f"[init] Lines from PLY: {args.init_samples}")
    else:
        raise SystemExit(
            "No seed points: pass --init_samples <points.ply> "
            "(any point cloud works, e.g. band samples around a coarse mesh).")

    length_threshold = float(args.length_threshold) if args.length_threshold >= 0 else init_line_length / 2.0

    print(f"[init] init_line_length={init_line_length:.6f} (mean KNN)  length_threshold={length_threshold:.6f}")
    print(f"Init: {line_edges_cpu.shape[0]} line primitives ({line_verts_cpu.shape[0]} verts)")
    print(f"Remesh: target={target_verts} verts, every {remesh_every} iters, {remesh_start}-{remesh_end}")
    print("[ablation] line opacity fixed to 1; tau=0.5; opacity pruning disabled.")

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
    lines_cur = line_edges_cpu.to(device=device).contiguous()

    N = int(line_verts_cpu.shape[0])

    # Trainables.
    verts = torch.nn.Parameter(
        line_verts_cpu.to(device).clone().detach().contiguous().requires_grad_(True))
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

    # Ablation: no opacity parameter.  Every render and every remesh below
    # passes a constant ones-vector sized to the current line count.
    def ones_opacity():
        return torch.ones(int(lines_cur.shape[0]), dtype=torch.float32, device=device)

    # Bresenham lines ignore the per-vertex radius slot, so no radius
    # parameter is created or optimized.  The rasterizer call packs a
    # zeros placeholder for the slot.

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def rebuild_opt():
        opt_verts = fuzzydr.optimize.VectorAdam([verts], lr=lr_verts_init)
        sh_groups = [{"params": [sh_dc], "lr": float(args.lr_sh_dc)}]
        if sh_rest is not None:
            sh_groups.append({"params": [sh_rest], "lr": float(args.lr_sh_rest)})
        opt_other = torch.optim.Adam(sh_groups, foreach=False)
        return opt_verts, opt_other

    def replace_tensor_to_optimizer(opt, old_param, new_param, vert_origin=None):
        """Swap a parameter in the optimizer, transferring state for survived vertices.

        Surviving vertices keep their Adam state; new vertices
        (vert_origin == -1) start from zero.
        """
        for group in opt.param_groups:
            if not any(p is old_param for p in group['params']):
                continue
            group['params'] = [new_param if p is old_param else p
                               for p in group['params']]
            old_state = opt.state.pop(old_param, None)
            if old_state is None or 'step' not in old_state:
                return
            is_vadam = 'g1' in old_state
            if vert_origin is not None:
                survived = vert_origin >= 0
                src = vert_origin.clamp(min=0)
                mask = survived
                def _gather(t):
                    out = t.index_select(0, src)
                    m = mask
                    for _ in range(out.ndim - 1):
                        m = m.unsqueeze(-1)
                    return out * m
            else:
                _gather = None
            if is_vadam:
                g1 = _gather(old_state['g1']) if _gather else torch.zeros_like(new_param)
                g2 = _gather(old_state['g2']) if _gather else torch.zeros_like(new_param[..., :1])
                opt.state[new_param] = {
                    'step': old_state['step'], 'g1': g1, 'g2': g2,
                }
            else:
                ea = _gather(old_state['exp_avg']) if _gather else torch.zeros_like(new_param)
                es = _gather(old_state['exp_avg_sq']) if _gather else torch.zeros_like(new_param)
                opt.state[new_param] = {
                    'step': old_state['step'], 'exp_avg': ea, 'exp_avg_sq': es,
                }
            return

    needs_topology_update = True

    # ------------------------------------------------------------------
    # Checkpoint helper
    # ------------------------------------------------------------------
    def save_ckpt(it: int):
        with torch.no_grad():
            _N = int(verts.shape[0])
            _sh = (torch.cat([sh_dc, sh_rest], dim=-1) if sh_rest is not None else sh_dc).detach().cpu()
            if num_sh_coeffs < SH_NUM_COEFFS:
                _sh = torch.cat([_sh, torch.zeros(_N, 3, SH_NUM_COEFFS - num_sh_coeffs)], dim=-1)
            _sh_int = _sh.permute(0, 2, 1).reshape(_N, SH_NUM_COEFFS * 3)
            ckpt_path = os.path.join(out_dir, f"{args.dataset}_{args.scene}_{it:05d}.npz")
            save_checkpoint(
                ckpt_path,
                verts=verts.detach().cpu(),
                colors=_sh_int,
                lines=lines_cur.detach().cpu(),
                # Ablation: constant logit, so eval.py's sigmoid reads ~1.
                line_opacity_logit=torch.full((int(lines_cur.shape[0]),),
                                              SAVED_OPACITY_LOGIT, dtype=torch.float32),
                radius=torch.full((_N,), LINE_RADIUS, dtype=torch.float32),
                color_mode="sh",
                sh_degree=3,
                extra_meta={
                    "scene": args.scene, "split": args.split,
                    "dataset": args.dataset, "iters": it,
                    "resolution": int(args.resolution),
                    "bresen_lines": True,
                },
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    fuzzydr.init()
    try:
        opt_verts, opt_other = rebuild_opt()

        decay_gamma = (lr_verts_final / lr_verts_init) ** (1.0 / max(iters, 1))
        sched_verts = torch.optim.lr_scheduler.ExponentialLR(opt_verts, gamma=decay_gamma)
        print(f"[lr] exp ramp: {lr_verts_init:.6g} -> {lr_verts_final:.6g} over {iters} iters  (gamma={decay_gamma:.8f})")

        losses: list[float] = []
        t_eval_total = 0.0  # accumulated test-eval time to subtract from training clock
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
            # Lines render as 1-px Bresenham segments, which ignore the
            # per-vertex radius slot entirely.  The value below is a dummy
            # placeholder only, NOT a trained or meaningful width.
            rad_slot = torch.ones(int(verts.shape[0]), dtype=torch.float32, device=device)
            va = fuzzydr.eval_sh_attrs(verts, get_sh(), rad_slot, campos=eye)

            # Ablation: opacity fixed to 1, and a deterministic tau=0.5 (the
            # eval value), whereas the full method trains with tau=-1.0 to
            # sample a random threshold.
            img = fuzzydr.msaa_downsample_rgba(fuzzydr.rasterize(
                va,
                viewproj=mvp, campos=eye,
                lines=lines_cur if needs_topology_update else lines_cur.shape[0],
                line_opacity=ones_opacity(),
                width=width * 2, height=height * 2,
                tau=0.5, seed=int(it),
                white_bg=True,
                bresen_lines=True,
            )).contiguous()
            needs_topology_update = False

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

            # Ablation: the opacity-mask auxiliary loss is dropped; with no
            # opacity parameter there is nothing for it to steer.

            # Dirichlet regularizer.
            dirichlet_reg = torch.tensor(0.0, device=device)
            if args.lambda_dirichlet > 0 and lines_cur.shape[0] > 0:
                edge_vecs = verts[lines_cur[:, 0].long()] - verts[lines_cur[:, 1].long()]
                dirichlet_reg = float(args.lambda_dirichlet) * edge_vecs.pow(2).sum(dim=-1).mean()

            (loss + dirichlet_reg).backward()

            opt_verts.step()
            opt_other.step()

            sched_verts.step()

            # ----------------------------------------------------------
            # Remesh
            # ----------------------------------------------------------
            if it >= remesh_start and (it % remesh_every == 0) and it <= remesh_end:
                with torch.no_grad():
                    Ml0 = int(lines_cur.shape[0])
                    N0 = int(verts.shape[0])

                    old_verts = verts
                    old_sh_dc = sh_dc
                    old_sh_rest = sh_rest

                    # Allow the vertex buffer to grow by up to 5%, capped at
                    # the target (a no-op when it already sits at target).
                    cur_target_v = min(target_verts, int(1.05 * N0))

                    if sh_rest is not None:
                        _sh_all = torch.cat([sh_dc, sh_rest], dim=-1)
                    else:
                        _sh_all = sh_dc
                    sh_flat = _sh_all.detach().reshape(N0, -1).contiguous()

                    # Ablation: opacity is 1 everywhere and the threshold is
                    # negative, so nothing is pruned by opacity.  Length
                    # pruning, snapping and splitting are unchanged.
                    (out_pos, out_sh,
                     out_lin, _,
                     out_vert_origin,
                     ) = discrete_updates.relining(
                         verts.detach(), sh_flat,
                         lines_cur.detach(), ones_opacity(),
                         opacity_threshold=-1.0,
                         length_threshold=length_threshold,
                         snap_endpoints=args.snap_endpoints,
                         target_num_verts=cur_target_v,
                     )

                    N1 = int(out_pos.shape[0])
                    Ml1 = int(out_lin.shape[0])
                    old_idx_t = out_vert_origin.to(torch.int64)

                    lines_cur = out_lin
                    verts = torch.nn.Parameter(out_pos.contiguous().requires_grad_(True))

                    _sh_full = out_sh.reshape(N1, 3, num_sh_coeffs)
                    sh_dc = torch.nn.Parameter(
                        _sh_full[:, :, :num_sh_dc].contiguous().requires_grad_(True))
                    if num_sh_rest > 0:
                        sh_rest = torch.nn.Parameter(
                            _sh_full[:, :, num_sh_dc:].contiguous().requires_grad_(True))

                    replace_tensor_to_optimizer(opt_verts, old_verts, verts, old_idx_t)
                    replace_tensor_to_optimizer(opt_other, old_sh_dc, sh_dc, old_idx_t)
                    if sh_rest is not None:
                        replace_tensor_to_optimizer(opt_other, old_sh_rest, sh_rest, old_idx_t)

                    needs_topology_update = True

                    pbar.write(
                        f"[remesh] it={it}  lines: {Ml0}->{Ml1}  "
                        f"verts: {N0}->{N1}  target={cur_target_v}  "
                        f"(opacity pruning disabled)")

            # Logging.
            loss_val = float(loss.detach().cpu())
            del loss
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.3e}", L=int(lines_cur.shape[0]), V=int(verts.shape[0]))

            # Timing milestones (subtract eval overhead).
            if it in time_iters:
                elapsed = time.time() - t_train_start - t_eval_total
                timing[f"iter_{it}"] = round(elapsed, 2)
                pbar.write(f"[time] iter {it}: {elapsed:.1f}s")

            # Checkpoints.
            if it in ckpt_iters:
                save_ckpt(it)

            # Periodic test-set MAE (log-spaced schedule).
            if do_test_mae and it in mae_eval_iters:
                t_eval_start = time.time()
                with torch.no_grad():
                    mae_sum = 0.0
                    N_here = int(verts.shape[0])
                    rad_slot = torch.ones(N_here, dtype=torch.float32, device=device)
                    for ti in range(num_test_views):
                        va = fuzzydr.eval_sh_attrs(verts, get_sh(), rad_slot, campos=test_eyes[ti])
                        pred = fuzzydr.msaa_downsample_rgba(fuzzydr.rasterize(
                            va, viewproj=test_mvps[ti], campos=test_eyes[ti],
                            lines=lines_cur, line_opacity=ones_opacity(),
                            width=test_W * 2, height=test_H * 2,
                            tau=0.5, seed=0, white_bg=True,
                            bresen_lines=True,
                        )).clamp(0, 1)
                        mae_sum += torch.abs(pred - test_images[ti]).mean().item()
                    needs_topology_update = True  # topology cache invalidated
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
