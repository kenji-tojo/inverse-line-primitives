#!/usr/bin/env python3
"""Triangle-primitive training on the Shelly scenes.

Triangle baseline for comparison against the line primitives: same spherical
harmonics, learning-rate schedule, loss, and checkpoint schedule.  Each seed
point spawns one right triangle whose legs are random orthonormal 3D vectors
of length 0.5 x mean KNN-1 distance.

Topology updates use discrete_updates.remesh, which prunes faces below the
opacity threshold or shorter than `length_threshold`, then splits the longest
edges until the free vertex slots are refilled.  The two children of a split
share the new midpoint vertex.

Usage:
    python scripts/baselines/train_shelly_triangles.py --scene khady
    python scripts/baselines/train_shelly_triangles.py --scene khady --iters 50000 --target_num_verts 2000001

Run every scene:
    bash scripts/baselines/run_shelly_triangles.sh

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
import discrete_updates
import fetch_data
from utils import (
    spatial_lr_scale,
    select_device,
    make_triangle_primitives_from_ply,
    load_nerf_synthetic,
    load_colmap_dataset,
    SH_BAND_SIZES,
    SH_NUM_COEFFS,
    save_checkpoint,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Triangle primitive training.")

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
    ap.add_argument("--target_num_verts", type=int, default=2_000_001,
                    help="Vertex budget (num_tris = target // 3). "
                         "Default 2_000_001 -> 666_667 tris x 3 verts.")
    ap.add_argument("--init_samples", type=str, default=None,
                    help="Path to pre-sampled PLY of seed origins "
                         "(auto-resolved for shelly using --init_mesh_band + num_tris)")
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
    ap.add_argument("--lr_fop", type=float, default=0.05,
                    help="Face-opacity logit LR")

    # Remeshing
    ap.add_argument("--remesh_start", type=int, default=500)
    ap.add_argument("--remesh_every", type=int, default=100)
    ap.add_argument("--remesh_end", type=int, default=35_000)
    ap.add_argument("--prune_thresh_final", type=float, default=0.5,
                    help="Target prune threshold at end of ramp window.")
    ap.add_argument("--prune_ramp_start", type=int, default=25_000,
                    help="Iter at which prune_thresh begins ramping linearly "
                         "from prune_thresh to prune_thresh_final, finishing "
                         "at remesh_end.")
    ap.add_argument("--prune_thresh", type=float, default=0.05,
                    help="Opacity threshold: faces with opacity < this are pruned.")
    ap.add_argument("--length_threshold", type=float, default=-1.0,
                    help="Short-edge prune threshold: faces whose longest edge is "
                         "below this value are removed (default: init_edge_length/2; 0=off)")

    # Regularization
    ap.add_argument("--lambda_dirichlet", type=float, default=1e-3)

    # Camera
    ap.add_argument("--z_near", type=float, default=0.01)
    ap.add_argument("--z_far", type=float, default=100.0)

    # Checkpoints & output
    ap.add_argument("--ckpt_iters", nargs="+", type=int, default=[7000, 50000])
    ap.add_argument("--time_iters", nargs="+", type=int, default=[50000])
    ap.add_argument("--out_dir", type=str, default="results/shelly_triangles")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--no_test_mae", action="store_true",
                    help="Disable periodic test-set MAE evaluation")

    args = ap.parse_args()

    # Triangle count from vertex budget.
    num_tris = int(args.target_num_verts) // 3
    target_verts = num_tris * 3  # may differ from arg by at most 2

    # Auto-resolve pre-sampled PLY for shelly.
    if args.dataset == "shelly" and args.init_samples is None:
        args.init_samples = os.path.join(
            "datasets", "fuzzy_dataset", "coarse", "neus2_shelly_dtu_15000steps",
            f"shelly_{args.scene}", "band_samples",
            f"band{float(args.init_mesh_band):.3f}_n{num_tris}.ply",
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
    if args.scene_root is not None:
        scene_root = args.scene_root
    elif args.dataset == "shelly":
        scene_root = os.path.join("datasets", "shelly_data_release", args.scene)
    else:
        raise SystemExit(
            f"--scene_root is required for --dataset {args.dataset} "
            f"(only shelly has a default location).")

    if args.dataset == "360_v2":
        images_cpu, mvps_cpu, eyes_cpu, width, height, _ = load_colmap_dataset(
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
    # Build triangle primitives
    # ------------------------------------------------------------------
    if args.init_samples is None:
        raise ValueError("Must provide --init_samples (or use --dataset shelly for auto-resolve).")
    if not os.path.isfile(args.init_samples):
        raise FileNotFoundError(
            f"PLY not found: {args.init_samples}.  Pre-sample via "
            f"`python datasets/fuzzy_dataset/coarse/band_sampling.py --band {args.init_mesh_band} "
            f"--n_points {num_tris} --mesh <coarse.obj> --out <out.ply>`."
        )

    tri_verts_cpu, tri_faces_cpu, init_edge_length = make_triangle_primitives_from_ply(
        num_tris, args.init_samples,
    )
    print(f"[init] Triangles from PLY: {args.init_samples}  "
          f"({num_tris} tris, {tri_verts_cpu.shape[0]} verts, "
          f"init_edge_length={init_edge_length:.6f})")

    length_threshold = float(args.length_threshold) if args.length_threshold >= 0 else init_edge_length / 2.0
    print(f"[init] length_threshold={length_threshold:.6f}")
    print(f"Remesh: fixed vertex budget {target_verts}, every {remesh_every} iters, "
          f"{remesh_start}-{remesh_end}")
    print(f"[remesh] prune_thresh ramp: {prune_thresh:.3f} -> {float(args.prune_thresh_final):.3f} "
          f"over iters {int(args.prune_ramp_start)}-{remesh_end} (linear)")

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
    faces_cur = tri_faces_cpu.to(device=device).contiguous()

    N = int(tri_verts_cpu.shape[0])
    M_f = int(faces_cur.shape[0])

    # Trainables.
    verts = torch.nn.Parameter(
        tri_verts_cpu.to(device).clone().detach().contiguous().requires_grad_(True))
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
    face_opacity_logit = torch.nn.Parameter(
        torch.full((M_f,), _init_opacity_logit, dtype=torch.float32, device=device).contiguous().requires_grad_(True))

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def rebuild_opt():
        opt_verts = fuzzydr.optimize.VectorAdam([verts], lr=lr_verts_init)
        sh_groups = [{"params": [sh_dc], "lr": float(args.lr_sh_dc)}]
        if sh_rest is not None:
            sh_groups.append({"params": [sh_rest], "lr": float(args.lr_sh_rest)})
        opt_other = torch.optim.Adam(
            sh_groups + [{"params": [face_opacity_logit], "lr": float(args.lr_fop)}],
            foreach=False,
        )
        return opt_verts, opt_other

    def replace_tensor_to_optimizer(opt, old_param, new_param, vert_origin=None):
        """Swap a parameter in the optimizer, transferring Adam/VectorAdam state.

        If vert_origin is provided (int32 per new vertex, -1 for fresh slots),
        survived entries inherit momenta; fresh entries get zero state.
        If vert_origin is None, state is fully reset (for face-indexed params
        whose ordering is not preserved across remesh).
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
    # Checkpoint helper (triangle format: faces > 0, lines empty)
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
                faces=faces_cur.detach().cpu(),
                face_opacity_logit=face_opacity_logit.detach().cpu(),
                color_mode="sh",
                sh_degree=args.sh_degree,
                extra_meta={
                    "scene": args.scene, "split": args.split,
                    "dataset": args.dataset, "iters": it,
                    "prune_thresh_train": prune_thresh,
                    "resolution": int(args.resolution),
                    "primitive": "triangles",
                    "init_edge_length": float(init_edge_length),
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
            radius_zero = torch.zeros(int(verts.shape[0]), dtype=torch.float32, device=device)
            va = fuzzydr.eval_sh_attrs(verts, get_sh(), radius_zero, campos=eye)
            face_opacity = torch.sigmoid(face_opacity_logit)

            img = fuzzydr.msaa_downsample_rgba(fuzzydr.rasterize(
                va,
                viewproj=mvp, campos=eye,
                faces=faces_cur if needs_topology_update else faces_cur.shape[0],
                face_opacity=face_opacity,
                width=width * 2, height=height * 2,
                tau=-1.0, seed=int(it),
                white_bg=True,
            ))[..., :3].contiguous()
            needs_topology_update = False

            gt = images[view_idx]
            diff = torch.abs(img - gt)

            l1_loss = diff.mean()
            if use_ssim:
                ssim_loss = 0.5 * (1.0 - ssim(
                    img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0))
                loss = 0.8 * l1_loss + 0.2 * ssim_loss
            else:
                loss = l1_loss

            error = diff.mean(dim=-1).detach().contiguous()
            aux = fuzzydr.opacity_mask_aux_loss(
                face_opacity=face_opacity,
                error=fuzzydr.upsample2x2_scalar(error),
                eps=eps,
            )

            # Dirichlet regularizer (mean over the 3 edges of each face).
            dirichlet_reg = torch.tensor(0.0, device=device)
            if args.lambda_dirichlet > 0 and faces_cur.shape[0] > 0:
                fv = verts[faces_cur.long()]  # [F, 3, 3]
                e01 = fv[:, 1] - fv[:, 0]
                e12 = fv[:, 2] - fv[:, 1]
                e20 = fv[:, 0] - fv[:, 2]
                edge_sq = torch.cat([
                    e01.pow(2).sum(dim=-1),
                    e12.pow(2).sum(dim=-1),
                    e20.pow(2).sum(dim=-1),
                ], dim=0)
                dirichlet_reg = float(args.lambda_dirichlet) * edge_sq.mean()

            (loss + aux + dirichlet_reg).backward()

            opt_verts.step()
            opt_other.step()

            sched_verts.step()

            # ----------------------------------------------------------
            # Remesh (fixed vertex budget, pure-PyTorch triangle rule)
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

                    Mf0 = int(faces_cur.shape[0])
                    N0 = int(verts.shape[0])

                    old_verts = verts
                    old_sh_dc = sh_dc
                    old_sh_rest = sh_rest
                    old_face_opacity_logit = face_opacity_logit

                    # Pack SH into a flat [N, 3*num_sh] buffer for interpolation.
                    if sh_rest is not None:
                        _sh_all = torch.cat([sh_dc, sh_rest], dim=-1)
                    else:
                        _sh_all = sh_dc
                    sh_flat = _sh_all.detach().reshape(N0, -1).contiguous()

                    (out_pos, out_attrs,
                     out_fac, out_fop,
                     out_vert_origin,
                     ) = discrete_updates.remesh(
                         verts.detach().clone(), sh_flat.clone(),
                         faces_cur.detach().clone(),
                         torch.sigmoid(face_opacity_logit.detach()).clone(),
                         opacity_threshold=cur_prune,
                         length_threshold=length_threshold,
                         shared_split=True,
                     )

                    N1 = int(out_pos.shape[0])
                    Mf1 = int(out_fac.shape[0])
                    old_idx_t = out_vert_origin.to(torch.int64)

                    faces_cur = out_fac
                    verts = torch.nn.Parameter(out_pos.contiguous().requires_grad_(True))

                    _sh_full = out_attrs.reshape(N1, 3, num_sh_coeffs)
                    sh_dc = torch.nn.Parameter(
                        _sh_full[:, :, :num_sh_dc].contiguous().requires_grad_(True))
                    if num_sh_rest > 0:
                        sh_rest = torch.nn.Parameter(
                            _sh_full[:, :, num_sh_dc:].contiguous().requires_grad_(True))

                    # Invert opacity to logit.  face_opacity optimizer state
                    # is refreshed (face buffer ordering changes across remesh).
                    fop = out_fop.clamp(1e-6, 1.0 - 1e-6)
                    face_opacity_logit = torch.nn.Parameter(
                        torch.log(fop / (1.0 - fop)).requires_grad_(True))

                    # Preserve vertex-indexed Adam state for intact verts.
                    replace_tensor_to_optimizer(opt_verts, old_verts, verts, old_idx_t)
                    replace_tensor_to_optimizer(opt_other, old_sh_dc, sh_dc, old_idx_t)
                    if sh_rest is not None:
                        replace_tensor_to_optimizer(opt_other, old_sh_rest, sh_rest, old_idx_t)
                    # Reset face_opacity optimizer state (no vert_origin mapping).
                    replace_tensor_to_optimizer(opt_other, old_face_opacity_logit, face_opacity_logit)

                    needs_topology_update = True

                    pbar.write(
                        f"[remesh] it={it}  faces: {Mf0}->{Mf1}  "
                        f"verts: {N0}->{N1}  prune_thresh={cur_prune:.4f}")

            # Logging.
            loss_val = float(loss.detach().cpu())
            del loss
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.3e}", F=int(faces_cur.shape[0]), V=int(verts.shape[0]))

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
                    rad_slot = torch.zeros(N_here, dtype=torch.float32, device=device)
                    for ti in range(num_test_views):
                        va = fuzzydr.eval_sh_attrs(verts, get_sh(), rad_slot, campos=test_eyes[ti])
                        pred = fuzzydr.msaa_downsample_rgba(fuzzydr.rasterize(
                            va, viewproj=test_mvps[ti], campos=test_eyes[ti],
                            faces=faces_cur,
                            face_opacity=torch.sigmoid(face_opacity_logit),
                            width=test_W * 2, height=test_H * 2,
                            tau=0.5, seed=0, white_bg=True,
                        )).clamp(0, 1)
                        mae_sum += torch.abs(pred[..., :3] - test_images[ti]).mean().item()
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
