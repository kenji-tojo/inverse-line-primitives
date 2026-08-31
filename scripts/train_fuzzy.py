# train_fuzzy.py
#
# Line-primitive optimization on the Fuzzy capture dataset.
#
# Each scene provides camera transforms and RGBA images, composited over a
# black background and rendered with the fuzzydr differentiable rasterizer.
# Topology is updated periodically by discrete_updates.relining.
#
# The scene's training views and seed points are downloaded from the Fuzzy
# dataset on startup if they are missing (fetch_data.py); --no_fetch turns
# that off.
#
# Data layout:
#   --scene_root        datasets/fuzzy_dataset/capture/<scene>
#   --transforms_train  transforms_images_4_masked_train.json
#   --transforms_test   transforms_images_4_masked_test.json
#   --init_samples      datasets/fuzzy_dataset/coarse/neus2_fuzzy_dtu_15000steps_images_4/
#                         <scene>/band_samples/band0.100_n1000000.ply
#
# Output:
#   <out_dir>/<scene>/
#     options.json  loss.txt
#     fuzzy_<scene>.mp4               optimization video from the hero view
#     fuzzy_<scene>_50000.npz         final checkpoint
#     eval_50000/                     metrics.json  train/test/combined metrics
#                                     per_view.json
#                                     renders/, gt_masked/  every train+test view
#
# Run one scene:
#   python scripts/train_fuzzy.py --device cuda --scene flowers --useall
#
# Run every scene:
#   bash scripts/run_fuzzy.sh
#
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import imageio.v2 as imageio
import torch
from tqdm import tqdm

# Nominal line radius recorded in the checkpoint, in world units
# (5e-4 = 1 mm diameter).  THIS IS A DUMMY VALUE: training renders lines as
# 1-px Bresenham segments, which have no width, so no radius is ever
# optimized or used by the renderer.  It exists only so downstream tools that
# expect a radius have a sensible number to draw with.
LINE_RADIUS = 5e-4

try:
    from pytorch_msssim import ssim
    use_ssim = True
except ImportError:
    print("[torch] WARNING: pytorch_msssim not found, SSIM loss disabled.")
    use_ssim = False

# fuzzydr renderer + pure-PyTorch discrete updates.
import fuzzydr
import discrete_updates
import fetch_data
from eval import lpips_vgg

from utils import (
    spatial_lr_scale,
    to_u8,
    select_device,
    make_line_primitives_from_ply,
    load_nerf_per_frame,
    SH_BAND_SIZES,
    SH_NUM_COEFFS,
    save_checkpoint,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Line primitives on the Fuzzy dataset."
    )

    # Data
    ap.add_argument("--scene", type=str, default="cactus1",
                    help="Fuzzy scene name, lowercase as in datasets/fuzzy_dataset/capture/. "
                         "Used to template scene_root and init_samples when "
                         "those are left at their defaults.")
    ap.add_argument("--scene_root", type=str, default=None,
                    help="Path to the Fuzzy scene root (contains transforms_*.json + images_4_rgba/). "
                         "Default: datasets/fuzzy_dataset/capture/<scene>.")
    ap.add_argument("--transforms_train", type=str,
                    default="transforms_images_4_masked_train.json",
                    help="Filename inside scene_root for the TRAIN split. "
                         "Frames listed here are used for optimization.")
    ap.add_argument("--transforms_test", type=str,
                    default="transforms_images_4_masked_test.json",
                    help="Filename inside scene_root for the held-out TEST split. "
                         "Final PSNR/SSIM are reported on these views.")
    ap.add_argument("--useall", action="store_true", default=False,
                    help="Optimize on TRAIN+TEST views combined. The held-out "
                         "test views are still used for the final PSNR/SSIM "
                         "evaluation at the end (same as the default mode).")
    ap.add_argument("--sh_degree", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--sh_upgrade_every", type=int, default=1000)

    # Device
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)

    # Training
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--seed",  type=int, default=42)

    # Primitive init
    ap.add_argument("--num_lines", type=int, default=1_000_000)
    ap.add_argument("--init_samples", type=str, default=None,
                    help="Path to the band-sampled PLY (see the dataset's band_sampling.py). "
                         "Default: datasets/fuzzy_dataset/coarse/neus2_fuzzy_dtu_15000steps_images_4/<scene>/band_samples/band0.100_n1000000.ply.")
    ap.add_argument("--no_fetch", dest="fetch", action="store_false", default=True,
                    help="Do not download missing scene data from the "
                         "Fuzzy dataset; fail instead.")

    # Learning rates
    ap.add_argument("--lr_verts", type=float, default=1.6e-5)
    ap.add_argument("--lr_verts_final", type=float, default=3.2e-6)
    # Default = exp ramp.
    # Default = spatial LR scale on.
    ap.add_argument("--no_spatial_lr_scale", dest="spatial_lr_scale", action="store_false",
                    default=True,
                    help="Disable scaling of the position learning rate by camera extent.")
    ap.add_argument("--lr_sh_dc",   type=float, default=2.5e-3)
    ap.add_argument("--lr_sh_rest", type=float, default=1.25e-4)
    ap.add_argument("--lr_lop", type=float, default=0.05)

    # Remeshing
    ap.add_argument("--remesh_start", type=int, default=500)
    ap.add_argument("--remesh_every", type=int, default=100)
    ap.add_argument("--remesh_end",   type=int, default=35_000)
    ap.add_argument("--prune_thresh", type=float, default=0.05)
    ap.add_argument("--prune_thresh_final", type=float, default=0.5,
                    help="Target prune threshold at end of ramp window.")
    ap.add_argument("--prune_ramp_start", type=int, default=25_000,
                    help="Iter at which prune_thresh begins ramping linearly "
                         "from prune_thresh to prune_thresh_final, finishing "
                         "at remesh_end.")
    ap.add_argument("--length_threshold",  type=float, default=-1.0,
                    help="Default: init_line_length / 2; 0 = off.")
    # Default = endpoint snapping on.
    ap.add_argument("--no_snap_endpoints", dest="snap_endpoints", action="store_false",
                    default=True,
                    help="Disable endpoint merging at remesh time.")
    ap.add_argument("--target_num_verts", type=int, default=2_000_000)
    ap.add_argument("--lambda_dirichlet", type=float, default=1e-3)

    # Camera
    ap.add_argument("--z_near", type=float, default=0.01)
    ap.add_argument("--z_far",  type=float, default=100.0)

    # Video / snapshot
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--num_frames", type=int, default=1200,
                    help="Frames in the optimization video, spread evenly over "
                         "--iters. Default 1200 = 20 s at 60 fps.")
    ap.add_argument("--no_video", dest="video", action="store_false", default=True,
                    help="Skip the optimization video.")
    ap.add_argument("--video_view", type=int, default=0)
    ap.add_argument("--log_view", type=str, default=None,
                    help="Filename (e.g. '_P6A1141.png') of the view whose "
                         "trajectory is logged in the training video. Matched "
                         "against the active view pool by basename. "
                         "Default: --video_view as a numeric index.")
    ap.add_argument("--view_json", type=str, default=None,
                    help="Path to a view JSON under views/ (views/teaser/<scene>.json "
                         "or views/gallery/<scene>_viewN.json). Its 'image' field names the "
                         "dataset frame used as the video camera; equivalent to "
                         "passing that filename as --log_view. The JSON 'scene' "
                         "must match --scene. Overrides --log_view when set.")

    # Output
    ap.add_argument("--out_dir", type=str, default="results/fuzzy")
    ap.add_argument("--eps", type=float, default=1e-6)

    args = ap.parse_args()

    # A view JSON names the dataset frame to use as the video camera.
    if args.view_json is not None:
        with open(args.view_json, "r", encoding="utf-8") as _f:
            _vj = json.load(_f)
        if _vj.get("scene") != args.scene:
            raise ValueError(
                f"--view_json scene {_vj.get('scene')!r} ({args.view_json}) "
                f"does not match --scene {args.scene!r}")
        args.log_view = _vj["image"]
        print(f"[view_json] {args.view_json} -> video view {args.log_view!r}")

    # Templated defaults from --scene
    scene_root_templated = args.scene_root is None
    if scene_root_templated:
        args.scene_root = os.path.join("datasets", "fuzzy_dataset", "capture", args.scene)
    if args.init_samples is None:
        args.init_samples = os.path.join(
            "datasets", "fuzzy_dataset", "coarse", "neus2_fuzzy_dtu_15000steps_images_4", args.scene,
            "band_samples", "band0.100_n1000000.ply",
        )

    # Only templated paths are fetched; an explicit --scene_root or
    # --init_samples is used as given.
    if scene_root_templated:
        fetch_data.ensure_capture(args.scene, enabled=args.fetch)
    fetch_data.ensure_seed_ply(args.init_samples, enabled=args.fetch)

    num_sh_coeffs = sum(SH_BAND_SIZES[:args.sh_degree + 1])
    num_sh_dc   = 1
    num_sh_rest = num_sh_coeffs - num_sh_dc
    sh_upgrade_every  = int(args.sh_upgrade_every)
    active_sh_degree  = 0 if sh_upgrade_every > 0 else args.sh_degree

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device, args.gpu_id)
    print(f"[torch] device={device}  cuda_available={torch.cuda.is_available()}  gpu_id={args.gpu_id}")

    # Validate the band-sampled PLY exists BEFORE creating the output dir,
    # so a missing PLY doesn't leave behind an empty results/...<stamp>/.
    if not os.path.isfile(args.init_samples):
        # Parse band width and point count out of the missing filename so the
        # suggested command reproduces exactly that file.
        _bm = re.fullmatch(
            r"band(\d+(?:\.\d+)?)_n(\d+)\.ply",
            os.path.basename(args.init_samples),
        )
        _band, _npts = (_bm.group(1), _bm.group(2)) if _bm else ("0.1", "1000000")
        _mesh = os.path.join(
            "datasets", "fuzzy_dataset", "coarse", "neus2_fuzzy_dtu_15000steps_images_4",
            args.scene, "mesh", "15000_masked.obj",
        )
        raise FileNotFoundError(
            f"Band-sampled PLY not found: {args.init_samples}\n"
            f"Fetch the shipped one with:\n"
            f"  python scripts/fetch_data.py train\n"
            f"Or generate it with the dataset band sampler:\n"
            f"  python datasets/fuzzy_dataset/coarse/band_sampling.py \\\n"
            f"      --mesh {_mesh} \\\n"
            f"      --out  {args.init_samples} \\\n"
            f"      --band {_band} --n_points {_npts}"
        )

    tag     = f"fuzzy_{args.scene}"
    out_dir = os.path.join(args.out_dir, args.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Save options.
    opts_path = os.path.join(out_dir, "options.json")
    with open(opts_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Options saved: {opts_path}")

    out_mp4      = os.path.join(out_dir, f"{tag}.mp4")
    out_loss_txt = os.path.join(out_dir, "loss.txt")
    out_ckpt     = os.path.join(out_dir, f"{tag}_{int(args.iters):05d}.npz")

    iters         = int(args.iters)
    remesh_start  = int(args.remesh_start)
    remesh_every  = int(args.remesh_every)
    remesh_end    = int(args.remesh_end)
    prune_thresh  = float(args.prune_thresh)
    target_verts  = int(args.target_num_verts)
    eps           = float(args.eps)
    bg_color      = (0.0, 0.0, 0.0)

    # ----------------------------------------------------------------
    # Load Fuzzy data (per-frame intrinsics, RGBA composited over black).
    # Optimize on *_train.json; the final eval renders both splits and reports
    # train, test and combined metrics.
    # ----------------------------------------------------------------
    train_images_cpu, train_mvps_cpu, train_eyes_cpu, width, height, train_names = load_nerf_per_frame(
        args.scene_root, args.transforms_train,
        float(args.z_near), float(args.z_far),
        bg_color=bg_color,
    )
    num_train = int(train_images_cpu.shape[0])
    print(f"[data] train: {num_train} views @ {width}x{height} from {args.transforms_train}")
    print(f"[data] bg_color={bg_color}")

    test_images_cpu, test_mvps_cpu, test_eyes_cpu, test_W, test_H, test_names = load_nerf_per_frame(
        args.scene_root, args.transforms_test,
        float(args.z_near), float(args.z_far),
        bg_color=bg_color,
    )
    num_test = int(test_images_cpu.shape[0])
    print(f"[data] test:  {num_test} views @ {test_W}x{test_H} from {args.transforms_test}")

    # Active optimization pool. With --useall, train+test views are concatenated
    # (test still kept separately for the final PSNR/SSIM eval below).
    if args.useall:
        if (test_W, test_H) != (width, height):
            raise ValueError(
                f"--useall requires train/test images to share resolution; "
                f"got train {width}x{height} vs test {test_W}x{test_H}."
            )
        images_cpu = torch.cat([train_images_cpu, test_images_cpu], dim=0).contiguous()
        mvps_cpu   = list(train_mvps_cpu) + list(test_mvps_cpu)
        eyes_cpu   = list(train_eyes_cpu) + list(test_eyes_cpu)
        pool_names = list(train_names)    + list(test_names)
        print(f"[data] --useall: optimizing over {len(pool_names)} views "
              f"({num_train} train + {num_test} test). Final eval still uses test only.")
    else:
        images_cpu = train_images_cpu
        mvps_cpu   = train_mvps_cpu
        eyes_cpu   = train_eyes_cpu
        pool_names = train_names
    num_views = int(images_cpu.shape[0])

    # Resolve --log_view (basename) to an index in the active pool, with
    # extension-tolerant matching. Falls back to numeric --video_view.
    def _strip_ext(s: str) -> str:
        return os.path.splitext(s)[0]
    if args.log_view is not None:
        target = args.log_view
        target_stripped = _strip_ext(target)
        match = -1
        for i, n in enumerate(pool_names):
            if n == target or _strip_ext(n) == target_stripped:
                match = i
                break
        if match < 0:
            preview = ", ".join(pool_names[:8]) + (" ..." if len(pool_names) > 8 else "")
            raise ValueError(
                f"--log_view {target!r} not found in active pool of "
                f"{len(pool_names)} views. First few: {preview}"
            )
        video_view = match
        print(f"[log_view] {args.log_view!r} -> pool index {video_view} "
              f"(basename {pool_names[video_view]!r})")
    else:
        video_view = max(0, min(int(args.video_view), num_views - 1))

    if args.spatial_lr_scale:
        slr_scale = spatial_lr_scale(eyes_cpu)
        lr_verts_init  = float(args.lr_verts)       * slr_scale
        lr_verts_final = float(args.lr_verts_final) * slr_scale
        print(f"[camera] spatial_lr_scale={slr_scale:.4f}  "
              f"lr_verts={lr_verts_init:.6f} (base {args.lr_verts} x {slr_scale:.2f})")
    else:
        lr_verts_init  = float(args.lr_verts)
        lr_verts_final = float(args.lr_verts_final)

    # ----------------------------------------------------------------
    # Build line primitives from the band-sampled PLY
    # ----------------------------------------------------------------
    line_verts_cpu, line_edges_cpu, init_line_length = make_line_primitives_from_ply(
        int(args.num_lines), args.init_samples,
    )
    length_threshold = float(args.length_threshold) if args.length_threshold >= 0 else init_line_length / 2.0
    print(f"[init] Lines from PLY: {args.init_samples}")
    print(f"[init] init_line_length={init_line_length:.6f} (mean KNN)  length_threshold={length_threshold:.6f}")
    print(f"[init] {line_edges_cpu.shape[0]} lines ({line_verts_cpu.shape[0]} verts)")
    print(f"[remesh] target={target_verts} verts, every {remesh_every} iters, {remesh_start}-{remesh_end}")
    print(f"[remesh] prune_thresh ramp: {prune_thresh:.3f} -> {float(args.prune_thresh_final):.3f} "
          f"over iters {int(args.prune_ramp_start)}-{remesh_end} (linear)")

    # ----------------------------------------------------------------
    # Move to device
    # ----------------------------------------------------------------
    images    = images_cpu.to(device=device).contiguous()
    mvps      = [m.to(device=device).contiguous() for m in mvps_cpu]
    eyes      = [e.to(device=device).contiguous() for e in eyes_cpu]
    test_images = test_images_cpu.to(device=device).contiguous()
    test_mvps   = [m.to(device=device).contiguous() for m in test_mvps_cpu]
    test_eyes   = [e.to(device=device).contiguous() for e in test_eyes_cpu]
    lines_cur = line_edges_cpu.to(device=device).contiguous()

    N   = int(line_verts_cpu.shape[0])
    M_l = int(lines_cur.shape[0])


    # ----------------------------------------------------------------
    # Trainable parameters
    # ----------------------------------------------------------------
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

    _init_opacity_logit = float(torch.tensor(0.1).logit())  # ~= -2.197
    line_opacity_logit  = torch.nn.Parameter(
        torch.full((M_l,), _init_opacity_logit, dtype=torch.float32, device=device).contiguous().requires_grad_(True))

    # ----------------------------------------------------------------
    # Optimizers (with state-transfer helper for remesh)
    # ----------------------------------------------------------------
    def rebuild_opt():
        opt_verts = fuzzydr.optimize.VectorAdam([verts], lr=lr_verts_init)
        sh_groups = [{"params": [sh_dc], "lr": float(args.lr_sh_dc)}]
        if sh_rest is not None:
            sh_groups.append({"params": [sh_rest], "lr": float(args.lr_sh_rest)})
        other_groups = sh_groups + [
            {"params": [line_opacity_logit], "lr": float(args.lr_lop)},
        ]
        opt_other = torch.optim.Adam(other_groups, foreach=False)
        return opt_verts, opt_other

    def replace_tensor_to_optimizer(opt, old_param, new_param, vert_origin=None):
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
                opt.state[new_param] = {'step': old_state['step'], 'g1': g1, 'g2': g2}
            else:
                ea = _gather(old_state['exp_avg'])    if _gather else torch.zeros_like(new_param)
                es = _gather(old_state['exp_avg_sq']) if _gather else torch.zeros_like(new_param)
                opt.state[new_param] = {'step': old_state['step'], 'exp_avg': ea, 'exp_avg_sq': es}
            return

    needs_topology_update = True

    # Video schedule (log/linear-uniform across iters).
    num_frames = int(args.num_frames)
    frame_to_iter = [0]
    for k in range(1, num_frames - 1):
        it = max(1, min(iters - 1, int(round(k / (num_frames - 1) * iters))))
        frame_to_iter.append(it)
    frame_to_iter.append(iters)
    target_iters = set(frame_to_iter)

    fuzzydr.init()
    try:
        opt_verts, opt_other = rebuild_opt()

        decay_gamma = (lr_verts_final / lr_verts_init) ** (1.0 / max(iters, 1))
        sched_verts = torch.optim.lr_scheduler.ExponentialLR(opt_verts, gamma=decay_gamma)
        print(f"[lr] exp ramp: {lr_verts_init:.6g} -> {lr_verts_final:.6g}  (gamma={decay_gamma:.8f})")

        writer = (imageio.get_writer(out_mp4, fps=int(args.fps), codec="libx264", quality=8)
                  if args.video else None)
        losses: list[float] = []

        def render_view(view_idx: int, *, seed: int):
            nonlocal needs_topology_update
            mvp = mvps[view_idx]
            eye = eyes[view_idx]

            # Lines render as 1-px Bresenham segments, which ignore the
            # per-vertex radius slot entirely.  The value below is a dummy
            # placeholder only, NOT a trained or meaningful width.
            rad_slot = torch.ones(int(verts.shape[0]), dtype=torch.float32, device=device)
            va = fuzzydr.eval_sh_attrs(verts, get_sh(), rad_slot, campos=eye)

            line_opacity = torch.sigmoid(line_opacity_logit)

            img = fuzzydr.msaa_downsample_rgba(fuzzydr.rasterize(
                va,
                viewproj=mvp, campos=eye,
                lines=lines_cur if needs_topology_update else lines_cur.shape[0],
                line_opacity=line_opacity,
                width=width * 2, height=height * 2,
                tau=-1.0, seed=int(seed),
                white_bg=False,                         # composite over BLACK
                bresen_lines=True,
            ))
            needs_topology_update = False
            return va, line_opacity, img.contiguous()

        def write_frame(cur: torch.Tensor, gt: torch.Tensor):
            writer.append_data(torch.cat([to_u8(cur), to_u8(gt)], dim=1).detach().cpu().numpy())

        # Frame 0 is the bare initialization, rendered before the optimization
        # loop takes a single step. frame_to_iter[0] == 0 reserves this slot;
        # the loop below only writes frames for it >= 1, so this is always the
        # first frame of the video.
        if writer is not None:
            with torch.no_grad():
                _, _, img0 = render_view(video_view, seed=0)
            write_frame(img0, images[video_view])

        pbar       = tqdm(range(1, iters + 1), desc=f"Optimizing (fuzzy/{args.scene} lines)", ncols=130)
        view_order = torch.randperm(num_views).tolist()
        for it in pbar:
            if sh_upgrade_every > 0 and it % sh_upgrade_every == 0:
                if active_sh_degree < args.sh_degree:
                    active_sh_degree += 1
                    pbar.write(f"[SH] active degree -> {active_sh_degree} "
                               f"({sum(SH_BAND_SIZES[:active_sh_degree + 1])} coeffs) at iter {it}")

            if (it - 1) % num_views == 0 and it != 1:
                view_order = torch.randperm(num_views).tolist()
            view_idx = view_order[(it - 1) % num_views]

            opt_verts.zero_grad(set_to_none=True)
            opt_other.zero_grad(set_to_none=True)

            va, line_opacity, img = render_view(view_idx, seed=it)
            gt   = images[view_idx]
            diff = torch.abs(img - gt)

            l1_loss = diff.mean()
            if use_ssim:
                ssim_loss = 0.5 * (1.0 - ssim(
                    img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0))
                loss = 0.8 * l1_loss + 0.2 * ssim_loss
            else:
                loss = l1_loss

            error = diff.mean(dim=-1).detach().contiguous()
            aux   = fuzzydr.opacity_mask_aux_loss(
                line_opacity=line_opacity,
                error=fuzzydr.upsample2x2_scalar(error),
                eps=eps,
            )
            dirichlet_reg = torch.tensor(0.0, device=device)
            if args.lambda_dirichlet > 0 and lines_cur.shape[0] > 0:
                edge_vecs = verts[lines_cur[:, 0].long()] - verts[lines_cur[:, 1].long()]
                dirichlet_reg = float(args.lambda_dirichlet) * edge_vecs.pow(2).sum(dim=-1).mean()

            (loss + aux + dirichlet_reg).backward()

            del l1_loss
            if use_ssim:
                del ssim_loss

            opt_verts.step()
            opt_other.step()
            sched_verts.step()

            # ----------------------------------------------------------
            # Periodic relining
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

                    Ml0 = int(lines_cur.shape[0])
                    N0  = int(verts.shape[0])

                    old_verts          = verts
                    old_sh_dc          = sh_dc
                    old_sh_rest        = sh_rest
                    old_line_opacity   = line_opacity_logit

                    cur_target_v = min(target_verts, int(1.05 * N0))

                    if sh_rest is not None:
                        _sh_all = torch.cat([sh_dc, sh_rest], dim=-1)
                    else:
                        _sh_all = sh_dc
                    sh_flat = _sh_all.detach().reshape(N0, -1).contiguous()
                    carried = sh_flat

                    (out_pos, out_carried,
                     out_lin, out_lop,
                     out_vert_origin,
                     ) = discrete_updates.relining(
                         verts.detach(), carried,
                         lines_cur.detach(), torch.sigmoid(line_opacity_logit.detach()),
                         opacity_threshold=cur_prune,
                         length_threshold=length_threshold,
                         snap_endpoints=args.snap_endpoints,
                         target_num_verts=cur_target_v,
                     )

                    N1  = int(out_pos.shape[0])
                    Ml1 = int(out_lin.shape[0])
                    old_idx_t = out_vert_origin.to(torch.int64)

                    lines_cur = out_lin
                    verts = torch.nn.Parameter(out_pos.contiguous().requires_grad_(True))

                    _sh_cols = num_sh_coeffs * 3
                    _sh_full = out_carried[:, :_sh_cols].reshape(N1, 3, num_sh_coeffs)
                    sh_dc = torch.nn.Parameter(
                        _sh_full[:, :, :num_sh_dc].contiguous().requires_grad_(True))
                    if num_sh_rest > 0:
                        sh_rest = torch.nn.Parameter(
                            _sh_full[:, :, num_sh_dc:].contiguous().requires_grad_(True))

                    lop = out_lop.clamp(1e-6, 1.0 - 1e-6)
                    line_opacity_logit = torch.nn.Parameter(
                        torch.log(lop / (1.0 - lop)).requires_grad_(True))

                    replace_tensor_to_optimizer(opt_verts, old_verts, verts, old_idx_t)
                    replace_tensor_to_optimizer(opt_other, old_sh_dc, sh_dc, old_idx_t)
                    if sh_rest is not None:
                        replace_tensor_to_optimizer(opt_other, old_sh_rest, sh_rest, old_idx_t)
                    replace_tensor_to_optimizer(opt_other, old_line_opacity, line_opacity_logit)

                    n_survived = int((old_idx_t >= 0).sum())
                    needs_topology_update = True

                    pbar.write(
                        f"[remesh] it={it}  lines: {Ml0}->{Ml1}  "
                        f"verts: {N0}->{N1} ({n_survived} kept)  "
                        f"target_verts={cur_target_v}  "
                        f"prune_thresh={cur_prune:.4f}"
                    )

            loss_val = float(loss.detach().cpu())
            del loss
            losses.append(loss_val)
            pbar.set_postfix(
                loss=f"{loss_val:.3e}",
                view=f"{view_idx:03d}",
                L=int(lines_cur.shape[0]),
            )

            if writer is not None and it in target_iters:
                with torch.no_grad():
                    _, _, cur = render_view(video_view, seed=it)
                write_frame(cur, images[video_view])

        if writer is not None:
            writer.close()
            print(f"Wrote {out_mp4}")

        with open(out_loss_txt, "w", encoding="utf-8") as f:
            for i, l in enumerate(losses, start=1):
                f.write(f"{i}\t{l}\n")
        print(f"Wrote {out_loss_txt}")

        # ----------------------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------------------
        N_final  = int(verts.shape[0])
        sh_final = (torch.cat([sh_dc, sh_rest], dim=-1) if sh_rest is not None else sh_dc).detach().cpu()
        if num_sh_coeffs < SH_NUM_COEFFS:
            sh_final = torch.cat(
                [sh_final, torch.zeros(N_final, 3, SH_NUM_COEFFS - num_sh_coeffs)], dim=-1)
        sh_interleaved = sh_final.permute(0, 2, 1).reshape(N_final, SH_NUM_COEFFS * 3)

        save_checkpoint(
            out_ckpt,
            verts=verts.detach().cpu(),
            colors=sh_interleaved,
            lines=lines_cur.detach().cpu(),
            line_opacity_logit=line_opacity_logit.detach().cpu(),
            radius=torch.full((N_final,), LINE_RADIUS, dtype=torch.float32),
            color_mode="sh",
            sh_degree=args.sh_degree,
            extra_meta={
                "scene":   args.scene,
                "split":   "train+test" if args.useall else "train",
                "dataset": "fuzzy",
                "iters":   iters,
                "transforms_train":   args.transforms_train,
                "transforms_test":    args.transforms_test,
                "bg_color":           list(bg_color),
                "bresen_lines":       True,
            },
        )
        print(f"Wrote {out_ckpt}")

        # ----------------------------------------------------------------
        # Eval: render every train and test view
        # ----------------------------------------------------------------
        eval_dir   = os.path.join(out_dir, f"eval_{iters}")
        gt_dir     = os.path.join(eval_dir, "gt_masked")
        render_dir = os.path.join(eval_dir, "renders")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)

        eval_splits = (
            ("train", train_names, train_mvps_cpu, train_eyes_cpu, train_images_cpu, width,  height),
            ("test",  test_names,  test_mvps_cpu,  test_eyes_cpu,  test_images_cpu,  test_W, test_H),
        )
        split_of: dict[str, str]   = {}
        psnr_of:  dict[str, float] = {}
        ssim_of:  dict[str, float] = {}
        lpips_of: dict[str, float] = {}

        t_eval0 = time.time()
        with torch.no_grad():
            line_opacity = torch.sigmoid(line_opacity_logit)
            rad_slot = torch.ones(int(verts.shape[0]), dtype=torch.float32, device=device)

            for split, names_s, mvps_s, eyes_s, images_s, W_s, H_s in eval_splits:
                num_s = len(names_s)
                for vi in range(num_s):
                    mvp_i = mvps_s[vi].to(device=device).contiguous()
                    eye_i = eyes_s[vi].to(device=device).contiguous()
                    va = fuzzydr.eval_sh_attrs(verts, get_sh(), rad_slot, campos=eye_i)

                    pred = fuzzydr.msaa_downsample_rgba(fuzzydr.rasterize(
                        va, viewproj=mvp_i, campos=eye_i,
                        lines=lines_cur, line_opacity=line_opacity,
                        width=W_s * 2, height=H_s * 2,
                        tau=0.5, seed=0, white_bg=False,
                        bresen_lines=True,
                    )).clamp(0, 1).contiguous()
                    gt = images_s[vi].to(device=device)

                    # Rendered and masked ground-truth images keep their dataset filenames.
                    fname = names_s[vi]
                    if not os.path.splitext(fname)[1]:
                        fname += ".png"
                    imageio.imwrite(os.path.join(render_dir, fname), to_u8(pred).cpu().numpy())
                    imageio.imwrite(os.path.join(gt_dir, fname), to_u8(gt).cpu().numpy())

                    pred_4d = pred.permute(2, 0, 1)[None]
                    gt_4d   = gt.permute(2, 0, 1)[None]
                    mse     = (pred - gt).pow(2).mean()

                    split_of[fname] = split
                    psnr_of[fname]  = float(-10.0 * torch.log10(mse.clamp_min(1e-10)))
                    if use_ssim:
                        ssim_of[fname] = float(ssim(gt_4d, pred_4d, data_range=1.0))
                    lpips_of[fname] = float(lpips_vgg(pred_4d, gt_4d))

                    if (vi + 1) % 20 == 0 or vi == num_s - 1:
                        print(f"[EVAL] {split}: rendered {vi + 1}/{num_s} views")

        def _mean_over(values: dict[str, float], keys: list[str]) -> float:
            return float(np.mean([values[k] for k in keys]))

        groups = {
            "train": [f for f, sp in split_of.items() if sp == "train"],
            "test":  [f for f, sp in split_of.items() if sp == "test"],
            "all":   list(split_of),
        }

        metrics: dict[str, dict[str, float]] = {}
        for group, keys in groups.items():
            entry = {"PSNR": _mean_over(psnr_of, keys)}
            if ssim_of:
                entry["SSIM"] = _mean_over(ssim_of, keys)
            entry["LPIPS"] = _mean_over(lpips_of, keys)
            entry["N"] = len(keys)
            metrics[group] = entry

        print(f"[EVAL] ({time.time() - t_eval0:.1f}s)")
        for group in ("train", "test", "all"):
            m = metrics[group]
            ssim_str = f"  SSIM={m['SSIM']:.4f}" if "SSIM" in m else ""
            print(f"[EVAL] {group:5s} (N={m['N']:3d})  PSNR={m['PSNR']:.3f} dB"
                  f"{ssim_str}  LPIPS={m['LPIPS']:.4f}")

        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        per_view = {"split": split_of, "PSNR": psnr_of}
        if ssim_of:
            per_view["SSIM"] = ssim_of
        per_view["LPIPS"] = lpips_of
        with open(os.path.join(eval_dir, "per_view.json"), "w") as f:
            json.dump(per_view, f, indent=2)

        print(f"[EVAL] Saved {eval_dir}/")

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
