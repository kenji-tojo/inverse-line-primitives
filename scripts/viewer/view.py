#!/usr/bin/env python3
"""Launch the interactive viewer on a trained line or triangle checkpoint.

Any line or triangle checkpoint written by the training scripts works; point
checkpoints have no viewer backend and are rejected. A released one named
under ``datasets/fuzzy_dataset/checkpoints/`` is downloaded from the Fuzzy dataset if
it is not on disk yet, about 430 MB per scene; ``--no_fetch`` turns that off.

Usage
-----
    source ~/VulkanSDK/<version>/setup-env.sh
    python scripts/viewer/view.py [--ckpt path/to/ckpt.npz]

    python scripts/viewer/view.py \\
        --ckpt datasets/fuzzy_dataset/checkpoints/shelly/shelly_khady.npz

Runs on any Vulkan-capable machine -- no CUDA and no PyTorch. Drag with the
left mouse button to orbit, right/middle to pan, scroll to zoom. AA mode,
line style, line width and background are all adjustable from the ImGui
panel, so they are not exposed as flags; the window size is, since dragging
cannot give an exact one.

Everything else is read from the checkpoint rather than supplied, so this
script carries no assumption about where training wrote its output or how it
named the file:

  * SH is stored interleaved as ``colors[N, 48]`` = ``[N, 16, 3]`` and is
    transposed to the viewer's ``[N, 3, 16]`` layout.
  * The primitive class is read from the checkpoint's ``primitive`` field,
    falling back on the array shapes for checkpoints saved before that field
    existed, which are all line-only. Triangles are drawn by the same backend
    with the same GPU SH evaluation; only the draw topology differs.
  * Primitives are pruned by opacity: one is kept iff
    ``sigmoid(opacity_logit) >= --line_thresh`` / ``--face_thresh``
    (default 0.5 either way).
    Low-opacity primitives that survive into the draw call leak soft pixels.
    This has no runtime equivalent in the viewer, so it stays a flag.
  * Per-vertex radius is a line-only concern -- it selects the quad line
    style -- and is passed only when the checkpoint says the lines are
    not Bresenham (``bresen_lines: false``), mirroring
    ``scripts/baselines/eval.py``. Checkpoints without the flag predate it and
    are Bresenham.
  * The background comes from ``bg_color`` when the checkpoint records it
    (``train_fuzzy.py`` trains against black); absent means white.
  * The panel's screenshot button writes ``screenshot_NNNN.png`` into
    ``--screenshot_dir``, the repository root by default.
  * The camera convention follows the checkpoint's ``dataset`` field: fuzzy
    scenes load with the up reference flipped, shelly scenes do not.  Pass
    ``--convention`` to override.  This affects only the camera, never the
    geometry, and the viewer panel's "Flip up" toggles it at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import fuzzydr_viewer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import fetch_data


# Eight colors for --color_by_polyline, picked per polyline under a fixed
# seed so a checkpoint always colors the same way.
_POLYLINE_PALETTE = np.array([
    [0.89, 0.29, 0.29], [0.24, 0.51, 0.85], [0.35, 0.72, 0.36],
    [0.95, 0.65, 0.18], [0.62, 0.39, 0.78], [0.20, 0.72, 0.72],
    [0.91, 0.48, 0.70], [0.55, 0.44, 0.28],
], dtype=np.float32)


def polyline_colors(lines: np.ndarray, num_verts: int, seed: int = 0) -> np.ndarray:
    """One palette color per connected polyline, returned as [num_verts, 3].

    No vertex in these checkpoints has degree above two, so a connected
    component of the segment graph is exactly one polyline.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e0 = lines[:, 0].astype(np.int64)
    e1 = lines[:, 1].astype(np.int64)
    adj = coo_matrix((np.ones(e0.size, np.uint8), (e0, e1)),
                     shape=(num_verts, num_verts))
    n_comp, labels = connected_components(adj, directed=False)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, _POLYLINE_PALETTE.shape[0], n_comp)
    return _POLYLINE_PALETTE[pick[labels]], n_comp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path,
                    default="datasets/fuzzy_dataset/checkpoints/fuzzy/full/fuzzy_kiwi.npz",
                    help="path to a checkpoint .npz")
    ap.add_argument("--color_by_polyline", action="store_true",
                    help="color each connected polyline instead of evaluating SH")
    ap.add_argument("--line_thresh", type=float, default=0.5,
                    help="keep lines with sigmoid(line_opacity_logit) >= this "
                         "(default: 0.5)")
    ap.add_argument("--face_thresh", type=float, default=0.5,
                    help="keep triangles with sigmoid(face_opacity_logit) >= "
                         "this (default: 0.5)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--convention", choices=["shelly", "fuzzy"], default=None,
                    help="camera orbit convention; default follows the "
                         "checkpoint's dataset field")
    ap.add_argument("--screenshot_dir", type=Path, default=_REPO_ROOT,
                    help=f"where the panel's screenshot button writes "
                         f"(default: {_REPO_ROOT})")
    ap.add_argument("--no_fetch", dest="fetch", action="store_false", default=True,
                    help="Do not download a missing checkpoint from the Fuzzy "
                         "dataset; fail instead.")
    args = ap.parse_args()

    # Only a path under datasets/fuzzy_dataset/ is a released model to fetch;
    # anything else is local training output.
    fetch_data.ensure_file(args.ckpt, enabled=args.fetch,
                           what=f"checkpoint {args.ckpt.name}")
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    print(f"[load] {args.ckpt}")
    ck = np.load(args.ckpt, allow_pickle=True)
    meta = json.loads(str(ck["meta_json"]))
    if meta.get("color_mode") != "sh" or meta.get("color_channels") != 48:
        raise SystemExit(f"expected a degree-3 SH checkpoint; got {meta.get('color_mode')!r} "
                         f"with {meta.get('color_channels')} colour channels")

    verts = ck["verts"].astype(np.float32)              # [N, 3]
    N = verts.shape[0]

    # Primitive class, detected the way scripts/baselines/eval.py detects it:
    # from the "primitive" metadata field when the checkpoint records one, and
    # from the array shapes otherwise -- checkpoints predating that field are
    # all line-only.  Points have no backend here, so they stop at the door.
    n_faces = int(ck["faces"].shape[0]) if "faces" in ck.files else 0
    n_lines = int(ck["lines"].shape[0]) if "lines" in ck.files else 0
    if (meta.get("primitive") == "points" or "n_points" in meta
            or (n_faces == 0 and n_lines == 0)):
        raise SystemExit("point checkpoints are not supported by this viewer; "
                         "it draws line and triangle primitives only")
    triangles = n_faces > 0 and n_lines == 0

    # sigmoid(x) >= t  <=>  x >= log(t / (1 - t)); avoids a 1.7M-element sigmoid.
    if triangles:
        faces = ck["faces"].astype(np.uint32).reshape(-1, 3)
        lines = None
        fop = ck["face_opacity_logit"].astype(np.float32)
        if fop.size == faces.shape[0]:
            t = float(np.clip(args.face_thresh, 1e-6, 1.0 - 1e-6))
            keep = fop >= float(np.log(t / (1.0 - t)))
            print(f"[prune] thresh={args.face_thresh}  "
                  f"{faces.shape[0]} -> {int(keep.sum())} triangles "
                  f"({100.0 * keep.mean():.1f}% kept)")
            faces = faces[keep]
        if faces.shape[0] == 0:
            raise SystemExit("every triangle was pruned; lower --face_thresh")
    else:
        faces = None
        lines = ck["lines"].astype(np.uint32).reshape(-1, 2)
        lop = ck["line_opacity_logit"].astype(np.float32)
        if lop.size == lines.shape[0]:
            t = float(np.clip(args.line_thresh, 1e-6, 1.0 - 1e-6))
            keep = lop >= float(np.log(t / (1.0 - t)))
            print(f"[prune] thresh={args.line_thresh}  "
                  f"{lines.shape[0]} -> {int(keep.sum())} lines ({100.0 * keep.mean():.1f}% kept)")
            lines = lines[keep]
        if lines.shape[0] == 0:
            raise SystemExit("every line was pruned; lower --line_thresh")

    if args.color_by_polyline:
        if triangles:
            raise SystemExit("--color_by_polyline needs a line-only checkpoint; "
                             f"this one carries {n_faces} faces")
        vert_colors, n_polylines = polyline_colors(lines, N)
        sh_coeffs = None
        print(f"[color] {n_polylines} polylines over "
              f"{_POLYLINE_PALETTE.shape[0]} colors")
    else:
        vert_colors = None
        # colors is [N, 16, 3] flattened; the viewer wants [N, 3, 16].
        sh_coeffs = (ck["colors"].astype(np.float32)
                     .reshape(N, 16, 3).transpose(0, 2, 1).copy())

    # Radius drives the quad line style, so it is line-only; triangle
    # checkpoints save none.  Checkpoints without the flag predate it and are
    # Bresenham.
    if triangles:
        bresen = None
        radius = None
    else:
        bresen = bool(meta.get("bresen_lines", True))
        radius = (ck["radius"].astype(np.float32)
                  if not bresen and "radius" in ck.files and ck["radius"].shape[0] else None)

    # Training renders against white or black; scripts that train on black
    # record it as "bg_color". Absent means white, as the Shelly runs use.
    background = meta.get("bg_color")

    # Fuzzy captures sit the other way up relative to the training Z-up
    # convention, so their scenes load upside down unless the camera's up
    # reference is flipped.  The checkpoint records which dataset it came from.
    convention = args.convention or ("fuzzy" if meta.get("dataset") == "fuzzy"
                                     else "shelly")
    flip_up = (convention == "fuzzy")

    scene = meta.get("scene", "?")
    if triangles:
        print(f"[view] {scene}: {N} verts, {faces.shape[0]} triangles, "
              f"{args.width}x{args.height}, "
              f"bg={background if background is not None else 'white (not recorded)'}, "
              f"convention={convention}")
    else:
        print(f"[view] {scene}: {N} verts, {lines.shape[0]} lines, "
              f"{args.width}x{args.height}, bresen={bresen}, "
              f"per-vertex radius={'yes' if radius is not None else 'no'}, "
              f"bg={background if background is not None else 'white (not recorded)'}, "
              f"convention={convention}")
    print("[view] close the window to exit.")
    print("[view] uploading geometry to the GPU ...", flush=True)
    fuzzydr_viewer.launch(verts, radius=radius, faces=faces, lines=lines,
                          colors=vert_colors, sh_coeffs=sh_coeffs,
                          # launch() rebakes the segment list into strips by
                          # default and rejects that default when there is no
                          # segment list to rebake (fuzzydr_viewer 0.1.0), so
                          # the triangle case asks for "list" instead.  With no
                          # lines the topology reaches no draw either way.
                          line_topology="list" if triangles else "strip",
                          width=args.width, height=args.height,
                          background=background, flip_up=flip_up,
                          screenshot_dir=str(args.screenshot_dir))


if __name__ == "__main__":
    main()
