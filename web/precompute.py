"""web/precompute.py - bake a checkpoint .npz into a compact binary
bundle for the three.js/WebGL2 web viewer.

Pipeline:

1. Load .npz, prune lines by opacity threshold, compact unused vertices.
2. Truncate SH coefficients to degree <= sh_max (default 1 -> 4 coeffs / chan).
3. Run lines_to_strips() - CSR-adjacency greedy walk that emits maximal
   polyline strips with 0xFFFFFFFF restart sentinels.
4. Serialise everything into a single little-endian .bin:

       tag       u32   0x454E494C, the four bytes "LINE"
       version   u32   1
       n_verts   u32
       n_idx     u32   (length of the line-strip index buffer)
       n_strips  u32
       sh_deg    u32   SH degree kept; the bundle stores (sh_deg + 1)^2
                       coefficients per colour channel
       opacity_thresh f32
       _pad      u32   0
       bbox      f32 x 6   (xmin, ymin, zmin, xmax, ymax, zmax)
       positions f32 x n_verts x 3
       sh        f32 x n_verts x 3 x (sh_deg + 1)^2
                 layout: per vertex, k-major then c-minor:
                   for k in 0..K:  for c in 0..3:  sh[i, k, c]
       indices   u32 x n_idx  (LINE_STRIP, primitive-restart = 0xFFFFFFFF)

In --batch mode the three degree-1 checkpoints are downloaded from the Fuzzy
dataset (1.3 GB) if the directory is empty; --no-fetch turns that off.

Usage
-----
    python web/precompute.py \\
        --ckpt datasets/fuzzy_dataset/checkpoints/fuzzy/web/fuzzy_kiwi_sh1.npz \\
        --out  web/data/fuzzy_kiwi_sh1.bin \\
        [--sh-max 1] [--line-opacity-thresh 0.5]

    # bake every checkpoint under datasets/fuzzy_dataset/checkpoints/fuzzy/web/
    python web/precompute.py
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fetch_data


# Checkpoints baked for the web viewer, at their path in the Fuzzy dataset.
CHECKPOINT_DIR = "datasets/fuzzy_dataset/checkpoints/fuzzy/web"

SH_NUM_COEFFS_FULL = 16   # coefficients per channel at degree 3
_RESTART = np.uint32(0xFFFFFFFF)
_LINE_BINARY = 0x454E494C   # the four bytes "LINE"
_VERSION = 1


# ---------------------------------------------------------------------------
# Pruning + compaction
# ---------------------------------------------------------------------------

def prune_lines_by_opacity(
    lines: np.ndarray,
    logits: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    keep = prob >= thresh
    print(f"  [prune] lines  thresh={thresh:.3f}  "
          f"{lines.shape[0]} -> {int(keep.sum())}  "
          f"({100.0 * keep.mean():.1f}% kept)")
    return lines[keep].copy(), logits[keep].copy()


def compact_vertices(
    per_vertex: list[np.ndarray],
    lines: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    N = per_vertex[0].shape[0]
    used = np.unique(lines.reshape(-1).astype(np.int64))
    remap = np.full(N, -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_lines = remap[lines.astype(np.int64)].astype(np.uint32)
    assert (new_lines.astype(np.int64) >= 0).all()
    print(f"  [compact] verts: {N} -> {len(used)}")
    return [a[used].copy() for a in per_vertex], new_lines


# ---------------------------------------------------------------------------
# Line-strip stripification (CSR adjacency + odd-degree-first greedy walk).
# ---------------------------------------------------------------------------

def lines_to_strips(lines: np.ndarray) -> tuple[np.ndarray, int]:
    """Rebake a LINE_LIST index buffer into LINE_STRIP-with-restart indices.

    Returns (flat_indices, n_strips).  Restart sentinel is 0xFFFFFFFF.
    """
    L = int(lines.shape[0])
    if L == 0:
        return np.empty(0, dtype=np.uint32), 0

    lines64 = np.ascontiguousarray(lines, dtype=np.int64).reshape(-1, 2)
    a = lines64[:, 0]
    b = lines64[:, 1]
    n_verts = int(max(a.max(), b.max())) + 1

    # Undirected adjacency as CSR (each edge contributes both directions, so
    # the walk can go either way without reordering).
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    eid = np.tile(np.arange(L, dtype=np.int64), 2)

    order = np.argsort(src, kind="stable")
    src = src[order]; dst = dst[order]; eid = eid[order]

    offsets = np.zeros(n_verts + 1, dtype=np.int64)
    np.add.at(offsets, src + 1, 1)
    np.cumsum(offsets, out=offsets)

    cursor = offsets[:-1].copy()
    upper  = offsets[1:].copy()
    used   = np.zeros(L, dtype=bool)

    # Walk-start order: odd-degree (== path endpoints / branching tips) first
    # so each simple polyline traverses as a single chain.  Even-degree
    # vertices seed loops only when no path remains.
    degree = upper - cursor
    odd  = np.flatnonzero(degree & 1)
    even = np.flatnonzero(((degree & 1) == 0) & (degree > 0))
    start_order = np.concatenate([odd, even]).tolist()

    out_chunks: list[np.ndarray] = []
    walk_buf: list[int] = []

    for start in start_order:
        while True:
            i = int(cursor[start]); up = int(upper[start])
            while i < up and used[int(eid[i])]:
                i += 1
            cursor[start] = i
            if i >= up:
                break

            walk_buf.clear()
            walk_buf.append(int(start))
            v = int(start)
            while True:
                i = int(cursor[v]); up = int(upper[v])
                while i < up and used[int(eid[i])]:
                    i += 1
                cursor[v] = i
                if i >= up:
                    break
                e = int(eid[i])
                used[e] = True
                v = int(dst[i])
                walk_buf.append(v)

            out_chunks.append(np.asarray(walk_buf, dtype=np.uint32))

    if not out_chunks:
        return np.empty(0, dtype=np.uint32), 0

    n_strips = len(out_chunks)
    total = sum(c.size for c in out_chunks) + (n_strips - 1)
    flat = np.empty(total, dtype=np.uint32)
    pos = 0
    for i, c in enumerate(out_chunks):
        if i > 0:
            flat[pos] = _RESTART
            pos += 1
        flat[pos:pos + c.size] = c
        pos += c.size
    return flat, n_strips


# ---------------------------------------------------------------------------
# SH truncation
# ---------------------------------------------------------------------------

def truncate_sh(colors: np.ndarray, sh_max: int) -> np.ndarray:
    """Take the first K = (sh_max+1)^2 coefficients from the interleaved
    [N, 16, 3] checkpoint layout and return an [N, K, 3] f32 array (k-major).
    """
    N = colors.shape[0]
    if colors.shape[1] != SH_NUM_COEFFS_FULL * 3:
        raise ValueError(
            f"colors must be [N, 48] (interleaved [N, 16, 3]); got {colors.shape}"
        )
    K = (sh_max + 1) ** 2
    if K > SH_NUM_COEFFS_FULL:
        raise ValueError(f"sh_max={sh_max} requires {K} coeffs, ckpt only has 16")
    # Checkpoint is interleaved [N, 16, 3]; the first K rows are the
    # lowest-order coefficients in basis-index order.  Emitted as [N, K, 3].
    return colors.reshape(N, SH_NUM_COEFFS_FULL, 3)[:, :K, :].astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Binary writer
# ---------------------------------------------------------------------------

def write_bundle(
    out_path: Path,
    positions: np.ndarray,
    sh: np.ndarray,
    indices: np.ndarray,
    n_strips: int,
    sh_max: int,
    opacity_thresh: float,
) -> None:
    assert positions.dtype == np.float32 and positions.ndim == 2 and positions.shape[1] == 3
    assert sh.dtype == np.float32 and sh.ndim == 3 and sh.shape[0] == positions.shape[0]
    assert sh.shape[1] == (sh_max + 1) ** 2 and sh.shape[2] == 3
    assert indices.dtype == np.uint32 and indices.ndim == 1

    N = positions.shape[0]
    bbox_min = positions.min(axis=0).astype(np.float32)
    bbox_max = positions.max(axis=0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        # Header (64 bytes, all little-endian).
        f.write(struct.pack(
            "<IIIIIIfI",
            _LINE_BINARY, _VERSION,
            N, int(indices.size), int(n_strips),
            int(sh_max), float(opacity_thresh), 0,
        ))
        f.write(bbox_min.tobytes()); f.write(bbox_max.tobytes())
        # Buffers.
        f.write(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(sh,        dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(indices,   dtype=np.uint32).tobytes())

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  [write] {out_path}  "
          f"{size_mb:.1f} MB  "
          f"(verts={N}  idx={indices.size}  strips={n_strips})")


# ---------------------------------------------------------------------------
# Per-ckpt driver
# ---------------------------------------------------------------------------

def process_one(
    ckpt_path: Path,
    out_path: Path,
    sh_max: int,
    line_opacity_thresh: float,
) -> None:
    print(f"[ckpt] {ckpt_path}")
    z = np.load(ckpt_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    print(f"  [meta] scene={meta.get('scene')}  sh_degree={meta.get('sh_degree')}  "
          f"color_channels={meta.get('color_channels')}")

    verts  = z["verts"].astype(np.float32)
    colors = z["colors"].astype(np.float32)
    lines  = z["lines"].astype(np.uint32)
    line_opacity_logit = z["line_opacity_logit"].astype(np.float32)

    if lines.shape[0] == 0:
        raise RuntimeError(f"{ckpt_path}: checkpoint has no lines.")
    if colors.shape[1] != SH_NUM_COEFFS_FULL * 3:
        raise RuntimeError(
            f"{ckpt_path}: expected SH ckpt with 48 color channels; got {colors.shape[1]}"
        )

    lines, _ = prune_lines_by_opacity(lines, line_opacity_logit, line_opacity_thresh)
    if lines.shape[0] == 0:
        raise RuntimeError("All lines pruned - try a lower opacity threshold.")

    [verts, colors], lines = compact_vertices([verts, colors], lines)

    sh = truncate_sh(colors, sh_max)
    indices, n_strips = lines_to_strips(lines)
    print(f"  [strips] {lines.shape[0]} segments -> {n_strips} strips  "
          f"(idx buffer length {indices.size}, "
          f"avg strip length {(indices.size - (n_strips - 1)) / max(n_strips, 1):.1f} verts)")

    write_bundle(
        out_path,
        positions=verts,
        sh=sh,
        indices=indices,
        n_strips=n_strips,
        sh_max=sh_max,
        opacity_thresh=line_opacity_thresh,
    )


def discover_batch(root: Path) -> list[tuple[Path, str]]:
    """Find every .npz directly under `root`; the file stem is the scene id.

    Checkpoints are named ``fuzzy_<scene>_<notes>.npz`` and the bundle beside
    each one takes the same stem, so a scene is identified by its filename
    rather than by an enclosing per-run directory.
    """
    results: list[tuple[Path, str]] = []
    for npz in sorted(root.glob("*.npz")):
        results.append((npz, npz.stem))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ckpt", type=str, help="single checkpoint .npz")
    src.add_argument("--batch", type=str, default=CHECKPOINT_DIR,
                     help=f"directory of checkpoint .npz files "
                          f"(default: {CHECKPOINT_DIR})")
    ap.add_argument("--out", type=str,
                    help="output .bin (with --ckpt) or scene name (with --batch is ignored)")
    ap.add_argument("--out-dir", type=str, default="web/data",
                    help="output directory for --batch mode")
    ap.add_argument("--sh-max", type=int, default=1,
                    help="max SH degree to keep (0=DC only, 1=l<=1, ...)")
    ap.add_argument("--line-opacity-thresh", type=float, default=0.5)
    ap.add_argument("--no-fetch", dest="fetch", action="store_false", default=True,
                    help="do not download missing checkpoints from the Fuzzy dataset")
    args = ap.parse_args()

    if args.sh_max < 0 or args.sh_max > 3:
        print(f"--sh-max must be in [0, 3]; got {args.sh_max}", file=sys.stderr)
        return 2

    if args.ckpt:
        if not args.out:
            print("--out is required with --ckpt", file=sys.stderr)
            return 2
        process_one(Path(args.ckpt), Path(args.out),
                    args.sh_max, args.line_opacity_thresh)
    else:
        # Only the shipped directory is fetchable; a --batch elsewhere holds
        # local training output.
        if args.batch == CHECKPOINT_DIR:
            fetch_data.ensure_checkpoints("fuzzy/web", enabled=args.fetch)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = discover_batch(Path(args.batch))
        if not targets:
            print(f"[batch] no .npz found under {args.batch}", file=sys.stderr)
            return 1
        manifest = []
        for ckpt_path, scene_id in targets:
            out_path = out_dir / f"{scene_id}.bin"
            try:
                process_one(ckpt_path, out_path,
                            args.sh_max, args.line_opacity_thresh)
                manifest.append({
                    "id": scene_id,
                    "file": out_path.name,
                    "ckpt": str(ckpt_path),
                })
            except Exception as e:
                print(f"  [skip] {ckpt_path}: {e}", file=sys.stderr)

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(
            {"sh_max": args.sh_max,
             "opacity_thresh": args.line_opacity_thresh,
             "scenes": manifest},
            indent=2,
        ))
        print(f"\n[manifest] {manifest_path}  ({len(manifest)} scenes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
