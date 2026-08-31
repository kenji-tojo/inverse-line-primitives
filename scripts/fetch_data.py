#!/usr/bin/env python3
"""scripts/fetch_data.py - download the Fuzzy dataset into datasets/fuzzy_dataset/.

huggingface.co/datasets/kenji-tojo/fuzzy_dataset is 50 GB, most of it
full-resolution captures and COLMAP output that nothing here reads.

    python scripts/fetch_data.py train        # capture/ + coarse/, 4.3 GB
    python scripts/fetch_data.py checkpoints  # the released line models, 7.3 GB
    python scripts/fetch_data.py all          # the whole dataset, 50 GB

`train` takes the 1/4-resolution images of all eight scenes, masked and
unmasked, their transforms, and the whole of coarse/. Every mode also takes the
dataset README and the scripts that regenerate its derived files.

The training and viewer scripts call the ensure_* helpers below on startup and
download what they are missing, so a clean checkout needs no separate step.
Pass --no_fetch to any of them to fail on missing data instead, on an offline
node or a metered link.

A large anonymous download can hit Hugging Face's rate limit.  To avoid that,
put a read token on one line in datasets/HF_TOKEN.txt, or log in with
`hf auth login`.  Files already on disk are skipped: re-running this script
after a failed download fetches only the files still missing.

The Shelly images are a third-party dataset and are not fetched here. Download
them from research.nvidia.com/labs/toronto-ai/adaptive-shells/ and extract them
to datasets/shelly_data_release/ (see README.md).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ID = "kenji-tojo/fuzzy_dataset"

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = REPO_ROOT / "datasets" / "fuzzy_dataset"

# Optional read token, one line.  datasets/ is gitignored, so it stays local.
TOKEN_FILE = REPO_ROOT / "datasets" / "HF_TOKEN.txt"

FUZZY_SCENES = ("cactus1", "cactus2", "dinosaur", "flowers",
                "fur", "kiwi", "tawashi", "textiles")
SHELLY_SCENES = ("fernvase", "horse", "khady", "kitten", "pug", "woolly")

FUZZY_COARSE = "coarse/neus2_fuzzy_dtu_15000steps_images_4"
SHELLY_COARSE = "coarse/neus2_shelly_dtu_15000steps"

# Training reads images_4_rgba/ and the transforms beside it.  images_4/ is
# the same views unmasked, 1.5 GB.  The raw input/, the full, 1/2 and 1/8
# pyramids and the COLMAP sparse/ reconstruction are another 38 GB and come
# only with `all`.
CAPTURE_PATTERNS = ("capture/{scene}/images_4_rgba/*",
                    "capture/{scene}/images_4/*",
                    "capture/{scene}/transforms_images_4_masked_*.json")
COARSE_PATTERNS = ("coarse/**",)
CHECKPOINT_PATTERNS = ("checkpoints/**",)

# The dataset README documents the whole 50 GB, not the slice fetched here.
# It and the regeneration scripts come with every fetch; coarse/** already
# carries the ones under coarse/.
DOC_PATTERNS = ("README.md",
                "capture/*.py", "capture/*.sh", "capture/*.txt")


def capture_patterns(scene: str = "*") -> list[str]:
    """Allow-patterns for the training portion of one Fuzzy scene, or all."""
    return [p.format(scene=scene) for p in CAPTURE_PATTERNS]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _token() -> str | None:
    """The token in datasets/HF_TOKEN.txt, or None to let the hub decide.

    None falls back to huggingface_hub's own resolution: HF_TOKEN in the
    environment, then the cached login from `hf auth login`.
    """
    if not TOKEN_FILE.is_file():
        return None
    tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not tok:
        print(f"[fetch] {TOKEN_FILE} is empty; downloading without it")
        return None
    return tok


def download(patterns: list[str] | None, *, what: str) -> None:
    """Fetch `patterns` into datasets/fuzzy_dataset/; None fetches everything.

    The dataset README and its scripts come along with every fetch.  Files
    already present are left alone, so a partial download resumes.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            f"huggingface_hub is needed to download {what}.\n"
            f"  pip install huggingface_hub\n"
            f"Or download {REPO_ID} yourself and reproduce its layout "
            f"under {LOCAL_DIR}."
        ) from exc

    print(f"[fetch] {what}")
    for p in patterns or ["(everything)"]:
        print(f"[fetch]   {p}")
    token = _token()
    if token:
        print(f"[fetch]   using the token in {TOKEN_FILE}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns and patterns + list(DOC_PATTERNS),
        local_dir=str(LOCAL_DIR),
        token=token,
    )
    print(f"[fetch] into {LOCAL_DIR}")


def _repo_relative(path: str | os.PathLike) -> str | None:
    """`path` relative to datasets/fuzzy_dataset/, or None if it lies outside.

    Both sides are resolved, so a `datasets/fuzzy_dataset` symlink into storage
    elsewhere is still recognized as the dataset.
    """
    try:
        return Path(path).resolve().relative_to(LOCAL_DIR.resolve()).as_posix()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Ensure: download what a run is missing
# ---------------------------------------------------------------------------

def ensure_file(path: str | os.PathLike, *, what: str,
                enabled: bool = True) -> bool:
    """Download one dataset file if it is missing.

    Returns True when the file is present afterwards.  A path outside
    datasets/fuzzy_dataset/ is not in the dataset and is used as given.
    """
    p = Path(path)
    if p.is_file():
        return True
    rel = _repo_relative(p)
    if rel is None or not enabled:
        return False
    download([rel], what=what)
    return p.is_file()


def missing_capture_files(scene: str) -> list[Path]:
    """Files a Fuzzy training run reads from capture/<scene>/ that are absent.

    Every frame listed in the two transform files is checked, so an
    interrupted download is reported as incomplete rather than as present.
    """
    root = LOCAL_DIR / "capture" / scene
    missing: list[Path] = []
    for split in ("train", "test"):
        tf = root / f"transforms_images_4_masked_{split}.json"
        if not tf.is_file():
            missing.append(tf)
            continue
        with open(tf, "r", encoding="utf-8") as f:
            frames = json.load(f)["frames"]
        for fr in frames:
            fp = fr["file_path"].replace("\\", "/").removeprefix("./")
            img = root / fp
            if not img.is_file():
                missing.append(img)
    return missing


def ensure_capture(scene: str, *, enabled: bool = True) -> None:
    """Download one Fuzzy scene's training views if they are incomplete."""
    missing = missing_capture_files(scene)
    if not missing:
        return
    if not enabled:
        raise FileNotFoundError(
            f"{len(missing)} file(s) missing from {LOCAL_DIR / 'capture' / scene}, "
            f"starting with {missing[0]}\n"
            f"Fetch them with:\n"
            f"  python scripts/fetch_data.py train"
        )
    download(capture_patterns(scene), what=f"Fuzzy scene {scene!r} (training views)")
    still = missing_capture_files(scene)
    if still:
        raise FileNotFoundError(
            f"{len(still)} file(s) still missing after fetching scene {scene!r}, "
            f"starting with {still[0]}"
        )


def missing_band_sample_dirs() -> list[Path]:
    """Scene directories under coarse/ with no band-sampled points on disk."""
    dirs = [LOCAL_DIR / FUZZY_COARSE / s for s in FUZZY_SCENES]
    dirs += [LOCAL_DIR / SHELLY_COARSE / f"shelly_{s}" for s in SHELLY_SCENES]
    return [d for d in dirs if not any((d / "band_samples").glob("*.ply"))]


def ensure_coarse(*, enabled: bool = True) -> None:
    """Download coarse/ whole if any scene is missing its seed points."""
    missing = missing_band_sample_dirs()
    if not missing:
        return
    if not enabled:
        raise FileNotFoundError(
            f"coarse/ is incomplete: {len(missing)} scene(s) have no "
            f"band-sampled seed points, starting with {missing[0]}\n"
            f"Fetch coarse/ with:\n"
            f"  python scripts/fetch_data.py train"
        )
    download(list(COARSE_PATTERNS), what="coarse meshes and seed points (2.0 GB)")


def ensure_seed_ply(path: str | os.PathLike, *, enabled: bool = True) -> None:
    """Make sure one band-sampled seed PLY is on disk, fetching coarse/ if not.

    Raises early rather than letting the missing file surface inside the PLY
    reader, which happens only after the dataset is loaded onto the GPU.
    """
    p = Path(path)
    if p.is_file():
        return
    if _repo_relative(p) is None:
        raise FileNotFoundError(
            f"Band-sampled PLY not found: {p}\n"
            f"This path is outside {LOCAL_DIR}, so it is not in the "
            f"published dataset.  Generate it with "
            f"datasets/fuzzy_dataset/coarse/band_sampling.py."
        )
    ensure_coarse(enabled=enabled)
    if not p.is_file():
        raise FileNotFoundError(
            f"Band-sampled PLY not found: {p}\n"
            f"coarse/ is on disk but carries no point cloud at this count.  "
            f"Generate one with "
            f"datasets/fuzzy_dataset/coarse/band_sampling.py."
        )


def ensure_checkpoints(subdir: str = "", *, enabled: bool = True) -> Path:
    """Download released checkpoints if `subdir` holds no .npz yet."""
    out = LOCAL_DIR / "checkpoints" / subdir if subdir else LOCAL_DIR / "checkpoints"
    if out.is_dir() and any(out.rglob("*.npz")):
        return out
    if not enabled:
        raise FileNotFoundError(
            f"no checkpoints under {out}\n"
            f"Fetch them with:\n"
            f"  python scripts/fetch_data.py checkpoints"
        )
    patterns = [f"checkpoints/{subdir}/*"] if subdir else list(CHECKPOINT_PATTERNS)
    download(patterns, what=f"released checkpoints ({subdir or 'all'})")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("train", "checkpoints", "all"),
                    help="train: capture/ + coarse/ (4.3 GB).  "
                         "checkpoints: the released line models (7.3 GB).  "
                         "all: the whole dataset (50 GB).")
    args = ap.parse_args()

    if args.mode == "train":
        download(capture_patterns() + list(COARSE_PATTERNS),
                 what="capture/ and coarse/ for all eight Fuzzy scenes")
    elif args.mode == "checkpoints":
        download(list(CHECKPOINT_PATTERNS), what="released checkpoints")
    else:
        download(None, what=f"the whole {REPO_ID} dataset")


if __name__ == "__main__":
    main()
