# utils.py
#
# Shared utilities for the training and evaluation scripts.
#
# Sections
# --------
#   1. Camera math       - perspective_rh_zo, spatial_lr_scale
#   2. Training misc     - to_u8, select_device
#   3. PLY loading       - load_ply_positions
#   4. Primitives        - make_line_primitives_from_ply,
#                          make_triangle_primitives_from_ply
#   5. NeRF data         - read_png_rgba, composite_white_bg,
#                          cameras_from_nerf_frames, load_nerf_synthetic,
#                          load_nerf_per_frame, nerf_frame_filenames
#   6. COLMAP data       - read_colmap_extrinsics_binary, read_colmap_intrinsics_binary,
#                          read_colmap_points3D_binary, load_colmap_dataset
#   7. SH utilities      - SH_BAND_SIZES, SH_NUM_COEFFS
#   8. Checkpoint I/O    - save_checkpoint, load_checkpoint

from __future__ import annotations

import collections
import json
import math
import os
import struct
import time

import imageio.v2 as imageio
import numpy as np
import torch


# =============================================================================
# 1. Camera math
# =============================================================================


def perspective_rh_zo(fovy_rad: float, aspect: float, z_near: float, z_far: float) -> torch.Tensor:
    """Right-handed perspective matrix, depth in [0, 1] (Vulkan convention).

    Parameters
    ----------
    fovy_rad : vertical FoV in radians.
    aspect   : width / height.
    z_near, z_far : clip planes.

    Returns
    -------
    M : float32 [4, 4]
    """
    f = 1.0 / torch.tan(torch.tensor(0.5 * fovy_rad, dtype=torch.float32))
    M = torch.zeros((4, 4), dtype=torch.float32)
    M[0, 0] =  f / aspect
    M[1, 1] =  f
    M[2, 2] =  z_far / (z_near - z_far)
    M[3, 2] = -1.0
    M[2, 3] = (z_far * z_near) / (z_near - z_far)
    return M


def spatial_lr_scale(eyes: list[torch.Tensor]) -> float:
    """Compute a position-LR scale from the spread of camera positions.

    Returns 1.1 x max distance from any camera to the camera centroid.
    Multiply with a base learning rate to get a scene-adaptive position LR.
    """
    eye_np = torch.stack(eyes).numpy()
    centroid = eye_np.mean(axis=0)
    dists = np.linalg.norm(eye_np - centroid, axis=1)
    return float(dists.max()) * 1.1


# =============================================================================
# 2. Training misc
# =============================================================================

def to_u8(img: torch.Tensor) -> torch.Tensor:
    """Clamp a float [0, 1] image tensor to uint8 [0, 255].

    Works for any shape; the float -> uint8 rounding uses ``floor(x * 255 + 0.5)``.
    """
    return (torch.clamp(img, 0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)


def select_device(device_str: str, gpu_id: int = 0) -> torch.device:
    """Return a ``torch.device``, falling back to CPU if CUDA is unavailable.

    Parameters
    ----------
    device_str : ``"cpu"`` or ``"cuda"``.
    gpu_id     : CUDA device index (ignored for CPU).
    """
    if device_str.lower() == "cuda":
        if not torch.cuda.is_available():
            print("[torch] WARNING: CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        torch.cuda.set_device(int(gpu_id))
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# 3. PLY loading
# =============================================================================


def load_ply_positions(path: str) -> np.ndarray:
    """Load positions from a binary little-endian PLY (float x y z only)."""
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(-1, 3).copy()


# =============================================================================
# 4. Primitives
# =============================================================================


def knn1_dists(pts: torch.Tensor) -> torch.Tensor:
    """Per-point distance to the nearest neighbor (KNN-1) via KD-tree."""
    from scipy.spatial import KDTree
    tree = KDTree(pts.numpy())
    dd, _ = tree.query(pts.numpy(), k=2)  # k=2: closest is self (dist=0)
    return torch.as_tensor(dd[:, 1], dtype=pts.dtype)


def make_line_primitives_from_ply(
    num_lines: int,
    ply_path: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build line segments from pre-sampled midpoints in a PLY file.

    Loads positions from *ply_path*, randomly sub-samples *num_lines* of them
    as midpoints, then creates line segments with random directions.
    All segments share a uniform length of ``mean(KNN-1 distance)``.

    Returns
    -------
    verts    : float32 [num_lines * 2, 3]
    lines    : uint32  [num_lines, 2]
    init_line_length : float   - the uniform segment length
    """
    pts = load_ply_positions(ply_path)  # (M, 3)
    if pts.shape[0] < num_lines:
        # Oversample with replacement
        idx = np.random.randint(0, pts.shape[0], size=num_lines)
    else:
        idx = np.random.choice(pts.shape[0], size=num_lines, replace=False)
    midpoints = torch.as_tensor(pts[idx], dtype=torch.float32)

    nn_dists = knn1_dists(midpoints).clamp_min(1e-7)
    init_line_length = float(nn_dists.mean())

    dirs = torch.randn((num_lines, 3), dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8) * (init_line_length * 0.5)

    verts = torch.stack([midpoints - dirs, midpoints + dirs], dim=1).reshape(num_lines * 2, 3).contiguous()

    base = torch.arange(num_lines, dtype=torch.int64) * 2
    lines = torch.stack([base, base + 1], dim=1).to(torch.uint32).contiguous()
    return verts, lines, init_line_length


def make_triangle_primitives_from_ply(
    num_tris: int,
    ply_path: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build right triangles from pre-sampled seed points in a PLY file.

    Loads positions from *ply_path*, randomly sub-samples *num_tris* of them
    as seed origins, then spawns one right triangle per seed:
      v0 = origin
      v1 = origin + dir1 * init_edge_length
      v2 = origin + dir2 * init_edge_length
    where (dir1, dir2) is a random orthonormal pair in 3D.

    Leg length ``init_edge_length = mean(KNN-1 distance of seeds)``.

    Returns
    -------
    verts    : float32 [num_tris * 3, 3]  - unshared vertex buffer
    faces    : uint32  [num_tris, 3]      - sequential triples [0,1,2], [3,4,5], ...
    init_edge_length : float                      - the uniform leg length
    """
    pts = load_ply_positions(ply_path)  # (M, 3)
    if pts.shape[0] < num_tris:
        idx = np.random.randint(0, pts.shape[0], size=num_tris)
    else:
        idx = np.random.choice(pts.shape[0], size=num_tris, replace=False)
    origins = torch.as_tensor(pts[idx], dtype=torch.float32)

    nn_dists = knn1_dists(origins).clamp_min(1e-7)
    init_edge_length = float(nn_dists.mean())

    dir1 = torch.randn((num_tris, 3), dtype=torch.float32)
    dir1 = dir1 / dir1.norm(dim=1, keepdim=True).clamp_min(1e-8)
    aux = torch.randn((num_tris, 3), dtype=torch.float32)
    dir2 = torch.cross(dir1, aux, dim=1)
    dir2 = dir2 / dir2.norm(dim=1, keepdim=True).clamp_min(1e-8)

    v0 = origins
    v1 = origins + dir1 * init_edge_length
    v2 = origins + dir2 * init_edge_length

    verts = torch.stack([v0, v1, v2], dim=1).reshape(num_tris * 3, 3).contiguous()
    base = torch.arange(num_tris, dtype=torch.int64) * 3
    faces = torch.stack([base, base + 1, base + 2], dim=1).to(torch.uint32).contiguous()
    return verts, faces, init_edge_length


# =============================================================================
# 5. NeRF data loading
# =============================================================================

def _nerf_normalize_path(fp: str) -> str:
    """Normalize a NeRF JSON ``file_path`` (strip leading ``./``, unify slashes)."""
    return fp.replace("\\", "/").removeprefix("./")


def read_png_rgba(path: str) -> torch.Tensor:
    """Read a PNG file as a float32 RGBA tensor on CPU.

    Alpha is synthesized as 1.0 if the file has no alpha channel.

    Returns
    -------
    rgba : float32 [H, W, 4]  values in [0, 1]
    """
    arr = imageio.imread(path)   # uint8 HxWx3 or HxWx4
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape {arr.shape} for {path}")
    t = torch.from_numpy(arr).to(torch.float32) / 255.0
    if t.shape[2] == 3:
        a = torch.ones((*t.shape[:2], 1), dtype=torch.float32)
        t = torch.cat([t, a], dim=2)
    return t.contiguous()


def composite_white_bg(rgba01: torch.Tensor) -> torch.Tensor:
    """Alpha-composite an RGBA image over a white background.

    Parameters
    ----------
    rgba01 : float32 [..., 4]  values in [0, 1]

    Returns
    -------
    rgb : float32 [..., 3]  values in [0, 1]
    """
    rgb   = rgba01[..., :3]
    alpha = rgba01[..., 3:4]
    return (rgb * alpha + (1.0 - alpha)).contiguous()


def cameras_from_nerf_frames(
    frames:      list[dict],
    cam_angle_x: float,
    width:       int,
    height:      int,
    z_near:      float,
    z_far:       float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build MVP matrices and camera positions from NeRF Synthetic JSON frames.

    The projection is right-handed with depth in [0, 1] and Y flipped
    (``proj[1,1] *= -1``).

    Parameters
    ----------
    frames      : list of frame dicts from ``transforms_*.json``.
    cam_angle_x : horizontal FoV in radians (``camera_angle_x`` field).
    width, height : image resolution in pixels.
    z_near, z_far : clip planes.

    Returns
    -------
    mvps : list of float32 [4, 4]  - model-view-projection matrices (CPU)
    eyes : list of float32 [3]     - camera world positions (CPU)
    """
    focal      = 0.5 * float(width) / math.tan(0.5 * cam_angle_x)
    cam_angle_y = 2.0 * math.atan(0.5 * float(height) / focal)
    proj = perspective_rh_zo(cam_angle_y, float(width) / float(height), z_near, z_far)
    proj[1, 1] *= -1.0   # Vulkan Y-flip

    mvps: list[torch.Tensor] = []
    eyes: list[torch.Tensor] = []
    for fr in frames:
        c2w  = torch.tensor(fr["transform_matrix"], dtype=torch.float32).contiguous()
        view = torch.linalg.inv(c2w)
        mvps.append((proj @ view).contiguous())
        eyes.append(c2w[:3, 3].contiguous())
    return mvps, eyes


def load_nerf_synthetic(
    scene_root: str,
    split:      str,
    z_near:     float,
    z_far:      float,
    *,
    dataset: str = "nerf_synthetic",
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], int, int]:
    """Load images and cameras from a NeRF Synthetic (or compatible) dataset.

    Images are read as RGBA and composited over white.
    PNG extension is appended when ``dataset == "nerf_synthetic"``.

    Parameters
    ----------
    scene_root : path to the scene directory (contains ``transforms_*.json``).
    split      : ``"train"``, ``"val"``, or ``"test"``.
    z_near, z_far : clip planes for the projection matrix.
    dataset    : dataset flavor - controls whether ``.png`` is appended.

    Returns
    -------
    images : float32 [V, H, W, 3]  - white-composited RGB images, CPU
    mvps   : list[float32 [4, 4]]  - MVP matrices, CPU
    eyes   : list[float32 [3]]     - camera positions, CPU
    width  : int
    height : int
    """
    tf_path = os.path.join(scene_root, f"transforms_{split}.json")
    with open(tf_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cam_angle_x: float = float(meta["camera_angle_x"])
    frames = meta["frames"]
    ext    = ".png" if dataset == "nerf_synthetic" else ""

    # Derive resolution from first frame
    fp0  = _nerf_normalize_path(frames[0]["file_path"]) + ext
    rgba0 = read_png_rgba(os.path.join(scene_root, fp0))
    H, W  = int(rgba0.shape[0]), int(rgba0.shape[1])

    imgs = []
    for fr in frames:
        fp   = _nerf_normalize_path(fr["file_path"]) + ext
        rgba = read_png_rgba(os.path.join(scene_root, fp))
        imgs.append(composite_white_bg(rgba))

    images = torch.stack(imgs, dim=0).to(dtype=torch.float32).contiguous()
    mvps, eyes = cameras_from_nerf_frames(frames, cam_angle_x, W, H, z_near, z_far)
    return images, mvps, eyes, W, H



def nerf_frame_filenames(
    scene_root: str,
    split:      str,
    *,
    dataset: str = "nerf_synthetic",
) -> list[str]:
    """Source image filenames of a split, ordered as ``load_nerf_synthetic`` returns them.

    Parameters
    ----------
    scene_root : path to the scene directory (contains ``transforms_*.json``).
    split      : ``"train"``, ``"val"``, or ``"test"``.
    dataset    : dataset flavor - controls whether ``.png`` is appended.

    Returns
    -------
    names : list[str] - one basename per frame, e.g. ``["0001.png", ...]``
    """
    tf_path = os.path.join(scene_root, f"transforms_{split}.json")
    with open(tf_path, "r", encoding="utf-8") as f:
        frames = json.load(f)["frames"]

    ext = ".png" if dataset == "nerf_synthetic" else ""
    return [os.path.basename(_nerf_normalize_path(fr["file_path"]) + ext) for fr in frames]


def load_nerf_per_frame(
    scene_root:      str,
    transforms_name: str,
    z_near:          float,
    z_far:           float,
    *,
    target_width:    int | None = None,
    bg_color:        tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], int, int, list[str]]:
    """Load a JSON with per-frame intrinsics.

    Each frame must carry ``fl_x``, ``fl_y``, ``cx``, ``cy``, ``w``, ``h``
    alongside ``transform_matrix`` and ``file_path``.  Images are loaded from
    whatever ``file_path`` says (so ``transforms_masked.json`` -> RGBA mask
    images works out of the box).  Intrinsics are assumed uniform across
    frames (a warning is printed if any vary); a single symmetric projection
    matrix is built from the first frame's intrinsics scaled to the chosen
    output resolution.

    Parameters
    ----------
    scene_root      : path to the dataset root that contains the JSON.
    transforms_name : JSON filename, e.g. ``"transforms_masked.json"``.
    z_near, z_far   : clip planes.
    target_width    : if set, all images are resized to this width
                      (height scaled proportionally).  ``None`` keeps the
                      original resolution.
    bg_color        : RGB triple in [0, 1] used to composite the alpha
                      channel.  Default is black.

    Returns
    -------
    images    : float32 [V, H, W, 3]  - composited images, CPU
    mvps      : list[float32 [4, 4]]  - per-frame MVPs, CPU
    eyes      : list[float32 [3]]     - camera world positions, CPU
    width     : int
    height    : int
    basenames : list[str]             - per-frame os.path.basename(file_path),
                                        as written in the JSON (may or may not
                                        include an extension).
    """
    tf_path = os.path.join(scene_root, transforms_name)
    with open(tf_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta["frames"]

    fl_x_set = {round(float(fr["fl_x"]), 3) for fr in frames}
    fl_y_set = {round(float(fr["fl_y"]), 3) for fr in frames}
    cx_set   = {round(float(fr["cx"]),   3) for fr in frames}
    cy_set   = {round(float(fr["cy"]),   3) for fr in frames}
    if len(fl_x_set) > 1 or len(fl_y_set) > 1 or len(cx_set) > 1 or len(cy_set) > 1:
        print(f"[load_nerf_per_frame] WARNING: intrinsics vary across "
              f"frames ({len(fl_x_set)} fl_x, {len(fl_y_set)} fl_y, "
              f"{len(cx_set)} cx, {len(cy_set)} cy distinct values). "
              f"Using first-frame intrinsics for the projection matrix.")

    fr0   = frames[0]
    W_orig = int(fr0["w"])
    H_orig = int(fr0["h"])
    if target_width is not None and int(target_width) != W_orig:
        scale = float(target_width) / float(W_orig)
        W = int(target_width)
        H = int(round(H_orig * scale))
    else:
        scale = 1.0
        W, H  = W_orig, H_orig

    fl_y_scaled = float(fr0["fl_y"]) * scale
    fovy = 2.0 * math.atan(0.5 * float(H) / fl_y_scaled)
    proj = perspective_rh_zo(fovy, float(W) / float(H), z_near, z_far)
    proj[1, 1] *= -1.0   # Y-flip

    bg = torch.tensor(bg_color, dtype=torch.float32).view(1, 1, 3)

    imgs: list[torch.Tensor] = []
    mvps: list[torch.Tensor] = []
    eyes: list[torch.Tensor] = []
    basenames: list[str] = []
    for fr in frames:
        fp       = _nerf_normalize_path(fr["file_path"])
        rgba     = read_png_rgba(os.path.join(scene_root, fp))   # [H_orig, W_orig, 4]
        if scale != 1.0:
            t = rgba.permute(2, 0, 1).unsqueeze(0)               # [1, 4, H_orig, W_orig]
            t = torch.nn.functional.interpolate(
                t, size=(H, W), mode="bilinear", align_corners=False)
            rgba = t.squeeze(0).permute(1, 2, 0).contiguous()
        rgb   = rgba[..., :3]
        alpha = rgba[..., 3:4]
        composited = (rgb * alpha + bg * (1.0 - alpha)).contiguous()
        imgs.append(composited)

        c2w  = torch.tensor(fr["transform_matrix"], dtype=torch.float32).contiguous()
        view = torch.linalg.inv(c2w)
        mvps.append((proj @ view).contiguous())
        eyes.append(c2w[:3, 3].contiguous())
        basenames.append(os.path.basename(fp))

    images = torch.stack(imgs, dim=0).to(dtype=torch.float32).contiguous()
    return images, mvps, eyes, W, H, basenames


# =============================================================================
# 6. COLMAP data loading
# =============================================================================

_ColmapCamera = collections.namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"])
_ColmapImage = collections.namedtuple(
    "ColmapImage", ["id", "qvec", "tvec", "camera_id", "name"])

_COLMAP_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def _qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_colmap_extrinsics_binary(path: str) -> dict[int, _ColmapImage]:
    """Read COLMAP images.bin -> dict of image_id -> _ColmapImage."""
    images = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            ch = _read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = _read_next_bytes(fid, 1, "c")[0]
            num_pts2d = _read_next_bytes(fid, 8, "Q")[0]
            _read_next_bytes(fid, 24 * num_pts2d, "ddq" * num_pts2d)
            images[image_id] = _ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name)
    return images


def read_colmap_intrinsics_binary(path: str) -> dict[int, _ColmapCamera]:
    """Read COLMAP cameras.bin -> dict of camera_id -> _ColmapCamera."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(fid, 24, "iiQQ")
            camera_id = props[0]
            model_id = props[1]
            width = props[2]
            height = props[3]
            model_name, num_params = _COLMAP_MODEL_IDS[model_id]
            params = np.array(
                _read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = _ColmapCamera(
                id=camera_id, model=model_name,
                width=width, height=height, params=params)
    return cameras


def read_colmap_points3D_binary(path: str) -> np.ndarray:
    """Read COLMAP points3D.bin -> float32 [N, 3] positions."""
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3), dtype=np.float64)
        for i in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            xyzs[i] = props[1:4]
            track_len = _read_next_bytes(fid, 8, "Q")[0]
            _read_next_bytes(fid, 8 * track_len, "ii" * track_len)
    return xyzs.astype(np.float32)


def load_colmap_dataset(
    scene_root: str,
    split: str,
    z_near: float,
    z_far: float,
    resolution: int = 1,
    llffhold: int = 8,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], int, int,
           np.ndarray | None]:
    """Load a COLMAP dataset.

    Cameras are sorted by image name; every ``llffhold``-th camera (0-indexed)
    goes to the test set.  Images are loaded from ``images_{resolution}/``
    when *resolution* > 1, otherwise from ``images/``.

    Parameters
    ----------
    scene_root : path to the scene directory (contains ``sparse/``, ``images/``).
    split      : ``"train"`` or ``"test"``.
    z_near, z_far : clip planes for the projection matrix.
    resolution : downsample factor (1, 2, 4, or 8).
    llffhold   : hold-out stride for the test split.

    Returns
    -------
    images    : float32 [V, H, W, 3]  - white-composited RGB images, CPU
    mvps      : list[float32 [4, 4]]  - MVP matrices, CPU
    eyes      : list[float32 [3]]     - camera positions, CPU
    width     : int
    height    : int
    sfm_pts   : float32 [P, 3]  - SfM points (None if split == "test")
    """
    sparse_dir = os.path.join(scene_root, "sparse", "0")
    cam_extrinsics = read_colmap_extrinsics_binary(
        os.path.join(sparse_dir, "images.bin"))
    cam_intrinsics = read_colmap_intrinsics_binary(
        os.path.join(sparse_dir, "cameras.bin"))

    # Build per-image camera info, sorted by name.
    cam_list = []
    for _key in cam_extrinsics:
        extr = cam_extrinsics[_key]
        intr = cam_intrinsics[extr.camera_id]
        cam_list.append((extr, intr))
    cam_list.sort(key=lambda x: os.path.basename(x[0].name).split(".")[0])

    # Train/test split: every llffhold-th camera is test.
    if split == "train":
        cam_list = [c for i, c in enumerate(cam_list) if i % llffhold != 0]
    elif split == "test":
        cam_list = [c for i, c in enumerate(cam_list) if i % llffhold == 0]

    # Image directory.
    if resolution > 1:
        img_dir = os.path.join(scene_root, f"images_{resolution}")
    else:
        img_dir = os.path.join(scene_root, "images")

    # Load images and build cameras.
    imgs = []
    mvps: list[torch.Tensor] = []
    eyes: list[torch.Tensor] = []
    W_out, H_out = None, None

    for extr, intr in cam_list:
        # Load image.
        img_path = os.path.join(img_dir, os.path.basename(extr.name))
        rgba = read_png_rgba(img_path)
        rgb = composite_white_bg(rgba)
        H_img, W_img = int(rgb.shape[0]), int(rgb.shape[1])

        if W_out is None:
            W_out, H_out = W_img, H_img

        imgs.append(rgb)

        # Compute FoV from COLMAP intrinsics, scaled by actual image size.
        if intr.model == "PINHOLE":
            fx, fy = intr.params[0], intr.params[1]
        elif intr.model == "SIMPLE_PINHOLE":
            fx = fy = intr.params[0]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        # Scale focal lengths from COLMAP resolution to loaded image resolution.
        scale_x = W_img / intr.width
        scale_y = H_img / intr.height
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y

        fov_x = 2.0 * math.atan(0.5 * W_img / fx_scaled)
        fov_y = 2.0 * math.atan(0.5 * H_img / fy_scaled)

        # Projection matrix (Vulkan convention, Y-flipped).
        proj = perspective_rh_zo(fov_y, W_img / H_img, z_near, z_far)
        proj[1, 1] *= -1.0

        # View matrix from COLMAP extrinsics.
        R_w2c = _qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        # Build c2w: camera center = -R_w2c^T @ tvec
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = -R_w2c.T @ T
        # Convert COLMAP (Y down, Z forward) to OpenGL (Y up, Z back)
        # so that perspective_rh_zo (which looks down -Z) works correctly.
        c2w[:3, 1] *= -1  # flip Y
        c2w[:3, 2] *= -1  # flip Z
        c2w = torch.tensor(c2w, dtype=torch.float32)

        view = torch.linalg.inv(c2w)
        mvps.append((proj @ view).contiguous())
        eyes.append(c2w[:3, 3].contiguous())

    images = torch.stack(imgs, dim=0).to(dtype=torch.float32).contiguous()

    # Load SfM points (only needed for training init).
    sfm_pts = None
    if split == "train":
        pts_path = os.path.join(sparse_dir, "points3D.bin")
        if os.path.exists(pts_path):
            sfm_pts = read_colmap_points3D_binary(pts_path)

    return images, mvps, eyes, W_out, H_out, sfm_pts


# =============================================================================
# 7. SH utilities  (real spherical harmonics l = 0 .. 3,  16 coefficients)
# =============================================================================

# Number of coefficients per SH band (l = 0, 1, 2, 3).
SH_BAND_SIZES: tuple[int, ...] = (1, 3, 5, 7)
# Total coefficients for l = 0 .. 3.
SH_NUM_COEFFS: int = sum(SH_BAND_SIZES)   # 16


# =============================================================================
# 8. Checkpoint I/O
# =============================================================================
#
# File format  (compressed NumPy .npz, format_version=1)
# -------------------------------------------------------
# Required arrays
#   verts               float32 [N, 3]    world-space vertex positions
#   colors              float32 [N, C]    per-vertex color data
#                                           C=3  -> plain RGB  (color_mode="rgb")
#                                           C=K*3 -> SH interleaved (color_mode="sh")
#                                                   colors[n, k*3 : k*3+3] = [R, G, B]
#                                                   for SH coefficient k; so colors[n, 0:3]
#                                                   is always the DC (l=0) term.
#   faces               uint32  [M, 3]    triangle index triples
#   face_opacity_logit  float32 [M]       raw logits; sigmoid gives probability
#
# Optional arrays  (zero-row if absent, presence indicated by has_* meta flags)
#   lines               uint32  [L, 2]    edge index pairs
#   line_opacity_logit  float32 [L]       raw logits for line opacity
#   radius              float32 [N]       per-vertex world-space radius
#
# Metadata  (stored as JSON string in the "meta_json" scalar array)
#   format_version      int               1
#   color_mode          str               "rgb" | "sh"
#   sh_degree           int               max SH degree (0 = DC only = plain RGB)
#   color_channels      int               C (= 3 for RGB, = 16*3 for SH degree-3)
#   has_lines           bool
#   has_line_opacity    bool
#   has_radius          bool
#   n_verts, n_faces, n_lines  int
#   saved_at            str               ISO timestamp
#   + any caller-supplied extra_meta fields

def save_checkpoint(
    path: str,
    *,
    verts:               torch.Tensor,              # [N, 3]  float32
    colors:              torch.Tensor,              # [N, C]  float32
    faces:               torch.Tensor | None = None,             # [M, 3]  uint32
    face_opacity_logit:  torch.Tensor | None = None,             # [M]     float32
    lines:               torch.Tensor | None = None,             # [L, 2]  uint32
    line_opacity_logit:  torch.Tensor | None = None,             # [L]     float32
    radius:              torch.Tensor | None = None,             # [N]     float32
    color_mode:  str = "rgb",
    sh_degree:   int = 0,
    extra_meta:  dict | None = None,
) -> None:
    """Save a primitive checkpoint to a compressed NumPy archive.

    All tensors are moved to CPU and cast to their canonical dtypes before
    saving.  Optional arrays are stored as zero-row placeholders so that
    readers never need to handle missing keys.

    Parameters
    ----------
    path               : output path (should end in ``.npz``).
    verts              : world-space positions [N, 3].
    colors             : per-vertex color / SH coefficients [N, C].
                         For SH, store in interleaved layout:
                         ``sh_all.permute(0, 2, 1).reshape(N, -1)``
                         so that ``colors[:, 0:3]`` is always the DC term.
    faces              : triangle index triples [M, 3].
    face_opacity_logit : per-face raw opacity logits [M].
    lines              : edge index pairs [L, 2], or None.
    line_opacity_logit : per-line raw opacity logits [L], or None.
    radius             : per-vertex world-space radius [N], or None.
    color_mode         : ``"rgb"`` or ``"sh"``.
    sh_degree          : maximum SH degree stored (0 for plain RGB).
    extra_meta         : optional dict merged into the JSON metadata.
    """
    def _to_np(t: torch.Tensor, dtype) -> np.ndarray:
        return t.detach().cpu().to(dtype).numpy()

    has_lines      = lines              is not None
    has_line_op    = line_opacity_logit is not None
    has_radius     = radius             is not None

    has_faces      = faces              is not None
    N = int(verts.shape[0])
    M = int(faces.shape[0]) if has_faces else 0
    L = int(lines.shape[0]) if has_lines else 0

    meta: dict = {
        "format_version":  1,
        "color_mode":      color_mode,
        "sh_degree":       sh_degree,
        "color_channels":  int(colors.shape[1]) if colors.ndim == 2 else 3,
        "has_lines":       has_lines,
        "has_line_opacity": has_line_op,
        "has_radius":      has_radius,
        "n_verts":  N,
        "n_faces":  M,
        "n_lines":  L,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)

    arrays: dict[str, np.ndarray] = {
        "verts":               _to_np(verts,              torch.float32),
        "colors":              _to_np(colors,             torch.float32),
        "faces":               (
            _to_np(faces, torch.uint32) if has_faces
            else np.zeros((0, 3), dtype=np.uint32)
        ),
        "face_opacity_logit":  (
            _to_np(face_opacity_logit, torch.float32)
            if face_opacity_logit is not None
            else np.zeros(0, dtype=np.float32)
        ),
        "lines":               (
            _to_np(lines, torch.uint32) if has_lines
            else np.zeros((0, 2), dtype=np.uint32)
        ),
        "line_opacity_logit":  (
            _to_np(line_opacity_logit, torch.float32) if has_line_op
            else np.zeros(0, dtype=np.float32)
        ),
        "radius":              (
            _to_np(radius, torch.float32) if has_radius
            else np.zeros(0, dtype=np.float32)
        ),
        "meta_json":           np.array(json.dumps(meta)),
    }
    np.savez(path, **arrays)
    print(f"[ckpt] Saved -> {path}  "
          f"(verts={N}, faces={M}, lines={L}, color_mode={color_mode})")


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns a plain ``dict`` whose values are NumPy arrays, plus a parsed
    ``"meta"`` key containing the JSON metadata dict.  Unknown array keys
    are passed through unchanged for forward-compatibility.
    """
    data = np.load(path, allow_pickle=True)
    out  = {k: data[k] for k in data.files}
    if "meta_json" in out:
        out["meta"] = json.loads(str(out["meta_json"]))
    return out
