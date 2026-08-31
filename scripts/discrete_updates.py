# discrete_updates.py
"""Topology updates for line and triangle primitives.

``relining`` updates line primitives; ``remesh`` updates triangle primitives.
"""
from __future__ import annotations

import torch


def _snap_endpoints(
    positions: torch.Tensor,    # float32 [N, 3]
    lines_i64: torch.Tensor,    # int64   [L, 2]
    line_opacity: torch.Tensor, # float32 [L]
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge degree-1 vertices closer than ``threshold``.

    For each merged pair the first vertex is kept, references to the second
    are redirected to it, and lines that become degenerate are removed.
    """
    device = positions.device
    N = positions.shape[0]

    degree = torch.zeros(N, dtype=torch.int32, device=device)
    degree.scatter_add_(0, lines_i64.reshape(-1),
                        torch.ones(lines_i64.numel(), dtype=torch.int32, device=device))
    ep_idx = (degree == 1).nonzero(as_tuple=False).squeeze(1)
    _empty = torch.empty(0, dtype=torch.int64, device=device)
    if ep_idx.shape[0] < 2:
        return lines_i64, line_opacity, _empty

    ep_pos = positions[ep_idx]
    cell_size = max(threshold, 1e-8)
    cell_coords = torch.floor(ep_pos / cell_size).to(torch.int64)
    P1, P2, P3 = 73856093, 19349663, 83492791
    cell_hash = (cell_coords[:, 0] * P1
                 + cell_coords[:, 1] * P2
                 + cell_coords[:, 2] * P3)
    sorted_order = torch.argsort(cell_hash)
    sorted_hash = cell_hash[sorted_order]
    sorted_pos = ep_pos[sorted_order]
    sorted_vidx = ep_idx[sorted_order]

    same_cell = sorted_hash[:-1] == sorted_hash[1:]
    if not same_cell.any():
        return lines_i64, line_opacity, _empty

    dist = (sorted_pos[:-1][same_cell] - sorted_pos[1:][same_cell]).norm(dim=1)
    close = dist < threshold
    if not close.any():
        return lines_i64, line_opacity, _empty

    va = sorted_vidx[:-1][same_cell][close]
    vb = sorted_vidx[1:][same_cell][close]

    # Greedy: each vertex participates in at most one merge.
    used = torch.zeros(N, dtype=torch.bool, device=device)
    keep: list[int] = []
    remove: list[int] = []
    for i in range(va.shape[0]):
        a, b = va[i].item(), vb[i].item()
        if not used[a] and not used[b]:
            keep.append(a)
            remove.append(b)
            used[a] = True
            used[b] = True

    if len(keep) == 0:
        return lines_i64, line_opacity, _empty

    keep_t = torch.tensor(keep, dtype=torch.int64, device=device)
    remove_t = torch.tensor(remove, dtype=torch.int64, device=device)

    remap = torch.arange(N, dtype=torch.int64, device=device)
    remap[remove_t] = keep_t
    lines_i64 = remap[lines_i64.reshape(-1)].reshape(-1, 2)

    # Remove degenerate lines (both endpoints same vertex after merge).
    valid = lines_i64[:, 0] != lines_i64[:, 1]
    lines_i64 = lines_i64[valid]
    line_opacity = line_opacity[valid]
    return lines_i64, line_opacity, keep_t


def relining(
    positions: torch.Tensor,          # float32 [N, 3]
    vert_attrs: torch.Tensor,         # float32 [N, C]
    lines: torch.Tensor,              # uint32  [L, 2]
    line_opacity: torch.Tensor,       # float32 [L]
    *,
    opacity_threshold: float,
    length_threshold: float = 0.0,
    target_num_verts: int,
    snap_endpoints: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune and split line primitives.

    Lines with opacity below ``opacity_threshold``, or shorter than
    ``length_threshold``, are removed.  The longest remaining lines are then
    split at their midpoint until the freed vertex slots are refilled, keeping
    the vertex count at ``target_num_verts``.

    Returns ``(positions, vert_attrs, lines, line_opacity, vert_origin)``.
    """
    device = positions.device
    lines_i64 = lines.to(torch.int64)

    # 1. Prune low-opacity and short lines.
    dead = line_opacity < opacity_threshold
    if length_threshold > 0.0:
        v0 = positions[lines_i64[:, 0]]
        v1 = positions[lines_i64[:, 1]]
        edge_len_sq = ((v1 - v0) ** 2).sum(dim=1)
        dead = dead | (edge_len_sq < length_threshold ** 2)
    lines_i64 = lines_i64[~dead]
    line_opacity = line_opacity[~dead]

    # 2. Snap close degree-1 endpoints.
    snapped_verts = torch.empty(0, dtype=torch.int64, device=device)
    if snap_endpoints and length_threshold > 0.0:
        lines_i64, line_opacity, snapped_verts = _snap_endpoints(
            positions, lines_i64, line_opacity, length_threshold,
        )

    # Grow the vertex buffer if the target exceeds its current size.
    N = positions.shape[0]
    if target_num_verts > N:
        C = vert_attrs.shape[1]
        pad_n = target_num_verts - N
        positions = torch.cat([
            positions,
            torch.zeros((pad_n, 3), dtype=positions.dtype, device=device),
        ], dim=0)
        vert_attrs = torch.cat([
            vert_attrs,
            torch.zeros((pad_n, C), dtype=vert_attrs.dtype, device=device),
        ], dim=0)

    # Find free vertex slots (not referenced by any alive line)
    alive_verts = torch.zeros(positions.shape[0], dtype=torch.bool, device=device)
    alive_verts[lines_i64.reshape(-1)] = True
    free_vert_idx = (~alive_verts).nonzero(as_tuple=False).squeeze(1)

    # 3. Split edges until every free vertex slot is used.  Lines created by
    #    a split can themselves be split in the next round.

    # vert_origin: identity map for original verts, -1 for grown/snapped slots.
    vert_origin = torch.arange(positions.shape[0], dtype=torch.int32, device=device)
    if positions.shape[0] > N:
        vert_origin[N:] = -1
    if snapped_verts.numel() > 0:
        vert_origin[snapped_verts] = -1

    slot_cursor = 0
    n_free = int(free_vert_idx.shape[0])

    # This loop only matters when more than half the lines are pruned.  In
    # this case, a single pass can't reach the target vertex count, since
    # splitting every surviving line once at most doubles their number.
    # This can happen for a highly random initialization, but is rare in
    # practice, such as under initialization of the lines from a coarse
    # proxy, as done in the paper.  Kept for robustness; in our experiments
    # it usually runs one pass, like the single if statement in remesh().
    while slot_cursor < n_free and lines_i64.shape[0] > 0:
        n_avail = n_free - slot_cursor
        n_lines = lines_i64.shape[0]
        n_splits = min(n_avail, n_lines)

        v0 = positions[lines_i64[:, 0]]
        v1 = positions[lines_i64[:, 1]]
        edge_len = ((v1 - v0) ** 2).sum(dim=1).sqrt()

        # split the longest edges
        _, sample_idx = torch.topk(edge_len, n_splits)

        split_va = lines_i64[sample_idx, 0]
        split_vb = lines_i64[sample_idx, 1]

        vm_slots = free_vert_idx[slot_cursor : slot_cursor + n_splits]
        positions[vm_slots] = 0.5 * (positions[split_va] + positions[split_vb])
        vert_attrs[vm_slots] = 0.5 * (vert_attrs[split_va] + vert_attrs[split_vb])
        vert_origin[vm_slots] = -1

        lines_i64[sample_idx, 1] = vm_slots

        new_lines = torch.stack([vm_slots, split_vb.to(torch.int64)], dim=1)
        new_opacity = line_opacity[sample_idx]
        lines_i64 = torch.cat([lines_i64, new_lines], dim=0)
        line_opacity = torch.cat([line_opacity, new_opacity], dim=0)

        slot_cursor += n_splits


    return (
        positions.contiguous(),
        vert_attrs.contiguous(),
        lines_i64.to(torch.uint32).contiguous(),
        line_opacity.contiguous(),
        vert_origin,
    )


def remesh(
    positions: torch.Tensor,          # float32 [N, 3]
    vert_attrs: torch.Tensor,         # float32 [N, C]
    faces: torch.Tensor,              # uint32  [F, 3]
    face_opacity: torch.Tensor,       # float32 [F]
    *,
    opacity_threshold: float,
    length_threshold: float = 0.0,
    shared_split: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune and split triangle primitives.

    Faces with opacity below ``opacity_threshold``, or whose longest edge is
    shorter than ``length_threshold``, are removed.  The faces with the longest
    edges are then split at that edge's midpoint until the freed vertex slots
    are refilled.

    With ``shared_split`` the two children share the new midpoint vertex
    (1 new vertex per split); otherwise each child gets its own copy
    (3 new vertices per split).

    Returns ``(positions, vert_attrs, faces, face_opacity, vert_origin)``.
    """
    device = positions.device
    faces_i64 = faces.to(torch.int64)

    # 1. Prune low opacity
    dead = face_opacity < opacity_threshold

    # 2. Prune small triangles (max edge < length_threshold)
    if length_threshold > 0.0:
        va = positions[faces_i64[:, 0]]
        vb = positions[faces_i64[:, 1]]
        vc = positions[faces_i64[:, 2]]
        e01_sq = ((vb - va) ** 2).sum(1)
        e12_sq = ((vc - vb) ** 2).sum(1)
        e20_sq = ((va - vc) ** 2).sum(1)
        max_edge_sq = torch.max(torch.max(e01_sq, e12_sq), e20_sq)
        dead = dead | (max_edge_sq < length_threshold ** 2)

    # 3. Compact dead faces
    faces_i64 = faces_i64[~dead]
    face_opacity = face_opacity[~dead]

    # 4. Find free vertex slots
    alive_verts = torch.zeros(positions.shape[0], dtype=torch.bool, device=device)
    alive_verts[faces_i64.reshape(-1)] = True
    free_vert_idx = (~alive_verts).nonzero(as_tuple=False).squeeze(1)

    n_faces = faces_i64.shape[0]

    # 5. Compute max edge length for split selection
    va = positions[faces_i64[:, 0]]
    vb = positions[faces_i64[:, 1]]
    vc = positions[faces_i64[:, 2]]
    e01 = ((vb - va) ** 2).sum(1).sqrt()
    e12 = ((vc - vb) ** 2).sum(1).sqrt()
    e20 = ((va - vc) ** 2).sum(1).sqrt()
    max_edge = torch.max(torch.max(e01, e12), e20)

    # 6. Split longest-edge triangles to fill free vertex slots
    # shared_split: 1 new vert (midpoint), apex shared between children.
    # unshared:    3 new verts (midpoint for each child + apex copy for new child).
    verts_per_split = 1 if shared_split else 3
    n_splits = min(int(free_vert_idx.shape[0]) // verts_per_split, n_faces)

    vert_origin = torch.arange(positions.shape[0], dtype=torch.int32, device=device)

    if n_splits > 0:
        # split the longest edges
        _, sample_idx = torch.topk(max_edge, n_splits)

        # Gather the 3 vertex indices per selected face
        f_v = faces_i64[sample_idx]  # [n_splits, 3]

        # Find which edge is longest for each selected face
        e_sq = torch.stack([
            ((positions[f_v[:, 1]] - positions[f_v[:, 0]]) ** 2).sum(1),  # edge 0-1
            ((positions[f_v[:, 2]] - positions[f_v[:, 1]]) ** 2).sum(1),  # edge 1-2
            ((positions[f_v[:, 0]] - positions[f_v[:, 2]]) ** 2).sum(1),  # edge 2-0
        ], dim=1)  # [n_splits, 3]
        longest = e_sq.argmax(dim=1)  # 0, 1, or 2

        # Map: edge 0 -> verts (0,1) opp 2; edge 1 -> (1,2) opp 0; edge 2 -> (2,0) opp 1
        idx_a = torch.tensor([0, 1, 2], device=device)[longest]
        idx_b = torch.tensor([1, 2, 0], device=device)[longest]
        idx_c = torch.tensor([2, 0, 1], device=device)[longest]

        ea = f_v.gather(1, idx_a.unsqueeze(1)).squeeze(1)
        eb = f_v.gather(1, idx_b.unsqueeze(1)).squeeze(1)
        ec = f_v.gather(1, idx_c.unsqueeze(1)).squeeze(1)

        midpoint_pos = 0.5 * (positions[ea] + positions[eb])
        midpoint_attr = 0.5 * (vert_attrs[ea] + vert_attrs[eb])

        if shared_split:
            # 1 new vert per split: midpoint. Apex shared.
            vm_slots = free_vert_idx[:n_splits]
            positions[vm_slots] = midpoint_pos
            vert_attrs[vm_slots] = midpoint_attr
            vert_origin[vm_slots] = -1

            # Original: (ea, vm, ec)
            faces_i64[sample_idx, 0] = ea
            faces_i64[sample_idx, 1] = vm_slots
            faces_i64[sample_idx, 2] = ec

            # New: (vm, eb, ec)
            new_faces = torch.stack([vm_slots, eb, ec], dim=1)
        else:
            # 3 new verts per split: vm1 for original, vm2+ec2 for new child.
            vm1_slots = free_vert_idx[0 * n_splits : 1 * n_splits]
            vm2_slots = free_vert_idx[1 * n_splits : 2 * n_splits]
            ec2_slots = free_vert_idx[2 * n_splits : 3 * n_splits]

            positions[vm1_slots] = midpoint_pos
            positions[vm2_slots] = midpoint_pos
            positions[ec2_slots] = positions[ec]
            vert_attrs[vm1_slots] = midpoint_attr
            vert_attrs[vm2_slots] = midpoint_attr
            vert_attrs[ec2_slots] = vert_attrs[ec]
            vert_origin[vm1_slots] = -1
            vert_origin[vm2_slots] = -1
            vert_origin[ec2_slots] = -1

            # Original: (ea, vm1, ec)
            faces_i64[sample_idx, 0] = ea
            faces_i64[sample_idx, 1] = vm1_slots
            faces_i64[sample_idx, 2] = ec

            # New: (vm2, eb, ec2)
            new_faces = torch.stack([vm2_slots, eb, ec2_slots], dim=1)

        new_opacity = face_opacity[sample_idx]
        faces_i64 = torch.cat([faces_i64, new_faces], dim=0)
        face_opacity = torch.cat([face_opacity, new_opacity], dim=0)

    return (
        positions.contiguous(),
        vert_attrs.contiguous(),
        faces_i64.to(torch.uint32).contiguous(),
        face_opacity.contiguous(),
        vert_origin,
    )
