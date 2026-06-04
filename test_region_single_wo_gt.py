import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

try:
    from skimage import filters, segmentation  # type: ignore
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    distance_transform_edt = None  # type: ignore

try:
    from model import LineArtTransformerModel  # type: ignore
except ImportError as e:
    LineArtTransformerModel = None
    _MODEL_IMPORT_ERROR = e


@dataclass
class PMESConfig:
    resize_hw: Tuple[int, int] = (512, 512)
    patch_size: int = 32
    bg_label: Optional[int] = None
    neigh_mode: str = "8"
    patch_thresh: float = 0.72
    edge_sigma: float = 2.0
    edge_gamma: float = 0.9
    watershed_line: bool = False
    min_region_ratio: float = 0.005
    out_dir: str = "custom_test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    debug: bool = False
    save_intermediates: bool = True
    seed: int = 42
    match_thresh: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_palette(n: int, seed: int = 0, bg_color: Tuple[int, int, int] = (200, 200, 200)) -> np.ndarray:
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (255, 192, 203),
        (128, 255, 0), (255, 0, 128), (0, 128, 255),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    cols = np.zeros((max(n, 1), 3), dtype=np.uint8)
    if n > 0:
        cols[0] = bg_color
    start_idx = 1 if n > 0 else 0
    for i in range(start_idx, min(n, len(predefined_colors) + start_idx)):
        cols[i] = predefined_colors[i - start_idx]
    if n > len(predefined_colors) + start_idx:
        rng = np.random.default_rng(seed)
        for i in range(len(predefined_colors) + start_idx, n):
            while True:
                color = rng.integers(0, 256, size=3, dtype=np.uint8)
                brightness = color.sum()
                if 150 < brightness < 600 and int(color.max()) - int(color.min()) > 100:
                    cols[i] = color
                    break
    return cols


def _pil_open_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_image(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    pil_img = Image.fromarray(img)
    return np.array(pil_img.resize((w, h), Image.BILINEAR), dtype=np.uint8)


def rgb2gray01(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        g = rgb.astype(np.float32)
    return g.astype(np.float32) / 255.0


@dataclass
class PatchMeta:
    H: int
    W: int
    P: int
    pr: int
    pc: int
    N: int

    @classmethod
    def from_hwP(cls, hw: Tuple[int, int], patch_size: int) -> "PatchMeta":
        h, w = hw
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"resize size {hw} must be divisible by patch size {patch_size}")
        pr, pc = h // patch_size, w // patch_size
        return cls(h, w, patch_size, pr, pc, pr * pc)

    def pid(self, r: int, c: int) -> int:
        return r * self.pc + c

    def rc(self, pid: int) -> Tuple[int, int]:
        return divmod(pid, self.pc)

    def patch_slice(self, pid: int) -> Tuple[slice, slice]:
        r, c = self.rc(pid)
        p = self.P
        return slice(r * p, (r + 1) * p), slice(c * p, (c + 1) * p)


def enumerate_patch_neighbors(meta: PatchMeta, mode: str = "4") -> List[Tuple[int, int]]:
    neigh: List[Tuple[int, int]] = []
    for r in range(meta.pr):
        for c in range(meta.pc):
            idx = meta.pid(r, c)
            if c + 1 < meta.pc:
                neigh.append((idx, meta.pid(r, c + 1)))
            if r + 1 < meta.pr:
                neigh.append((idx, meta.pid(r + 1, c)))
            if mode == "8":
                if r + 1 < meta.pr and c + 1 < meta.pc:
                    neigh.append((idx, meta.pid(r + 1, c + 1)))
                if r + 1 < meta.pr and c - 1 >= 0:
                    neigh.append((idx, meta.pid(r + 1, c - 1)))
    return neigh


def load_model(model_path: str, meta: PatchMeta, device: str = "cuda", patch_size: int = 32):
    if LineArtTransformerModel is None:
        raise ImportError(f"Import error when loading model: {_MODEL_IMPORT_ERROR}")
    model = LineArtTransformerModel(
        embed_dim=768,
        num_heads=12,
        num_layers=4,
        num_patches=meta.N,
        patch_size=patch_size,
    ).to(device)
    raw_state = torch.load(model_path, map_location=device)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    skip_keys = {"pos_embed", "encoder.pos_embedding", "_orig_pos_embed"}
    filtered = {k: v for k, v in cleaned.items() if k not in skip_keys}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def preprocess_for_model(rgb: np.ndarray) -> torch.Tensor:
    pil_img = Image.fromarray(rgb)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0)


def get_model_probs(model, ref_rgb: np.ndarray, tgt_rgb: np.ndarray, cfg: PMESConfig) -> torch.Tensor:
    ref_t = preprocess_for_model(ref_rgb).to(cfg.device)
    tgt_t = preprocess_for_model(tgt_rgb).to(cfg.device)
    with torch.no_grad():
        logits = model(ref_t, tgt_t)
    return torch.sigmoid(logits).cpu()


def slice_intra_blocks(prob_mat: torch.Tensor, meta: PatchMeta) -> Tuple[np.ndarray, np.ndarray]:
    n = meta.N
    if prob_mat.shape[1] != 2 * n:
        raise ValueError("Unexpected probability tensor shape")
    ref_ref = prob_mat[0, :n, :n].numpy()
    tgt_tgt = prob_mat[0, n:, n:].numpy()
    return ref_ref, tgt_tgt


class UnionFind:
    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def compress(self) -> None:
        for i in range(self.parent.shape[0]):
            self.parent[i] = self.find(i)


def mean_edge_on_border(edge_map: np.ndarray, pid_a: int, pid_b: int, meta: PatchMeta) -> float:
    ra, ca = meta.rc(pid_a)
    rb, cb = meta.rc(pid_b)
    p = meta.P
    if ra == rb and abs(ca - cb) == 1:
        x = max(ca, cb) * p
        y0, y1 = ra * p, (ra + 1) * p
        band = edge_map[y0:y1, max(x - 1, 0):x + 1]
    elif ca == cb and abs(ra - rb) == 1:
        y = max(ra, rb) * p
        x0, x1 = ca * p, (ca + 1) * p
        band = edge_map[max(y - 1, 0):y + 1, x0:x1]
    else:
        y = max(ra, rb) * p
        x = max(ca, cb) * p
        band = edge_map[max(y - 1, 0):y + 1, max(x - 1, 0):x + 1]
    return float(band.mean()) if band.size else 1.0


def patch_merge_model_only(sim_model: np.ndarray, meta: PatchMeta, cfg: PMESConfig, edge_map: np.ndarray) -> np.ndarray:
    uf = UnionFind(meta.N)
    for i, j in enumerate_patch_neighbors(meta, cfg.neigh_mode):
        if sim_model[i, j] < cfg.patch_thresh or sim_model[j, i] < cfg.patch_thresh:
            continue
        if mean_edge_on_border(edge_map, i, j, meta) > 0.35:
            continue
        uf.union(i, j)
    uf.compress()
    _, inv = np.unique(uf.parent, return_inverse=True)
    return inv.astype(np.int32)


def patch_labels_to_pixel(labels_flat: np.ndarray, meta: PatchMeta) -> np.ndarray:
    out = np.zeros((meta.H, meta.W), dtype=np.int32)
    lab2d = labels_flat.reshape(meta.pr, meta.pc)
    p = meta.P
    for r in range(meta.pr):
        for c in range(meta.pc):
            y0, y1 = r * p, (r + 1) * p
            x0, x1 = c * p, (c + 1) * p
            out[y0:y1, x0:x1] = lab2d[r, c]
    return out


def build_structure_edge_map(gray01: np.ndarray, cfg: PMESConfig) -> np.ndarray:
    g = gray01
    if _HAVE_CV2:
        g_blur = cv2.GaussianBlur(g, (0, 0), sigmaX=cfg.edge_sigma, sigmaY=cfg.edge_sigma, borderType=cv2.BORDER_REFLECT101)
        gx = cv2.Sobel(g_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g_blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
    elif _HAVE_SKIMAGE:
        g_blur = filters.gaussian(g, sigma=cfg.edge_sigma, truncate=3.0, preserve_range=True)
        grad = filters.sobel(g_blur)
    else:
        grad_y, grad_x = np.gradient(g)
        grad = np.abs(grad_y) + np.abs(grad_x)
    edge = 0.5 * grad + 0.5 * (1.0 - g)
    edge = np.power(edge, max(cfg.edge_gamma, 1e-6)).astype(np.float32)
    mn, mx = edge.min(), edge.max()
    return (edge - mn) / (mx - mn) if mx > mn else np.zeros_like(edge, dtype=np.float32)


def _label_to_boundary(lab: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(lab, dtype=bool)
    boundary[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    boundary[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return boundary


def edgesnap_watershed(edge: np.ndarray, patch_map: np.ndarray, cfg: PMESConfig, seed_margin: Optional[int] = None) -> np.ndarray:
    if not _HAVE_SKIMAGE or not _HAVE_SCIPY:
        raise RuntimeError("scikit-image and scipy are required for watershed refinement.")
    if seed_margin is None:
        seed_margin = max(1, cfg.patch_size // 2)
    boundary = _label_to_boundary(patch_map)
    dist = distance_transform_edt(~boundary)
    core = dist > seed_margin
    markers = np.zeros_like(patch_map, dtype=np.int32)
    markers[core] = patch_map[core] + 1
    for rid in np.unique(patch_map):
        mask = patch_map == rid
        if not np.any(core & mask):
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            idx = np.argmax(dist[mask])
            markers[ys[idx], xs[idx]] = rid + 1
    seg = segmentation.watershed(
        image=edge,
        markers=markers,
        connectivity=1 if cfg.neigh_mode == "4" else 2,
        watershed_line=cfg.watershed_line,
    )
    _, inv = np.unique(seg, return_inverse=True)
    return inv.reshape(seg.shape).astype(np.int32)


def build_region_adjacency_with_edge(lab: np.ndarray, edge_map: np.ndarray, neigh_mode: str = "4", band_w: int = 2) -> Dict[int, Dict[int, Tuple[int, float]]]:
    h, w = lab.shape
    adjacency: Dict[int, Dict[int, Tuple[int, float]]] = {}

    def accumulate(a: int, b: int, y0: int, y1: int, x0: int, x1: int) -> None:
        if a == b:
            return
        y0, y1 = max(y0, 0), min(y1, h)
        x0, x1 = max(x0, 0), min(x1, w)
        if y1 <= y0 or x1 <= x0:
            return
        band = edge_map[y0:y1, x0:x1]
        if band.size == 0:
            return
        e = float(band.mean())
        for u, v in ((a, b), (b, a)):
            adjacency.setdefault(u, {})
            length_old, edge_old = adjacency[u].get(v, (0, 0.0))
            length_new = length_old + band.size
            edge_new = (edge_old * length_old + e * band.size) / length_new
            adjacency[u][v] = (length_new, edge_new)

    diff = lab[:, :-1] != lab[:, 1:]
    ys, xs = np.where(diff)
    for y, x in zip(ys, xs):
        a, b = int(lab[y, x]), int(lab[y, x + 1])
        x_mid = x + 1
        accumulate(a, b, y - band_w // 2, y + 1 + band_w // 2, x_mid - band_w, x_mid + band_w)

    diff = lab[:-1, :] != lab[1:, :]
    ys, xs = np.where(diff)
    for y, x in zip(ys, xs):
        a, b = int(lab[y, x]), int(lab[y + 1, x])
        y_mid = y + 1
        accumulate(a, b, y_mid - band_w, y_mid + band_w, x - band_w // 2, x + 1 + band_w // 2)

    if neigh_mode == "8":
        diff = lab[:-1, :-1] != lab[1:, 1:]
        ys, xs = np.where(diff)
        for y, x in zip(ys, xs):
            a, b = int(lab[y, x]), int(lab[y + 1, x + 1])
            m, n = y + 1, x + 1
            accumulate(a, b, m - band_w, m + band_w, n - band_w, n + band_w)
        diff = lab[:-1, 1:] != lab[1:, :-1]
        ys, xs = np.where(diff)
        for y, x in zip(ys, xs):
            a, b = int(lab[y, x + 1]), int(lab[y + 1, x])
            m, n = y + 1, x + 1
            accumulate(a, b, m - band_w, m + band_w, n - band_w, n + band_w)

    return adjacency


def merge_small_regions_edgeaware(label_img: np.ndarray, edge_map: np.ndarray, min_px: int, neigh_mode: str = "4", edge_gate_perc: float = 85.0, alpha: float = 1.0, max_pass: int = 3) -> np.ndarray:
    lab = label_img.copy()
    for _ in range(max_pass):
        ids, counts = np.unique(lab, return_counts=True)
        small_ids = [int(i) for i, c in zip(ids, counts) if c < min_px]
        if not small_ids:
            break
        adjacency = build_region_adjacency_with_edge(lab, edge_map, neigh_mode)
        all_means = [e for dest in adjacency.values() for (_, e) in dest.values()] if adjacency else [0.0]
        gate = float(np.percentile(all_means, edge_gate_perc)) if all_means else 0.0
        changed = False
        for sid in small_ids:
            if sid not in adjacency or not adjacency[sid]:
                continue
            candidates = []
            for nid, (length, e_mean) in adjacency[sid].items():
                score = length * ((1.0 - e_mean) ** alpha)
                if e_mean > gate:
                    score *= 0.1
                candidates.append((score, nid))
            if not candidates:
                continue
            _, best = max(candidates, key=lambda item: item[0])
            lab[lab == sid] = best
            changed = True
        if not changed:
            break
        lab = relabel_contiguous(lab)
    return lab


def relabel_contiguous(lab: np.ndarray) -> np.ndarray:
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape).astype(np.int32)


def patch_to_region_map(seg_lab: np.ndarray, meta: PatchMeta) -> np.ndarray:
    patch2reg = np.zeros(meta.N, dtype=np.int32)
    for pid in range(meta.N):
        ys, xs = meta.patch_slice(pid)
        sub = seg_lab[ys, xs]
        vals, cnts = np.unique(sub, return_counts=True)
        patch2reg[pid] = int(vals[np.argmax(cnts)])
    return patch2reg


def compute_region_similarity(cross_probs: np.ndarray, ref_p2r: np.ndarray, tgt_p2r: np.ndarray) -> np.ndarray:
    r1 = int(ref_p2r.max()) + 1
    r2 = int(tgt_p2r.max()) + 1
    sim_sum = np.zeros((r1, r2), dtype=np.float32)
    cnt_sum = np.zeros((r1, r2), dtype=np.int32)
    for p_idx, r_id in enumerate(ref_p2r):
        for q_idx, t_id in enumerate(tgt_p2r):
            sim_sum[r_id, t_id] += cross_probs[p_idx, q_idx]
            cnt_sum[r_id, t_id] += 1
    return sim_sum / np.maximum(cnt_sum, 1)


def build_color_groups_for_matches(pred_matches: List[Tuple[int, int]]) -> List[Dict[str, set]]:
    class UF:
        def __init__(self) -> None:
            self.parent: Dict[str, str] = {}

        def find(self, x: str) -> str:
            if x not in self.parent:
                self.parent[x] = x
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, a: str, b: str) -> None:
            pa, pb = self.find(a), self.find(b)
            if pa != pb:
                self.parent[pa] = pb

    uf = UF()
    for rid, tid in pred_matches:
        uf.union(f"ref_{rid}", f"tgt_{tid}")

    groups: Dict[str, Dict[str, set]] = {}
    for rid, tid in pred_matches:
        root = uf.find(f"ref_{rid}")
        groups.setdefault(root, {"ref": set(), "tgt": set()})
        groups[root]["ref"].add(rid)
        groups[root]["tgt"].add(tid)
    return list(groups.values())


def assign_colors_to_groups(color_groups: List[Dict[str, set]], palette_match: np.ndarray) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]:
    pred_id2col_ref: Dict[int, Tuple[int, int, int]] = {}
    pred_id2col_tgt: Dict[int, Tuple[int, int, int]] = {}
    pal_len = palette_match.shape[0]
    if pal_len == 0:
        return pred_id2col_ref, pred_id2col_tgt
    for idx, group in enumerate(color_groups):
        col = tuple(int(x) for x in palette_match[(idx + 1) % pal_len])
        for rid in group["ref"]:
            pred_id2col_ref[int(rid)] = col
        for tid in group["tgt"]:
            pred_id2col_tgt[int(tid)] = col
    return pred_id2col_ref, pred_id2col_tgt


def colorize_label(lab: np.ndarray, palette: Optional[np.ndarray] = None, seed: int = 0, bg_label: Optional[int] = None, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    ids = np.unique(lab)
    fg_ids = [int(i) for i in ids if bg_label is None or i != bg_label]
    if palette is None or palette.shape[0] < len(fg_ids) + 1:
        palette = make_palette(len(fg_ids) + 1, seed=seed)
    id2col: Dict[int, Tuple[int, int, int]] = {}
    if bg_label is not None and bg_label in ids:
        id2col[int(bg_label)] = bg_color
    for idx, region_id in enumerate(fg_ids):
        color = tuple(int(x) for x in palette[(idx + 1) % palette.shape[0]])
        id2col[region_id] = color
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    for region_id, color in id2col.items():
        rgb[lab == region_id] = color
    return rgb


def colorize_label_with_mapping(lab: np.ndarray, id2col: Dict[int, Tuple[int, int, int]], seed: int = 0, bg_label: Optional[int] = None, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    invalid_color = (160, 160, 160)
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    for idx in np.unique(lab):
        if bg_label is not None and idx == bg_label:
            color = bg_color
        elif idx in id2col:
            color = id2col[int(idx)]
        elif idx == -1:
            color = invalid_color
        else:
            color = invalid_color
        rgb[lab == idx] = color
    return rgb


def overlay_boundaries(base_rgb: np.ndarray, lab: np.ndarray, color_line: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    out = base_rgb.copy()
    out[_label_to_boundary(lab)] = color_line
    return out


def save_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def save_label_map(path: str, lab: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(lab.astype(np.uint16)).save(path)


def region_seg_single(img_rgb: np.ndarray, img_gray: np.ndarray, sim_model: np.ndarray, meta: PatchMeta, cfg: PMESConfig, out_dir: str, prefix: str, palette_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    edge = build_structure_edge_map(img_gray, cfg)
    labels_flat = patch_merge_model_only(sim_model, meta, cfg, edge)
    patch_map = patch_labels_to_pixel(labels_flat, meta)
    seg_snap = edgesnap_watershed(edge, patch_map, cfg)
    if cfg.min_region_ratio > 0:
        min_px_local = max(1, int(round(cfg.min_region_ratio * img_gray.size)))
        seg_clean = merge_small_regions_edgeaware(seg_snap, edge_map=edge, min_px=min_px_local, neigh_mode=cfg.neigh_mode)
    else:
        seg_clean = seg_snap
    if cfg.save_intermediates:
        os.makedirs(out_dir, exist_ok=True)
        color = colorize_label(seg_clean, seed=palette_seed, bg_label=cfg.bg_label, bg_color=(255, 255, 255))
        overlay = (0.5 * img_rgb.astype(np.float32) + 0.5 * color.astype(np.float32)).astype(np.uint8)
        save_image(os.path.join(out_dir, f"{prefix}_seg_color.png"), color)
        save_image(os.path.join(out_dir, f"{prefix}_seg_overlay.png"), overlay)
        save_image(os.path.join(out_dir, f"{prefix}_seg_boundary.png"), overlay_boundaries(img_rgb, seg_clean))
        np.save(os.path.join(out_dir, f"{prefix}_seg.npy"), seg_clean.astype(np.int32))
        save_label_map(os.path.join(out_dir, f"{prefix}_seg.png"), seg_clean)
    return seg_clean, edge


def create_match_visualization(ref_img: np.ndarray, tgt_img: np.ndarray, ref_seg: np.ndarray, tgt_seg: np.ndarray, matches: List[Tuple[int, int]], cfg: PMESConfig) -> np.ndarray:
    palette_match = make_palette(max(len(matches) + 1, 2), seed=cfg.seed + 99)
    color_groups = build_color_groups_for_matches(matches)
    pred_id2col_ref, pred_id2col_tgt = assign_colors_to_groups(color_groups, palette_match)
    ref_match_color = colorize_label_with_mapping(ref_seg, pred_id2col_ref, seed=cfg.seed, bg_label=cfg.bg_label, bg_color=(255, 255, 255))
    tgt_match_color = colorize_label_with_mapping(tgt_seg, pred_id2col_tgt, seed=cfg.seed + 1, bg_label=cfg.bg_label, bg_color=(255, 255, 255))
    ref_overlay = (0.5 * ref_img.astype(np.float32) + 0.5 * ref_match_color.astype(np.float32)).astype(np.uint8)
    tgt_overlay = (0.5 * tgt_img.astype(np.float32) + 0.5 * tgt_match_color.astype(np.float32)).astype(np.uint8)
    gap = np.full((ref_overlay.shape[0], 10, 3), 255, dtype=np.uint8)
    return np.concatenate([ref_overlay, gap, tgt_overlay], axis=1)


def process_pair(ref_img_path: str, tgt_img_path: str, model_path: str, cfg: PMESConfig) -> Dict[str, Any]:
    if not os.path.exists(ref_img_path) or not os.path.exists(tgt_img_path):
        raise FileNotFoundError("Reference or target image path does not exist")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    ref_rgb = _resize_image(_pil_open_rgb(ref_img_path), cfg.resize_hw)
    tgt_rgb = _resize_image(_pil_open_rgb(tgt_img_path), cfg.resize_hw)
    ref_gray = rgb2gray01(ref_rgb)
    tgt_gray = rgb2gray01(tgt_rgb)
    meta = PatchMeta.from_hwP(cfg.resize_hw, cfg.patch_size)
    model = load_model(model_path, meta, cfg.device, patch_size=cfg.patch_size)
    probs = get_model_probs(model, ref_rgb, tgt_rgb, cfg)
    ref_ref_probs, tgt_tgt_probs = slice_intra_blocks(probs, meta)
    ref_seg, ref_edge = region_seg_single(ref_rgb, ref_gray, ref_ref_probs, meta, cfg, os.path.join(cfg.out_dir, "ref"), "ref", palette_seed=cfg.seed)
    tgt_seg, tgt_edge = region_seg_single(tgt_rgb, tgt_gray, tgt_tgt_probs, meta, cfg, os.path.join(cfg.out_dir, "tgt"), "tgt", palette_seed=cfg.seed + 1)
    cross_probs = probs[0, :meta.N, meta.N:].numpy()
    ref_p2r = patch_to_region_map(ref_seg, meta)
    tgt_p2r = patch_to_region_map(tgt_seg, meta)
    sim_reg = compute_region_similarity(cross_probs, ref_p2r, tgt_p2r)
    matches_with_scores: List[Tuple[int, int, float]] = []
    for rid in range(sim_reg.shape[0]):
        if cfg.bg_label is not None and rid == cfg.bg_label:
            continue
        tid = int(np.argmax(sim_reg[rid]))
        score = float(sim_reg[rid, tid])
        if score >= cfg.match_thresh:
            matches_with_scores.append((rid, tid, score))
    match_pairs = [(rid, tid) for rid, tid, _ in matches_with_scores]
    match_vis = create_match_visualization(ref_rgb, tgt_rgb, ref_seg, tgt_seg, match_pairs, cfg)
    save_image(os.path.join(cfg.out_dir, "match_vis.png"), match_vis)
    match_records = [
        {"ref_region": int(rid), "tgt_region": int(tid), "score": float(score)}
        for rid, tid, score in matches_with_scores
    ]
    with open(os.path.join(cfg.out_dir, "matches.json"), "w", encoding="utf-8") as f:
        json.dump({"matches": match_records}, f, indent=2)
    return {
        "matches": match_records,
        "ref_seg_path": os.path.join(cfg.out_dir, "ref", "ref_seg.png"),
        "tgt_seg_path": os.path.join(cfg.out_dir, "tgt", "tgt_seg.png"),
        "num_matches": len(match_records),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Region Segmentation + Matching (no GT evaluation)")
    parser.add_argument("--ref-img", type=str, required=True, help="Path to the reference line-art image")
    parser.add_argument("--tgt-img", type=str, required=True, help="Path to the target line-art image")
    parser.add_argument("--model-path", type=str, default='models/lineart_transformer_200k_p32_best.pth', help="Model checkpoint path")
    parser.add_argument("--out-dir", type=str, default="results_custom", help="Directory to save results")
    parser.add_argument("--resize-h", type=int, default=512, help="Resize height for inference")
    parser.add_argument("--resize-w", type=int, default=512, help="Resize width for inference")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size used by the model")
    parser.add_argument("--neigh-mode", type=str, choices=["4", "8"], default="8", help="Neighbor mode for patch merging")
    parser.add_argument("--patch-thresh", type=float, default=0.72, help="Patch similarity threshold")
    parser.add_argument("--edge-sigma", type=float, default=1.8, help="Edge detection Gaussian sigma")
    parser.add_argument("--edge-gamma", type=float, default=1.2, help="Edge enhancement gamma")
    parser.add_argument("--watershed-line", action="store_true", help="Keep watershed lines instead of merging")
    parser.add_argument("--min-region-ratio", type=float, default=0.005, help="Min region pixel ratio for merging")
    parser.add_argument("--match-thresh", type=float, default=0.6, help="Similarity threshold for region matching")
    parser.add_argument("--bg-label", type=int, default=None, help="Optional background label id")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--no-save-intermediates", action="store_true", help="Skip saving intermediate segmentation files")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for color palette")
    return parser


def cfg_from_args(args: argparse.Namespace) -> PMESConfig:
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    return PMESConfig(
        resize_hw=(args.resize_h, args.resize_w),
        patch_size=args.patch_size,
        bg_label=args.bg_label,
        neigh_mode=args.neigh_mode,
        patch_thresh=args.patch_thresh,
        edge_sigma=args.edge_sigma,
        edge_gamma=args.edge_gamma,
        watershed_line=args.watershed_line,
        min_region_ratio=args.min_region_ratio,
        out_dir=args.out_dir,
        device=device,
        debug=args.debug,
        save_intermediates=not args.no_save_intermediates,
        seed=args.seed,
        match_thresh=args.match_thresh,
    )


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)
    if not _HAVE_SKIMAGE or not _HAVE_SCIPY:
        raise RuntimeError("scikit-image and scipy are required for this script without ground truth.")
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.debug:
        print("=== Custom Dataset Inference (no evaluation) ===")
        for key, value in cfg.to_dict().items():
            print(f"{key}: {value}")
    try:
        result = process_pair(args.ref_img, args.tgt_img, args.model_path, cfg)
        print(f"Finished. Saved outputs to {cfg.out_dir}. Matched {result['num_matches']} region pairs.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print(f"Error during processing: {exc}")
        if cfg.debug:
            raise


if __name__ == "__main__":
    main()
