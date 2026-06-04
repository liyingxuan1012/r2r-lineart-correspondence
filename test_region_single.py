import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
from PIL import Image
import pandas as pd
from collections import defaultdict

import torch
import torchvision.transforms as T


# 在文件开头添加数据集导入
from data import LineArtDataset, PBCLineArtDataset
from torch.utils.data import DataLoader, random_split


# ----------------- optional deps -----------------
try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

try:
    from skimage import filters, segmentation
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

try:
    from scipy.ndimage import (
        binary_dilation, generate_binary_structure,
        distance_transform_edt
    )
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    from sklearn.metrics import adjusted_rand_score
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

# model import
try:
    from model import LineArtTransformerModel
except ImportError as e:
    LineArtTransformerModel = None
    _MODEL_IMPORT_ERROR = e


# ================= Config =================
@dataclass
class PMESConfig:
    resize_hw: Tuple[int, int] = (512, 896)
    patch_size: int = 16
    bg_label: int = 0
    neigh_mode: str = '8'  # 改为8
    patch_thresh: float = 0.727  # 更新默认值

    edge_sigma: float = 2.0  # 更新默认值
    edge_gamma: float = 0.9  # 更新默认值
    edge_percentile: float = 80.0
    watershed_line: bool = False

    # ---- ONLY ratio ----
    min_region_ratio: float = 0.006  # small-region px = ratio * #FG

    out_dir: str = "region_seg_pm_es_out"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    debug: bool = False
    save_intermediates: bool = True
    seed: int = 42

    # 新增：区域匹配阈值
    match_thresh: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============== basic utils ==============
def make_palette(n: int, seed: int = 0, bg_color=(200, 200, 200)) -> np.ndarray:
    # 预定义一些鲜艳且容易区分的颜色
    predefined_colors = [
        (255, 0, 0),     # 红色
        (0, 255, 0),     # 绿色
        (0, 0, 255),     # 蓝色
        (255, 255, 0),   # 黄色
        (255, 0, 255),   # 紫色
        (0, 255, 255),   # 青色
        (255, 128, 0),   # 橙色
        (128, 0, 255),   # 紫罗兰
        (255, 192, 203), # 粉色
        (128, 255, 0),   # 青绿色
        (255, 0, 128),   # 玫红色
        (0, 128, 255),   # 天蓝色
        (128, 128, 0),   # 橄榄色
        (128, 0, 128),   # 紫色
        (0, 128, 128),   # 深青色
    ]
    
    cols = np.zeros((n, 3), dtype=np.uint8)
    
    # 先设置bg_color（如果需要）
    if n > 0:
        cols[0] = bg_color
    
    # 然后使用预定义颜色（从索引1开始，跳过bg_color）
    start_idx = 1 if n > 0 else 0
    for i in range(start_idx, min(n, len(predefined_colors) + start_idx)):
        cols[i] = predefined_colors[i - start_idx]
    
    # 如果需要更多颜色，随机生成但避免太亮或太暗
    if n > len(predefined_colors) + start_idx:
        rng = np.random.default_rng(seed)
        for i in range(len(predefined_colors) + start_idx, n):
            # 生成饱和度高的颜色，避免灰白色
            while True:
                color = rng.integers(0, 256, size=3, dtype=np.uint8)
                # 避免太亮（接近白色）或太暗（接近黑色）或接近灰色
                brightness = color.sum()
                if 150 < brightness < 600 and max(color) - min(color) > 100:
                    cols[i] = color
                    break
    
    return cols

def _pil_open_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

def _pil_open_int(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('I'), dtype=np.int32)

def _resize_image(img: np.ndarray, size_hw: Tuple[int, int], is_label=False) -> np.ndarray:
    H, W = size_hw
    pil_img = Image.fromarray(img)
    interp = Image.NEAREST if is_label else Image.BILINEAR
    arr = np.array(pil_img.resize((W, H), interp))
    return arr.astype(np.int32) if is_label else arr

def load_image_and_label(img_path: str, gt_path: str, cfg: PMESConfig):
    rgb = _resize_image(_pil_open_rgb(img_path), cfg.resize_hw, False)
    lbl = _resize_image(_pil_open_int(gt_path),  cfg.resize_hw, True)
    gray = rgb2gray01(rgb)
    return rgb, gray, lbl

def rgb2gray01(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        g = rgb.astype(np.float32)
    return g.astype(np.float32) / 255.0

def _to_square512(img: np.ndarray) -> np.ndarray:
    """把任意 H×W×3 的 uint8 图像缩放成 512×512"""
    try:
        import cv2
        return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        return np.array(Image.fromarray(img).resize((512, 512), Image.BILINEAR))


# ============== Patch meta ==============
@dataclass
class PatchMeta:
    H: int; W: int; P: int; pr: int; pc: int; N: int
    @classmethod
    def from_hwP(cls, hw: Tuple[int, int], P: int):
        H, W = hw; assert H % P == 0 and W % P == 0
        pr, pc = H // P, W // P
        return cls(H, W, P, pr, pc, pr * pc)
    def pid(self, r, c): return r * self.pc + c
    def rc(self, pid):   return pid // self.pc, pid % self.pc
    def patch_slice(self, pid):
        r, c = self.rc(pid); P = self.P
        return slice(r*P, (r+1)*P), slice(c*P, (c+1)*P)

def enumerate_patch_neighbors(meta: PatchMeta, mode='4'):
    neigh = []
    for r in range(meta.pr):
        for c in range(meta.pc):
            i = meta.pid(r, c)
            if c + 1 < meta.pc: neigh.append((i, meta.pid(r, c + 1)))
            if r + 1 < meta.pr: neigh.append((i, meta.pid(r + 1, c)))
            if mode == '8':
                if r + 1 < meta.pr and c + 1 < meta.pc:
                    neigh.append((i, meta.pid(r + 1, c + 1)))
                if r + 1 < meta.pr and c - 1 >= 0:
                    neigh.append((i, meta.pid(r + 1, c - 1)))
    return neigh


# ============== model forward ==============
def load_model(model_path: str, meta: PatchMeta, device='cuda', patch_size: int = 16):
    if LineArtTransformerModel is None:
        raise ImportError(f"Import error: {_MODEL_IMPORT_ERROR}")
    model = LineArtTransformerModel(
        embed_dim=768, num_heads=12, num_layers=4, 
        num_patches=meta.N, patch_size=patch_size
    ).to(device)
    raw_state = torch.load(model_path, map_location=device)
    new_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    
    # 跳过位置编码参数，让模型自己根据实际尺寸初始化
    # 这样可以支持不同 patch_size 的模型权重加载
    keys_to_skip = ['pos_embed', 'encoder.pos_embedding', '_orig_pos_embed']
    filtered_state = {k: v for k, v in new_state.items() if k not in keys_to_skip}
    
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model

def preprocess_for_model(rgb: np.ndarray, cfg: PMESConfig):
    pil = Image.fromarray(rgb)
    t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    return t(pil).unsqueeze(0)

def get_model_probs(model, ref_rgb, tgt_rgb, cfg):
    ref_t = preprocess_for_model(ref_rgb, cfg).to(cfg.device)
    tgt_t = preprocess_for_model(tgt_rgb, cfg).to(cfg.device)
    with torch.no_grad():
        logits = model(ref_t, tgt_t)
    return torch.sigmoid(logits).cpu()

def slice_intra_blocks(prob_mat: torch.Tensor, meta: PatchMeta):
    N = meta.N; S = prob_mat.shape[1]; assert S == 2*N
    ref_ref = prob_mat[0, :N, :N].numpy()
    tgt_tgt = prob_mat[0, N:, N:].numpy()
    return ref_ref, tgt_tgt


# ============== patch merge ==============
class UnionFind:
    __slots__ = ('parent', 'size')
    def __init__(self, n): self.parent=np.arange(n,dtype=np.int32); self.size=np.ones(n,np.int32)
    def find(self,x):
        p=self.parent
        while p[x]!=x:
            p[x]=p[p[x]]; x=p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.size[ra]<self.size[rb]: ra,rb=rb,ra
        self.parent[rb]=ra; self.size[ra]+=self.size[rb]
    def compress(self):
        for i in range(self.parent.shape[0]): self.parent[i]=self.find(i)

def mean_edge_on_border(edge_map, pid_a, pid_b, meta: PatchMeta):
    ra, ca = meta.rc(pid_a); rb, cb = meta.rc(pid_b); P = meta.P
    if ra == rb and abs(ca-cb)==1:
        x = max(ca, cb) * P
        y0,y1 = ra*P, (ra+1)*P
        band = edge_map[y0:y1, max(x-1,0):x+1]
    elif ca == cb and abs(ra-rb)==1:
        y = max(ra, rb) * P
        x0,x1 = ca*P, (ca+1)*P
        band = edge_map[max(y-1,0):y+1, x0:x1]
    else:
        y = max(ra, rb) * P; x = max(ca, cb) * P
        band = edge_map[max(y-1,0):y+1, max(x-1,0):x+1]
    return float(band.mean()) if band.size else 1.0

def patch_merge_model_only(sim_model, meta: PatchMeta, cfg: PMESConfig, edge_map):
    N = meta.N; uf = UnionFind(N)
    th = cfg.patch_thresh
    for i, j in enumerate_patch_neighbors(meta, cfg.neigh_mode):
        if sim_model[i, j] < th or sim_model[j, i] < th: continue
        if mean_edge_on_border(edge_map, i, j, meta) > 0.35: continue
        uf.union(i, j)
    uf.compress()
    _, inv = np.unique(uf.parent, return_inverse=True)
    return inv.astype(np.int32)

def patch_labels_to_pixel(labels_flat, meta: PatchMeta):
    out = np.zeros((meta.H, meta.W), np.int32)
    lab2d = labels_flat.reshape(meta.pr, meta.pc)
    P = meta.P
    for r in range(meta.pr):
        for c in range(meta.pc):
            y0,y1 = r*P,(r+1)*P; x0,x1=c*P,(c+1)*P
            out[y0:y1, x0:x1] = lab2d[r,c]
    return out


# ============== edge map & watershed ==============
def build_structure_edge_map(gray01, cfg: PMESConfig):
    g = gray01
    if _HAVE_CV2:
        g_blur = cv2.GaussianBlur(g, (0,0), sigmaX=cfg.edge_sigma, sigmaY=cfg.edge_sigma,
                                  borderType=cv2.BORDER_REFLECT101)
        gx = cv2.Sobel(g_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g_blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx*gx + gy*gy)
    elif _HAVE_SKIMAGE:
        g_blur = filters.gaussian(g, sigma=cfg.edge_sigma, truncate=3.0, preserve_range=True)
        grad = filters.sobel(g_blur)
    else:
        g_blur = g
        grad = np.abs(np.gradient(g_blur, axis=0)) + np.abs(np.gradient(g_blur, axis=1))

    edge = 0.5*grad + 0.5*(1.0-g_blur)
    edge = np.power(edge, max(cfg.edge_gamma,1e-6)).astype(np.float32)
    mn,mx=edge.min(),edge.max()
    edge = (edge-mn)/(mx-mn) if mx>mn else np.zeros_like(edge,dtype=np.float32)
    return edge

def _label_to_boundary(lab):
    b = np.zeros_like(lab, dtype=bool)
    b[:, :-1] |= (lab[:, :-1] != lab[:, 1:])
    b[:-1, :] |= (lab[:-1, :] != lab[1:, :])
    return b

def edgesnap_watershed(edge, patch_map, cfg: PMESConfig, seed_margin=None):
    if not _HAVE_SKIMAGE:
        raise ImportError("skimage required.")
    if seed_margin is None:
        seed_margin = cfg.patch_size // 2
    b = _label_to_boundary(patch_map)
    dist = distance_transform_edt(~b)
    core = dist > seed_margin
    markers = np.zeros_like(patch_map, np.int32)
    markers[core] = patch_map[core] + 1
    ids = np.unique(patch_map)
    for rid in ids:
        mask_r = (patch_map == rid)
        if not np.any(core & mask_r):
            idx = np.argmax(dist[mask_r])
            ys, xs = np.where(mask_r)
            y, x = ys[idx], xs[idx]
            markers[y, x] = rid + 1
    seg = segmentation.watershed(
        image=edge, markers=markers,
        connectivity=1 if cfg.neigh_mode=='4' else 2,
        watershed_line=cfg.watershed_line, mask=None, compactness=0.0
    )
    _, inv = np.unique(seg, return_inverse=True)
    return inv.reshape(seg.shape).astype(np.int32)


# ============== small region merge (edge-aware) ==============
def build_region_adjacency_with_edge(lab, edge_map, neigh_mode='4', band_w=2):
    H,W = lab.shape
    adj: Dict[int, Dict[int, Tuple[int,float]]] = {}

    def _acc(a,b,y0,y1,x0,x1):
        if a==b: return
        y0,y1 = max(y0,0), min(y1,H)
        x0,x1 = max(x0,0), min(x1,W)
        if y1<=y0 or x1<=x0: return
        band = edge_map[y0:y1, x0:x1]
        L = band.size
        if L==0: return
        e = float(band.mean())

        for u,v in ((a,b),(b,a)):
            if u not in adj: adj[u]={}
            if v not in adj[u]:
                adj[u][v]=(0,0.0)
            L_old,e_old=adj[u][v]
            L_new=L_old+L
            e_new=(e_old*L_old + e*L)/L_new
            adj[u][v]=(L_new,e_new)

    diff = lab[:, :-1] != lab[:, 1:]
    ys,xs = np.where(diff)
    for y,x in zip(ys,xs):
        a,b = lab[y,x], lab[y,x+1]
        x_mid = x+1
        _acc(a,b, y-(band_w//2), y+1+(band_w//2), x_mid-band_w, x_mid+band_w)

    diff = lab[:-1, :] != lab[1:, :]
    ys,xs = np.where(diff)
    for y,x in zip(ys,xs):
        a,b = lab[y,x], lab[y+1,x]
        y_mid = y+1
        _acc(a,b, y_mid-band_w, y_mid+band_w, x-(band_w//2), x+1+(band_w//2))

    if neigh_mode=='8':
        diff = lab[:-1, :-1] != lab[1:, 1:]
        ys,xs = np.where(diff)
        for y,x in zip(ys,xs):
            a,b=lab[y,x], lab[y+1,x+1]
            m=y+1; n=x+1
            _acc(a,b, m-band_w, m+band_w, n-band_w, n+band_w)
        diff = lab[:-1, 1:] != lab[1:, :-1]
        ys,xs = np.where(diff)
        for y,x in zip(ys,xs):
            a,b=lab[y,x+1], lab[y+1,x]
            m=y+1; n=x+1
            _acc(a,b, m-band_w, m+band_w, n-band_w, n+band_w)

    return adj

def merge_small_regions_edgeaware(label_img, edge_map, min_px,
                                  neigh_mode='4', edge_gate_perc=85.0,
                                  alpha=1.0, max_pass=3):
    lab = label_img.copy()
    for _ in range(max_pass):
        ids, counts = np.unique(lab, return_counts=True)
        small_ids = [i for i,c in zip(ids,counts) if c < min_px]
        if not small_ids: break

        adj = build_region_adjacency_with_edge(lab, edge_map, neigh_mode)
        all_means = [e for d in adj.values() for (_,e) in d.values()] if adj else [0.0]
        gate = np.percentile(all_means, edge_gate_perc)

        changed=False
        for sid in small_ids:
            if sid not in adj or len(adj[sid])==0: continue
            cand=[]
            for nid,(L,e_mean) in adj[sid].items():
                score = L * ((1.0 - e_mean)**alpha)
                if e_mean > gate: score *= 0.1
                cand.append((score,nid))
            if not cand: continue
            score,best_nid = max(cand, key=lambda t:t[0])
            if score<=0:
                best_nid = min(adj[sid].items(), key=lambda kv: kv[1][1])[0]
            lab[lab==sid]=best_nid
            changed=True
        if not changed: break
        lab = relabel_contiguous(lab)
    return lab

def relabel_contiguous(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape).astype(np.int32)


# ============== 区域匹配辅助函数 ==============

def patch_to_region_map(seg_lab: np.ndarray, meta: PatchMeta) -> np.ndarray:
    """根据最终像素级分割结果，把每个 patch 映射到所属的 region id (通过多数像素原则)。"""
    patch2reg = np.zeros(meta.N, dtype=np.int32)
    for pid in range(meta.N):
        ys, xs = meta.patch_slice(pid)
        sub_lab = seg_lab[ys, xs]
        vals, cnts = np.unique(sub_lab, return_counts=True)
        patch2reg[pid] = int(vals[np.argmax(cnts)])
    return patch2reg


def compute_region_similarity(cross_probs: np.ndarray,
                              ref_p2r: np.ndarray,
                              tgt_p2r: np.ndarray) -> np.ndarray:
    """按 patch→region 映射，把 patch 级跨图相似度聚合为区域级平均相似度。"""
    R1 = int(ref_p2r.max()) + 1
    R2 = int(tgt_p2r.max()) + 1
    sim_sum = np.zeros((R1, R2), dtype=np.float32)
    cnt_sum = np.zeros((R1, R2), dtype=np.int32)
    for p_idx, r_id in enumerate(ref_p2r):
        for q_idx, t_id in enumerate(tgt_p2r):
            sim_sum[r_id, t_id] += cross_probs[p_idx, q_idx]
            cnt_sum[r_id, t_id] += 1
    sim_avg = sim_sum / np.maximum(cnt_sum, 1)
    return sim_avg


# ============== 区域匹配定量评价函数 ==============

def get_corresponding_gt_region(pred_id, seg, gt, bg_label, thresh):
    """为预测区域找到对应的GT区域（通过投票+纯度验证）"""
    pred_mask = (seg == pred_id)
    if pred_mask.sum() == 0:
        return None, 0.0
        
    # 投票找最频繁的GT标签
    gt_votes = gt[pred_mask]
    valid_votes = gt_votes[gt_votes != bg_label]
    if len(valid_votes) == 0:
        return None, 0.0
        
    gt_id = np.bincount(valid_votes).argmax()
    
    # 计算纯度
    gt_mask = (gt == gt_id)
    intersection = np.sum(pred_mask & gt_mask)
    purity = intersection / np.sum(pred_mask)
    
    if purity >= thresh:
        return gt_id, purity
    return None, purity


def group_gt_matches(gt_matches):
    """
    将 GT 匹配对按联通关系合并成组：
    每个组包含若干 ref_id 和若干 tgt_id，它们应被视为一个"GT 匹配单元"。
    """
    if not gt_matches:
        return []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # ref / tgt 放不同命名空间
    for rid, tid in gt_matches:
        union(("ref", int(rid)), ("tgt", int(tid)))

    groups = defaultdict(lambda: {"ref": set(), "tgt": set()})
    for rid, tid in gt_matches:
        root = find(("ref", int(rid)))
        groups[root]["ref"].add(int(rid))
        groups[root]["tgt"].add(int(tid))

    return list(groups.values())


def compute_purity_aware_matching_precision_recall(pred_matches, gt_matches, 
                                                   ref_seg, tgt_seg, ref_gt, tgt_gt, 
                                                   bg_label=0, purity_thresh=0.8):
    """
    precision 的定义不变；
    recall 的分母改为：GT 合并后的 group 数量。
    只要某个 group 中的任意(ref_gt_id, tgt_gt_id) 被正确覆盖，就算该 group 被召回。
    """
    # --- 构建 GT group ---
    gt_groups = group_gt_matches(gt_matches)
    # 还保留原始 pair 级，用于 precision
    gt_match_set = set(gt_matches)

    # === 精度 ===
    valid_pred_matches = 0
    correct_pred_matches = 0
    precision_details = []

    for ref_id, tgt_id in pred_matches:
        ref_gt_id, ref_purity = get_corresponding_gt_region(ref_id, ref_seg, ref_gt, bg_label, purity_thresh)
        tgt_gt_id, tgt_purity = get_corresponding_gt_region(tgt_id, tgt_seg, tgt_gt, bg_label, purity_thresh)

        detail = {
            'pred_match': (ref_id, tgt_id),
            'ref_gt_id': ref_gt_id,
            'tgt_gt_id': tgt_gt_id,
            'ref_purity': ref_purity,
            'tgt_purity': tgt_purity,
            'is_valid': False,
            'is_correct': False
        }

        if ref_gt_id is not None and tgt_gt_id is not None:
            detail['is_valid'] = True
            valid_pred_matches += 1
            if (ref_gt_id, tgt_gt_id) in gt_match_set:
                detail['is_correct'] = True
                correct_pred_matches += 1

        precision_details.append(detail)

    precision = correct_pred_matches / valid_pred_matches if valid_pred_matches > 0 else 0.0

    # === 召回（按 group 计） ===
    covered_groups = 0
    recall_details = []

    if gt_groups:
        # 预先把每个预测匹配映射到 GT id（如果有）
        mapped_pred = []
        for ref_id, tgt_id in pred_matches:
            r_gt, r_p = get_corresponding_gt_region(ref_id, ref_seg, ref_gt, bg_label, purity_thresh)
            t_gt, t_p = get_corresponding_gt_region(tgt_id, tgt_seg, tgt_gt, bg_label, purity_thresh)
            mapped_pred.append((r_gt, t_gt, r_p, t_p))

        for g in gt_groups:
            ref_set, tgt_set = g['ref'], g['tgt']
            cov = False
            cover_list = []
            for (r_gt, t_gt, r_p, t_p) in mapped_pred:
                if r_gt in ref_set and t_gt in tgt_set:
                    cov = True
                    cover_list.append((r_gt, t_gt, r_p, t_p))
            if cov:
                covered_groups += 1
            recall_details.append({
                'gt_group_ref': list(ref_set),
                'gt_group_tgt': list(tgt_set),
                'covering_pred_matches': cover_list,
                'is_covered': cov
            })

        recall = covered_groups / len(gt_groups)
    else:
        # 没有 GT 匹配，召回定义为 0 或 1 皆可，这里保持 0
        recall = 0.0

    debug_info = {
        'purity_threshold': purity_thresh,
        'valid_pred_matches': valid_pred_matches,
        'correct_pred_matches': correct_pred_matches,
        'covered_gt_groups': covered_groups,
        'num_gt_groups': len(gt_groups),
        'precision_details': precision_details,
        'recall_details': recall_details
    }
    return precision, recall, debug_info


def compute_bidirectional_matching_metrics(ref_seg, tgt_seg, ref_gt, tgt_gt, 
                                           cross_probs, ref_p2r, tgt_p2r, 
                                           gt_matches, cfg):
    """同之前，只是内部精/召回调用的函数已更新，返回值不变。"""
    # ref->tgt
    ref_to_tgt_matches = []
    sim_reg = compute_region_similarity(cross_probs, ref_p2r, tgt_p2r)
    for rid in range(sim_reg.shape[0]):
        if rid == cfg.bg_label:
            continue
        tid = int(np.argmax(sim_reg[rid]))
        if sim_reg[rid, tid] >= cfg.match_thresh:
            ref_to_tgt_matches.append((rid, tid))

    # tgt->ref
    tgt_to_ref_matches = []
    sim_reg_T = compute_region_similarity(cross_probs.T, tgt_p2r, ref_p2r)
    for tid in range(sim_reg_T.shape[0]):
        if tid == cfg.bg_label:
            continue
        rid = int(np.argmax(sim_reg_T[tid]))
        if sim_reg_T[tid, rid] >= cfg.match_thresh:
            tgt_to_ref_matches.append((rid, tid))

    precision1, recall1, dbg1 = compute_purity_aware_matching_precision_recall(
        ref_to_tgt_matches, gt_matches, ref_seg, tgt_seg, ref_gt, tgt_gt, cfg.bg_label
    )
    precision2, recall2, dbg2 = compute_purity_aware_matching_precision_recall(
        tgt_to_ref_matches, gt_matches, ref_seg, tgt_seg, ref_gt, tgt_gt, cfg.bg_label
    )

    avg_precision = (precision1 + precision2) / 2
    avg_recall    = (recall1 + recall2) / 2

    all_matches = list(set(ref_to_tgt_matches + tgt_to_ref_matches))

    avg_valid_matches = (dbg1['valid_pred_matches'] + dbg2['valid_pred_matches']) / 2

    return avg_precision, avg_recall, all_matches, int(avg_valid_matches)


def filter_valid_matches_for_visualization(matches, ref_seg, tgt_seg, ref_gt, tgt_gt, cfg):
    """过滤出有效匹配用于可视化"""
    valid_matches = []
    
    for rid, tid in matches:
        # 检查纯度
        ref_gt_id, ref_purity = get_corresponding_gt_region(rid, ref_seg, ref_gt, cfg.bg_label, 0.8)
        tgt_gt_id, tgt_purity = get_corresponding_gt_region(tid, tgt_seg, tgt_gt, cfg.bg_label, 0.8)
        
        if ref_gt_id is not None and tgt_gt_id is not None:
            valid_matches.append((rid, tid))
    
    return valid_matches


# ============== viz ==============
def colorize_label(lab, palette=None, seed=0, bg_label=None, bg_color=(0, 0, 0)):
    """
    给标签图着色，背景使用指定颜色
    
    Args:
        lab: 标签图 (H, W)
        palette: 调色板
        seed: 随机种子
        bg_label: 背景标签ID，如果指定则背景使用bg_color
        bg_color: 背景颜色，默认为黑色
    """
    ids = np.unique(lab)
    
    # 分离背景和前景标签
    if bg_label is not None and bg_label in ids:
        fg_ids = [i for i in ids if i != bg_label]
        bg_ids = [bg_label]
    else:
        fg_ids = list(ids)
        bg_ids = []
    
    # 为前景标签生成调色板
    if palette is None or palette.shape[0] < len(fg_ids):
        palette = make_palette(len(fg_ids), seed=seed)
    
    # 创建颜色映射
    id2col = {}
    
    # 背景标签使用指定的背景颜色
    for bg_id in bg_ids:
        id2col[bg_id] = bg_color
    
    # 前景标签使用调色板
    for k, fg_id in enumerate(fg_ids):
        # 跳过索引0（bg_color），从索引1开始取颜色
        color_idx = ((k + 1) % len(palette)) if len(palette) > 1 else 0
        id2col[fg_id] = palette[color_idx]
    
    # 生成彩色图像
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    for i, col in id2col.items():
        rgb[lab == i] = col
    
    return rgb

def overlay_boundaries(base_rgb, lab, color_line=(0,255,0)):
    out = base_rgb.copy()
    b = _label_to_boundary(lab)
    out[b] = color_line
    return out

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


# ============== adaptive min_px from ratio ==============
def compute_min_region_px_ratio(gt_lab: np.ndarray,
                                cfg: PMESConfig) -> int:
    fg_pixels = int((gt_lab != cfg.bg_label).sum())
    min_px = max(1, int(round(cfg.min_region_ratio * fg_pixels)))
    return min_px, fg_pixels


# ============== 数据集相关函数 ==============
def load_pbc_gt_label(line_path, resize_hw, bg_label=1):
    """加载PBC数据集的真实GT标签，参考data.py中PBCLineArtDataset的实现"""
    import imageio.v2 as imageio
    import json
    from PIL import Image
    
    # 从line_path推导seg_path: .../char_dir/line/frame.png -> .../char_dir/seg/frame.png
    char_dir = os.path.dirname(os.path.dirname(line_path))
    frame = os.path.splitext(os.path.basename(line_path))[0]
    seg_path = os.path.join(char_dir, "seg", f"{frame}.png")
    
    # 从line_path推导json_index_path: .../char_dir/line/frame.png -> .../char_dir/json_index/frame.json
    json_index_path = os.path.join(char_dir, "json_index", f"{frame}.json")
    
    # 使用与data.py相同的方法加载
    seg_img = imageio.imread(seg_path)
    if seg_img is None:
        raise RuntimeError(f"Cannot read seg: {seg_path}")
    
    # 使用channel 2 (与data.py一致)
    if seg_img.ndim >= 3:
        seg_id_map = seg_img[:, :, 2]
    else:
        seg_id_map = seg_img
    
    # 使用PIL resize (与data.py一致)
    label_img = Image.fromarray(seg_id_map.astype(np.uint8))
    label_img = label_img.resize(resize_hw[::-1], Image.NEAREST)
    seg_id_map = np.array(label_img)
    
    # 加载JSON索引文件来获取语义标签映射
    if os.path.exists(json_index_path):
        with open(json_index_path, 'r') as f:
            region_to_semantic = json.load(f)
        
        # 创建语义标签到区域ID的映射
        semantic_to_regions = {}
        for region_id_str, (pixel_count, semantic_label) in region_to_semantic.items():
            region_id = int(region_id_str)
            if semantic_label not in semantic_to_regions:
                semantic_to_regions[semantic_label] = []
            semantic_to_regions[semantic_label].append(region_id)
        
        # 找到所有背景区域（语义标签为-1的区域）
        background_regions = semantic_to_regions.get(-1, [])
        
        # 创建新的标签图，将背景区域统一标记为bg_label
        corrected_seg_id_map = seg_id_map.copy()
        for bg_region_id in background_regions:
            corrected_seg_id_map[seg_id_map == bg_region_id] = bg_label
        
        return corrected_seg_id_map.astype(np.int32)
    else:
        # 如果JSON文件不存在，使用原来的方法（向后兼容）
        print(f"Warning: JSON index file not found: {json_index_path}, using fallback method")
        return seg_id_map.astype(np.int32)

def setup_single_pair_from_dataset(is_genai: bool, csv_path: str, lineart_dir: str, labels_dir: str, 
                                   pbc_root: str, patch_size: int, pair_index: int = 0, seed: int = 42):
    """从数据集中获取单对图片数据"""
    
    if is_genai:
        resize_hw = (512, 896)
        transform = T.Compose([
            T.Resize(resize_hw),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        dataset = LineArtDataset(
            csv_path=csv_path,
            root_dir_lineart=lineart_dir,
            root_dir_label=labels_dir,
            transform_image=transform,
            patch_size=patch_size,
            label_img_resize_size=resize_hw
        )
        
        # 获取指定索引的数据
        if pair_index >= len(dataset):
            pair_index = 0
        
        # 直接从数据集获取数据（这里需要特殊处理，因为GenAI数据集返回的是文件路径信息）
        return dataset, resize_hw, 'csv', pair_index
        
    else:  # PBC
        resize_hw = (512, 512)
        transform = T.Compose([
            T.Resize(resize_hw),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        full_dataset = PBCLineArtDataset(
            root_dir=pbc_root,
            transform_image=transform,
            patch_size=patch_size,
            label_img_resize_size=resize_hw
        )
        
        # 使用与batch版本相同的验证集划分
        val_size = 300
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        # 获取指定索引的数据
        if pair_index >= len(val_dataset):
            pair_index = 0
            
        return val_dataset, resize_hw, 'pbc', pair_index


# ============== per-image pipeline ==============
def ensure_background_label(pred_lab, gt_lab, bg_label):
    """
    确保预测标签中的背景区域使用正确的背景标签
    
    Args:
        pred_lab: 预测标签图 (H, W)
        gt_lab: GT标签图 (H, W) 
        bg_label: 正确的背景标签ID
    
    Returns:
        修正后的预测标签图
    """
    pred_corrected = pred_lab.copy()
    
    # 找到GT中的背景区域
    bg_mask = (gt_lab == bg_label)
    
    # 将预测中的背景区域设置为正确的背景标签
    pred_corrected[bg_mask] = bg_label
    
    return pred_corrected


# ============== metrics ==============
@dataclass
class MetricsResult:
    ari: float; miou_pg: float; miou_gp: float;
    count_ratio: float; num_pred_fg: int; num_gt_fg: int

def _extract_masks_in_valid(label_img, valid):
    ids = np.unique(label_img[valid]); out={}
    for i in ids: out[int(i)] = (label_img==i) & valid
    return out

def _iou(a,b):
    inter = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    return float(inter/union) if union>0 else 0.0

def compute_metrics(pred_lab, gt_lab, bg_label):
    valid = gt_lab != bg_label
    if not np.any(valid):
        return MetricsResult(0,0,0,0,0,0)

    if _HAVE_SKLEARN:
        ari = float(adjusted_rand_score(gt_lab[valid].ravel(), pred_lab[valid].ravel()))
    else:
        ari = 0.0

    pred_masks = _extract_masks_in_valid(pred_lab, valid)
    gt_masks   = _extract_masks_in_valid(gt_lab, valid)

    pg = [max((_iou(pm, gm) for gm in gt_masks.values()), default=0.0) for pm in pred_masks.values()]
    gp = [max((_iou(gm, pm) for pm in pred_masks.values()), default=0.0) for gm in gt_masks.values()]
    miou_pg = float(np.mean(pg)) if pg else 0.0
    miou_gp = float(np.mean(gp)) if gp else 0.0

    num_pred_fg = len(pred_masks); num_gt_fg = len(gt_masks)
    cr = float(num_pred_fg / num_gt_fg) if num_gt_fg>0 else 0.0

    return MetricsResult(ari, miou_pg, miou_gp, cr, num_pred_fg, num_gt_fg)


# ============== per-image pipeline ==============
def region_seg_single(img_rgb, img_gray, gt_lab,
                      sim_model, meta, cfg,
                      out_dir, prefix, palette_seed=0) -> MetricsResult:

    edge = build_structure_edge_map(img_gray, cfg)

    labels_flat = patch_merge_model_only(sim_model, meta, cfg, edge)
    patch_map   = patch_labels_to_pixel(labels_flat, meta)

    seg_snap = edgesnap_watershed(edge, patch_map, cfg)

    # adaptive min_px (only ratio)
    min_px_local, FG = compute_min_region_px_ratio(gt_lab, cfg)
    
    seg_clean = merge_small_regions_edgeaware(
        seg_snap, edge_map=edge, min_px=min_px_local,
        neigh_mode=cfg.neigh_mode, edge_gate_perc=85.0, alpha=1.0, max_pass=3
    )

    # 确保背景区域使用正确的标签
    seg_clean = ensure_background_label(seg_clean, gt_lab, cfg.bg_label)

    metrics = compute_metrics(seg_clean, gt_lab, bg_label=cfg.bg_label)

    if cfg.save_intermediates and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_image(os.path.join(out_dir, f"{prefix}_gt_color.png"),
                   colorize_label(gt_lab, seed=palette_seed+100, bg_label=cfg.bg_label, bg_color=(255, 255, 255)))
        save_image(os.path.join(out_dir, f"{prefix}_overlay.png"),
                   overlay_boundaries(img_rgb, seg_clean, (0,255,0)))
        pred_col = colorize_label(seg_clean, seed=palette_seed+300, bg_label=cfg.bg_label, bg_color=(255, 255, 255))
        overlay_pred_col = (0.5*img_rgb.astype(np.float32) + 0.5*pred_col.astype(np.float32)).astype(np.uint8)
        save_image(os.path.join(out_dir, f"{prefix}_orig_pred_color.png"), overlay_pred_col)
        with open(os.path.join(out_dir, f"{prefix}_metrics.txt"), 'w') as f:
            f.write(json.dumps(asdict(metrics), indent=2))

    # 返回指标和最终分割结果，供跨图区域匹配使用
    return metrics, seg_clean


# ============== pair pipeline ==============
def region_seg_pair_from_dataset(is_genai: bool, csv_path: str, lineart_dir: str, labels_dir: str,
                                pbc_root: str, model_path: str, cfg: PMESConfig, 
                                pair_index: int = 0):
    """从数据集中处理单对图片"""
    
    # 设置背景标签（PBC数据集现在通过JSON文件自动处理背景区域）
    if is_genai:
        cfg.bg_label = 0  # GenAI: 背景=0
    else:
        cfg.bg_label = 1  # PBC: 背景=1 (通过JSON文件自动识别背景区域)
    
    dataset, resize_hw, data_mode, actual_index = setup_single_pair_from_dataset(
        is_genai, csv_path, lineart_dir, labels_dir, pbc_root, cfg.patch_size, pair_index, cfg.seed
    )
    
    # 更新配置中的分辨率
    cfg.resize_hw = resize_hw
    
    # 初始化模型
    meta = PatchMeta.from_hwP(cfg.resize_hw, cfg.patch_size)
    model = load_model(model_path, meta, cfg.device, patch_size=cfg.patch_size)
    
    dataset_type = "GENAI" if is_genai else "PBC"
    if cfg.debug:
        print(f"数据集类型: {dataset_type}")
        print(f"图像分辨率: {resize_hw}")
        print(f"处理第 {pair_index + 1} 对图片")
    
    if data_mode == 'pbc':
        # PBC数据集处理
        ref_img, tgt_img, gt_matrix, ref_path, tgt_path, patch_labels_ref, patch_labels_tgt = dataset[actual_index]
        print(ref_path, tgt_path)

        # 将tensor转换为numpy数组
        ref_img_np = ref_img.permute(1, 2, 0).numpy()
        tgt_img_np = tgt_img.permute(1, 2, 0).numpy()
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        ref_img_np = (ref_img_np * std + mean) * 255
        tgt_img_np = (tgt_img_np * std + mean) * 255
        ref_img_np = np.clip(ref_img_np, 0, 255).astype(np.uint8)
        tgt_img_np = np.clip(tgt_img_np, 0, 255).astype(np.uint8)
        
        # 转换为灰度图
        ref_gray = rgb2gray01(ref_img_np)
        tgt_gray = rgb2gray01(tgt_img_np)
        
        # 直接加载真实GT (使用从dataset返回的路径)
        ref_gt = load_pbc_gt_label(ref_path, cfg.resize_hw, cfg.bg_label)
        tgt_gt = load_pbc_gt_label(tgt_path, cfg.resize_hw, cfg.bg_label)
        
        # 获取模型预测
        prob_mat = get_model_probs(model, ref_img_np, tgt_img_np, cfg)
        ref_ref_probs, tgt_tgt_probs = slice_intra_blocks(prob_mat, meta)
        
    else:  # GenAI数据集处理
        # 从CSV文件读取图片对信息
        df = pd.read_csv(csv_path)
        
        if pair_index >= len(df):
            raise IndexError(f"图片对索引 {pair_index} 超出范围，CSV文件只有 {len(df)} 对图片")
        
        row = df.iloc[pair_index]
        
        # 构建文件路径
        ref_img_path = os.path.join(lineart_dir, str(row['dir']), row['reference'])
        tgt_img_path = os.path.join(lineart_dir, str(row['dir']), row['target'])
        ref_gt_path = os.path.join(labels_dir, str(row['dir']), row ['reference'])
        tgt_gt_path = os.path.join(labels_dir, str(row['dir']), row['target'])
        
        # 检查文件是否存在
        for path, name in [(ref_img_path, "参考图片"), (tgt_img_path, "目标图片"),
                          (ref_gt_path, "参考GT"), (tgt_gt_path, "目标GT")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name}不存在: {path}")
        
        # 加载图片和标签
        ref_rgb, ref_gray, ref_gt = load_image_and_label(ref_img_path, ref_gt_path, cfg)
        tgt_rgb, tgt_gray, tgt_gt = load_image_and_label(tgt_img_path, tgt_gt_path, cfg)
        
        # 获取模型预测
        prob_mat = get_model_probs(model, ref_rgb, tgt_rgb, cfg)
        ref_ref_probs, tgt_tgt_probs = slice_intra_blocks(prob_mat, meta)
        
        # 设置numpy数组变量
        ref_img_np = ref_rgb
        tgt_img_np = tgt_rgb
    
    # 进行区域分割
    ref_out = os.path.join(cfg.out_dir, "ref")
    tgt_out = os.path.join(cfg.out_dir, "tgt")
    
    ref_metrics, ref_seg = region_seg_single(ref_img_np, ref_gray, ref_gt,
                                    ref_ref_probs, meta, cfg,
                                    out_dir=ref_out, prefix="ref", palette_seed=0)
    tgt_metrics, tgt_seg = region_seg_single(tgt_img_np, tgt_gray, tgt_gt,
                                    tgt_tgt_probs, meta, cfg,
                                    out_dir=tgt_out, prefix="tgt", palette_seed=1000)
    
    # ========= 区域匹配（方案一：阈值多对一） =========
    cross_probs = prob_mat[0, :meta.N, meta.N:].numpy() if isinstance(prob_mat, torch.Tensor) else prob_mat[0, :meta.N, meta.N:]

    ref_p2r = patch_to_region_map(ref_seg, meta)
    tgt_p2r = patch_to_region_map(tgt_seg, meta)

    sim_reg = compute_region_similarity(cross_probs, ref_p2r, tgt_p2r)

    matches = []
    for rid in range(sim_reg.shape[0]):
        if rid == cfg.bg_label:
            continue  # 跳过背景
        tid = int(np.argmax(sim_reg[rid]))
        if sim_reg[rid, tid] >= cfg.match_thresh:
            matches.append((rid, tid))

    # ========= 加载GT匹配数据 =========
    gt_matches = []
    if is_genai:
        try:
            gt_matches = load_gt_region_matches(csv_path, labels_dir, pair_index)
        except Exception as e:
            if cfg.debug:
                print(f"加载GT匹配失败: {e}")
    else:  # PBC数据集
        try:
            gt_matches = load_pbc_gt_region_matches(ref_path, tgt_path, pbc_root)
        except Exception as e:
            if cfg.debug:
                print(f"加载PBC GT匹配失败: {e}")

    # ========= 跨图匹配评价 =========
    match_precision, match_recall, matches, avg_valid_matches = compute_bidirectional_matching_metrics(
        ref_seg, tgt_seg, ref_gt, tgt_gt, cross_probs, ref_p2r, tgt_p2r, gt_matches, cfg
    )

    # ==== 统计GT group数量 ====
    gt_groups = group_gt_matches(gt_matches)
    gt_group_count = len(gt_groups)

    # ========= 可视化：匹配区域用相同颜色，拼接到同一张图 =========
    
    # 过滤出有效匹配用于可视化
    valid_matches_for_vis = filter_valid_matches_for_visualization(matches, ref_seg, tgt_seg, ref_gt, tgt_gt, cfg)
    
    # # 创建三行组合可视化
    # combined_vis = create_combined_visualization_with_captions(
    #     ref_img_np, tgt_img_np, ref_gt, tgt_gt,
    #     ref_seg, tgt_seg, valid_matches_for_vis, matches, gt_matches, cfg, avg_valid_matches, gt_group_count
    # )
    combined_vis = create_grid_visualization(
        ref_img_np, tgt_img_np,
        ref_gt, tgt_gt,
        ref_seg, tgt_seg,
        valid_matches_for_vis, matches, gt_matches,
        cfg, avg_valid_matches, gt_group_count
    )

    save_image(os.path.join(cfg.out_dir, "match_vis.png"), combined_vis)

    # 保存匹配元数据
    pair_metrics = {
        "ref": asdict(ref_metrics), 
        "tgt": asdict(tgt_metrics), 
        "cross_matching": {
            "precision_purity": match_precision, # 考虑分割纯度的匹配精度
            "recall_coverage": match_recall,      # 考虑分割纯度的匹配召回率
            "num_pred_matches": len(matches),
            "num_valid_matches": avg_valid_matches,
            "num_gt_matches": gt_group_count
        }
    }
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "pair_metrics.json"), 'w') as f:
        json.dump(pair_metrics, f, indent=2)
    
    # 输出结果
    if cfg.debug:
        print(f"\n=== {dataset_type} 数据集单对结果 (详细模式) ===")
        print(f"检测到 {len(matches)} 对区域匹配 (阈值={cfg.match_thresh})")
        if is_genai:
            print(f"加载到 {len(gt_matches)} 对GT匹配")
        print("\n--- 区域分割评价 ---")
        print(f"Ref - ARI: {ref_metrics.ari:.4f}, mIoU_PG: {ref_metrics.miou_pg:.4f}, mIoU_GP: {ref_metrics.miou_gp:.4f}, CR: {ref_metrics.count_ratio:.4f}")
        print(f"Tgt - ARI: {tgt_metrics.ari:.4f}, mIoU_PG: {tgt_metrics.miou_pg:.4f}, mIoU_GP: {tgt_metrics.miou_gp:.4f}, CR: {tgt_metrics.count_ratio:.4f}")
        print("\n--- 跨图匹配评价 ---")
        print(f"Match Precision: {match_precision:.4f}")  # 双向平均精度
        print(f"Match Recall: {match_recall:.4f}")        # 双向平均召回率
        print(f"Matches - Predicted: {len(matches)}, Valid: {avg_valid_matches}, GT: {len(gt_groups)}")

    else:
        print(f"\n=== {dataset_type} 数据集单对结果 ===")
        print("--- 区域分割评价 ---")
        print(f"Ref - ARI: {ref_metrics.ari:.4f}, mIoU_PG: {ref_metrics.miou_pg:.4f}, mIoU_GP: {ref_metrics.miou_gp:.4f}, CR: {ref_metrics.count_ratio:.4f}")
        print(f"Tgt - ARI: {tgt_metrics.ari:.4f}, mIoU_PG: {tgt_metrics.miou_pg:.4f}, mIoU_GP: {tgt_metrics.miou_gp:.4f}, CR: {tgt_metrics.count_ratio:.4f}")
        print("--- 跨图匹配评价 ---")
        print(f"Match Precision: {match_precision:.4f}")  # 双向平均精度
        print(f"Match Recall: {match_recall:.4f}")        # 双向平均召回率
        print(f"Matches - Predicted: {len(matches)}, Valid: {avg_valid_matches}, GT: {len(gt_groups)}")
    
    return pair_metrics


def load_gt_region_matches(csv_path: str, labels_dir: str, pair_index: int):
    """加载GT区域匹配信息（GenAI数据集）"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    row = df.iloc[pair_index]
    dir_name = str(row['dir'])
    region_map_path = os.path.join(labels_dir, dir_name, "region_map.json")
    if not os.path.exists(region_map_path):
        return []
    with open(region_map_path, 'r') as f:
        data = json.load(f)
    ref_name = row['reference']
    tgt_name = row['target']
    for item in data:
        if item['reference'] == ref_name and item['target'] == tgt_name:
            gt_matches = []
            for ref_id_str, match_info in item['region_map'].items():
                ref_id = int(ref_id_str)
                match_region = match_info['match_region']
                if isinstance(match_region, list):
                    for tgt_id in match_region:
                        if tgt_id != -1:
                            gt_matches.append((ref_id, int(tgt_id)))
                else:
                    if match_region != -1:
                        gt_matches.append((ref_id, int(match_region)))
            return gt_matches
    return []


def load_pbc_gt_region_matches(ref_path: str, tgt_path: str, pbc_root: str):
    """加载PBC数据集的GT区域匹配信息"""
    char_dir = os.path.dirname(os.path.dirname(ref_path))
    ref_name = os.path.basename(ref_path)
    tgt_name = os.path.basename(tgt_path)
    region_map_path = os.path.join(char_dir, "seg", "region_map.json")
    if not os.path.exists(region_map_path):
        return []
    with open(region_map_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if item['reference'] == ref_name and item['target'] == tgt_name:
            gt_matches = []
            for ref_id_str, match_info in item['region_map'].items():
                ref_id = int(ref_id_str)
                match_region = match_info['match_region']
                if isinstance(match_region, list):
                    for tgt_id in match_region:
                        if tgt_id != -1:
                            gt_matches.append((ref_id, int(tgt_id)))
                else:
                    if match_region != -1:
                        gt_matches.append((ref_id, int(match_region)))
            return gt_matches
    return []


def add_caption_to_image(img: np.ndarray, caption: str, font_scale: float = 1.0, 
                        thickness: int = 2, color=(0, 0, 0), bg_color=(255, 255, 255),
                        caption_height: int = 50) -> np.ndarray:
    """在图片上方添加caption"""
    try:
        import cv2
        h, w = img.shape[:2]
        
        # 创建caption区域
        caption_img = np.full((caption_height, w, 3), bg_color, dtype=np.uint8)
        
        # 计算文字位置（居中）
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (w - text_w) // 2
        y = (caption_height + text_h) // 2
        
        # 添加文字
        cv2.putText(caption_img, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness, cv2.LINE_AA)
        
        # 拼接caption和原图
        result = np.concatenate([caption_img, img], axis=0)
        return result
    except ImportError:
        # 如果没有cv2，直接返回原图
        return img


def create_combined_visualization_with_captions(ref_img_np, tgt_img_np, ref_gt, tgt_gt, 
                                              ref_seg, tgt_seg, valid_pred_matches, all_pred_matches, gt_matches, 
                                              cfg: PMESConfig, avg_valid_matches: int, gt_group_count: int):
    """创建带caption和空白间隔的三行可视化
    
    Args:
        avg_valid_matches: 双向匹配评价中的平均有效匹配数（考虑纯度验证）
    """
    
    # 生成颜色
    all_matches = list(set(valid_pred_matches + gt_matches))
    palette_match = make_palette(len(all_matches)+10, seed=cfg.seed+99)
    invalid_color = (160, 160, 160)
    
    # 确保调色板不包含灰色
    for k in range(len(palette_match)):
        while tuple(palette_match[k]) == invalid_color:
            palette_match[k] = np.random.randint(0, 255, 3, dtype=np.uint8)
    
    # === 第一行：GT匹配 ===
    gt_id2col_ref, gt_id2col_tgt = build_gt_color_mapping(gt_matches, ref_gt, tgt_gt, palette_match)
    
    gt_ref_colored = colorize_label_with_mapping(ref_gt, gt_id2col_ref, seed=cfg.seed, bg_label=cfg.bg_label, bg_color=(255,255,255))
    gt_tgt_colored = colorize_label_with_mapping(tgt_gt, gt_id2col_tgt, seed=cfg.seed+1, bg_label=cfg.bg_label, bg_color=(255,255,255))
    
    # 叠加到原图上
    gt_ref_overlay = (0.5*ref_img_np.astype(np.float32) + 0.5*gt_ref_colored.astype(np.float32)).astype(np.uint8)
    gt_tgt_overlay = (0.5*tgt_img_np.astype(np.float32) + 0.5*gt_tgt_colored.astype(np.float32)).astype(np.uint8)
    
    # === 第二行：分割结果（不考虑颜色对应） ===
    ref_seg_colored = colorize_label(ref_seg, seed=cfg.seed+300, bg_label=cfg.bg_label, bg_color=(255,255,255))
    tgt_seg_colored = colorize_label(tgt_seg, seed=cfg.seed+400, bg_label=cfg.bg_label, bg_color=(255,255,255))
    
    # 叠加到原图上
    ref_seg_overlay = (0.5*ref_img_np.astype(np.float32) + 0.5*ref_seg_colored.astype(np.float32)).astype(np.uint8)
    tgt_seg_overlay = (0.5*tgt_img_np.astype(np.float32) + 0.5*tgt_seg_colored.astype(np.float32)).astype(np.uint8)
    
    # === 第三行：预测匹配（只显示有效匹配的区域）===
    # 构建颜色组，确保相关区域使用相同颜色
    color_groups = build_color_groups_for_matches(valid_pred_matches)
    pred_id2col_ref, pred_id2col_tgt = assign_colors_to_groups(color_groups, palette_match, cfg)
    
    # 创建显示有效匹配的分割结果（有效匹配=彩色，无效区域=灰色，背景=白色）
    ref_seg_valid = create_valid_matches_segmentation(ref_seg, valid_pred_matches, all_pred_matches, ref_gt, 'ref', cfg.bg_label)
    tgt_seg_valid = create_valid_matches_segmentation(tgt_seg, valid_pred_matches, all_pred_matches, tgt_gt, 'tgt', cfg.bg_label)
    
    # 使用相同的颜色映射确保匹配区域颜色一致
    pred_ref_colored = colorize_label_with_mapping(ref_seg_valid, pred_id2col_ref, seed=cfg.seed, bg_label=cfg.bg_label, bg_color=(255,255,255))
    pred_tgt_colored = colorize_label_with_mapping(tgt_seg_valid, pred_id2col_tgt, seed=cfg.seed+1, bg_label=cfg.bg_label, bg_color=(255,255,255))
    
    # 叠加到原图上
    pred_ref_overlay = (0.5*ref_img_np.astype(np.float32) + 0.5*pred_ref_colored.astype(np.float32)).astype(np.uint8)
    pred_tgt_overlay = (0.5*tgt_img_np.astype(np.float32) + 0.5*pred_tgt_colored.astype(np.float32)).astype(np.uint8)
    
    # === 水平拼接每一行 ===
    row1 = np.concatenate([gt_ref_overlay, gt_tgt_overlay], axis=1)
    row2 = np.concatenate([ref_seg_overlay, tgt_seg_overlay], axis=1)  
    row3 = np.concatenate([pred_ref_overlay, pred_tgt_overlay], axis=1)
    
    # === 添加caption ===
    row1_with_caption = add_caption_to_image(row1, f"Ground Truth Matches ({gt_group_count} pairs)")
    row2_with_caption = add_caption_to_image(row2, "Region Segmentation Results")
    row3_with_caption = add_caption_to_image(row3, f"Valid Predicted Matches ({avg_valid_matches} pairs)")
    
    # === 添加空白间隔并垂直拼接 ===
    gap_height = 20
    gap = np.full((gap_height, row1_with_caption.shape[1], 3), 255, dtype=np.uint8)  # 白色间隔
    
    combined = np.concatenate([
        row1_with_caption,
        gap,
        row2_with_caption, 
        gap,
        row3_with_caption
    ], axis=0)
    
    return combined

def create_grid_visualization(ref_img_np, tgt_img_np,
                              ref_gt, tgt_gt,
                              ref_seg, tgt_seg,
                              valid_pred_matches, all_pred_matches, gt_matches,
                              cfg: PMESConfig,
                              avg_valid_matches: int, gt_group_count: int):
    """
    生成 2×5 网格：
       原图 | GT Overlay | Segmentation | GT Overlay | Pred-Match
       （ref 在上，tgt 在下；每块 512×512；列间 20 px，行间 20 px）
    """
    # ---------- 调色板 ----------
    all_matches = list(set(valid_pred_matches + gt_matches))
    palette_match = make_palette(len(all_matches) + 10, seed=cfg.seed + 99)

    # ---------- GT 颜色映射 ----------
    gt_id2col_ref, gt_id2col_tgt = build_gt_color_mapping(
        gt_matches, ref_gt, tgt_gt, palette_match
    )

    # ======= 1. 原图 =======
    ref_orig_sq = _to_square512(ref_img_np)
    tgt_orig_sq = _to_square512(tgt_img_np)

    # ======= 2. GT Overlay =======
    gt_ref_overlay = (0.5 * ref_img_np + 0.5 *
                      colorize_label_with_mapping(
                          ref_gt, gt_id2col_ref,
                          seed=cfg.seed, bg_label=cfg.bg_label,
                          bg_color=(255, 255, 255)
                      )).astype(np.uint8)
    gt_tgt_overlay = (0.5 * tgt_img_np + 0.5 *
                      colorize_label_with_mapping(
                          tgt_gt, gt_id2col_tgt,
                          seed=cfg.seed + 1, bg_label=cfg.bg_label,
                          bg_color=(255, 255, 255)
                      )).astype(np.uint8)

    gt_ref_overlay_sq = _to_square512(gt_ref_overlay)
    gt_tgt_overlay_sq = _to_square512(gt_tgt_overlay)

    # ======= 3. Segmentation Overlay =======
    ref_seg_overlay = (0.5 * ref_img_np + 0.5 *
                       colorize_label(ref_seg,
                                      seed=cfg.seed + 300,
                                      bg_label=cfg.bg_label,
                                      bg_color=(255, 255, 255))).astype(np.uint8)
    tgt_seg_overlay = (0.5 * tgt_img_np + 0.5 *
                       colorize_label(tgt_seg,
                                      seed=cfg.seed + 400,
                                      bg_label=cfg.bg_label,
                                      bg_color=(255, 255, 255))).astype(np.uint8)

    ref_seg_overlay_sq = _to_square512(ref_seg_overlay)
    tgt_seg_overlay_sq = _to_square512(tgt_seg_overlay)

    # ======= 4. GT Overlay =======
    gt_ref_color_sq = gt_ref_overlay_sq
    gt_tgt_color_sq = gt_tgt_overlay_sq

    # ======= 5. Pred-Match Overlay =======
    color_groups = build_color_groups_for_matches(valid_pred_matches)
    pred_id2col_ref, pred_id2col_tgt = assign_colors_to_groups(
        color_groups, palette_match, cfg
    )

    ref_seg_valid = create_valid_matches_segmentation(
        ref_seg, valid_pred_matches, all_pred_matches,
        ref_gt, 'ref', cfg.bg_label
    )
    tgt_seg_valid = create_valid_matches_segmentation(
        tgt_seg, valid_pred_matches, all_pred_matches,
        tgt_gt, 'tgt', cfg.bg_label
    )

    pred_ref_overlay = (0.5 * ref_img_np + 0.5 *
                        colorize_label_with_mapping(
                            ref_seg_valid, pred_id2col_ref,
                            seed=cfg.seed, bg_label=cfg.bg_label,
                            bg_color=(255, 255, 255)
                        )).astype(np.uint8)
    pred_tgt_overlay = (0.5 * tgt_img_np + 0.5 *
                        colorize_label_with_mapping(
                            tgt_seg_valid, pred_id2col_tgt,
                            seed=cfg.seed + 1, bg_label=cfg.bg_label,
                            bg_color=(255, 255, 255)
                        )).astype(np.uint8)

    pred_ref_overlay_sq = _to_square512(pred_ref_overlay)
    pred_tgt_overlay_sq = _to_square512(pred_tgt_overlay)

    # ---------- 拼接 ----------
    col_gap = 20
    row_gap = 20
    gap_col = np.full((512, col_gap, 3), 255, np.uint8)
    gap_row = np.full((row_gap, 512 * 5 + col_gap * 6, 3), 255, np.uint8)
    gap_side = np.full((512, col_gap, 3), 255, np.uint8)  # 左右边距

    # ref 行
    row_ref = np.concatenate([
        gap_side,             # 左边距
        ref_orig_sq,          gap_col,
        gt_ref_overlay_sq,    gap_col,
        ref_seg_overlay_sq,   gap_col,
        gt_ref_color_sq,      gap_col,
        pred_ref_overlay_sq,
        gap_side              # 右边距
    ], axis=1)

    # tgt 行
    row_tgt = np.concatenate([
        gap_side,             # 左边距
        tgt_orig_sq,          gap_col,
        gt_tgt_overlay_sq,    gap_col,
        tgt_seg_overlay_sq,   gap_col,
        gt_tgt_color_sq,      gap_col,
        pred_tgt_overlay_sq,
        gap_side              # 右边距
    ], axis=1)

    combined = np.concatenate([row_ref, gap_row, row_tgt], axis=0)
    return combined


def build_color_groups_for_matches(pred_matches):
    """构建颜色组：每个组内的区域应该使用相同颜色"""
    
    # 使用Union-Find来构建连通组件
    class UnionFind:
        def __init__(self):
            self.parent = {}
            
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px != py:
                self.parent[px] = py
    
    uf = UnionFind()
    
    # 将每个匹配对中的ref和tgt区域连接
    for rid, tid in pred_matches:
        ref_key = f"ref_{rid}"
        tgt_key = f"tgt_{tid}"
        uf.union(ref_key, tgt_key)
    
    # 构建颜色组
    groups = {}
    for rid, tid in pred_matches:
        ref_key = f"ref_{rid}"
        tgt_key = f"tgt_{tid}"
        root = uf.find(ref_key)
        
        if root not in groups:
            groups[root] = {'ref': set(), 'tgt': set()}
        
        groups[root]['ref'].add(rid)
        groups[root]['tgt'].add(tid)
    
    return list(groups.values())

def create_valid_matches_segmentation(seg, pred_matches, all_pred_matches, gt, side, bg_label):
    """
    创建显示有效匹配的分割结果
    
    Args:
        seg: 原始分割结果 (H, W)
        pred_matches: 有效匹配对 [(ref_id, tgt_id), ...] (纯度≥0.8)
        all_pred_matches: 所有预测匹配对 [(ref_id, tgt_id), ...] (包含纯度不够的)
        gt: GT分割结果，用于确定背景区域 (实际未使用，保持接口一致性)
        side: 'ref' 或 'tgt'，指定是参考图还是目标图
        bg_label: 背景标签
    
    Returns:
        分割结果：背景=bg_label, 有效匹配=原ID, 无效区域=-1
    """
    INVALID_LABEL = -1  # 特殊标签表示无效区域（灰色）
    
    # 提取区域ID
    if side == 'ref':
        valid_region_ids = set([rid for rid, tid in pred_matches])
        all_pred_region_ids = set([rid for rid, tid in all_pred_matches])
    else:  # side == 'tgt'
        valid_region_ids = set([tid for rid, tid in pred_matches])
        all_pred_region_ids = set([tid for rid, tid in all_pred_matches])
    
    # 创建新的分割结果
    seg_result = np.full_like(seg, INVALID_LABEL)  # 默认无效区域（灰色）
    
    # 1. 设置背景区域（白色）- 基于分割结果，不是GT
    bg_mask = (seg == bg_label)
    seg_result[bg_mask] = bg_label
    
    # 2. 设置有效匹配区域（彩色）
    for region_id in valid_region_ids:
        mask = (seg == region_id)
        seg_result[mask] = region_id
    
    # 3. 无效前景区域保持 INVALID_LABEL（灰色）
    # 包括：
    # - 纯度不够的匹配区域
    # - 分割结果中完全未被匹配的前景区域
    
    return seg_result

def assign_colors_to_groups(color_groups, palette_match, cfg):
    """为每个颜色组分配颜色"""
    pred_id2col_ref = {}
    pred_id2col_tgt = {}
    
    for group_idx, group in enumerate(color_groups):
        if group_idx < len(palette_match):
            color_idx = (group_idx + 1) % len(palette_match)
            col = palette_match[color_idx]
            
            # 为组内所有ref区域分配相同颜色
            for rid in group['ref']:
                pred_id2col_ref[rid] = col
            
            # 为组内所有tgt区域分配相同颜色  
            for tid in group['tgt']:
                pred_id2col_tgt[tid] = col
    
    return pred_id2col_ref, pred_id2col_tgt


def build_gt_color_mapping(gt_matches, ref_gt, tgt_gt, palette):
    """
    给 GT 中成组匹配的区域分配同一颜色。
    支持一对多 / 多对多：用 Union-Find 把所有互相关联的 id 合并成组。
    """
    from collections import defaultdict

    # 没有 GT 匹配就返回空映射，后续会走随机/灰色逻辑
    if not gt_matches:
        return {}, {}

    # ------- Union-Find -------
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 把 ref/tgt 的 id 放到不同命名空间，避免冲突
    for rid, tid in gt_matches:
        union(("ref", int(rid)), ("tgt", int(tid)))

    groups = defaultdict(lambda: {"ref": set(), "tgt": set()})
    for rid, tid in gt_matches:
        root = find(("ref", int(rid)))
        groups[root]["ref"].add(int(rid))
        groups[root]["tgt"].add(int(tid))

    # ------- 为每个组分配颜色 -------
    gt_id2col_ref, gt_id2col_tgt = {}, {}
    pal_len = len(palette)
    for idx, g in enumerate(groups.values()):
        col = tuple(palette[(idx + 1) % pal_len])  # 跳过 0 保留给背景
        for rid in g["ref"]:
            gt_id2col_ref[rid] = col
        for tid in g["tgt"]:
            gt_id2col_tgt[tid] = col

    return gt_id2col_ref, gt_id2col_tgt


def colorize_label_with_mapping(lab: np.ndarray, id2col: Dict[int, Tuple[int,int,int]],
                                seed: int = 0, bg_label: Optional[int] = None,
                                bg_color=(255, 255, 255)) -> np.ndarray:
    """按给定 id→颜色 映射着色，其余 id 用灰色。"""
    # 灰色用于无效区域，确保与调色板颜色区分
    invalid_color = (160, 160, 160)
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    ids = np.unique(lab)
    
    for i in ids:
        if i == -1:  # 特殊标签：无效区域（灰色）
            col = invalid_color
        elif bg_label is not None and i == bg_label:
            col = bg_color
        elif i in id2col:
            col = id2col[i]
        else:
            # 其他未指定区域也用灰色
            col = invalid_color
        rgb[lab == i] = col
    
    return rgb


# ============== CLI ==============
def build_argparser():
    p = argparse.ArgumentParser("PM+ES Region Segmentation (单对图片处理)")
    # 数据集选择 - 使用布尔标志
    p.add_argument('--is-pbc', action='store_true',
                   help='使用PBC数据集 (默认使用GenAI数据集)')
    # 图片对索引参数
    p.add_argument('--pair-index', type=int, default=0,
                   help='数据集中的图片对索引')
    # GenAI数据集参数
    p.add_argument('--csv-path', type=str,
                   default=None,
                   help='Path to the CSV file listing evaluation image pairs (required for GenAI mode)')
    p.add_argument('--lineart-dir', type=str,
                   default=None,
                   help='Root directory of line-art images (required for GenAI mode)')
    p.add_argument('--labels-dir', type=str,
                   default=None,
                   help='Root directory of label images (required for GenAI mode)')
    # PBC dataset args
    p.add_argument('--pbc-root', type=str,
                   default=None,
                   help='Root directory of PaintBucket-Character dataset (required for PBC mode)')
    # 通用参数
    p.add_argument('--model-path', type=str, default='models/lineart_transformer_200k_p32_best.pth', help='模型文件路径')
    p.add_argument('--out-dir', type=str, default='test', help='输出目录')
    p.add_argument('--patch-size', type=int, default=32)
    p.add_argument('--neigh-mode', type=str, default='8', choices=['4','8'])
    p.add_argument('--patch-thresh', type=float, default=0.72)
    p.add_argument('--edge-sigma', type=float, default=2.0)
    p.add_argument('--edge-gamma', type=float, default=0.9)
    p.add_argument('--edge-percentile', type=float, default=80.0)
    p.add_argument('--watershed-line', action='store_true')
    # ONLY ratio
    p.add_argument('--min-region-ratio', type=float, default=0.01,
                   help='min_px = ratio * #FG pixels; set 0 to disable merge.')
    # 区域匹配阈值
    p.add_argument('--match-thresh', type=float, default=0.6,
                   help='ref 区域匹配到 tgt 区域的相似度阈值')
    p.add_argument('--no-save-intermediates', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p

def cfg_from_args(args):
    return PMESConfig(
        patch_size=args.patch_size,
        neigh_mode=args.neigh_mode,
        patch_thresh=args.patch_thresh,
        edge_sigma=args.edge_sigma,
        edge_gamma=args.edge_gamma,
        edge_percentile=args.edge_percentile,
        watershed_line=args.watershed_line,
        min_region_ratio=args.min_region_ratio,
        out_dir=args.out_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        debug=args.debug,
        save_intermediates=not args.no_save_intermediates,
        seed=args.seed,
        match_thresh=args.match_thresh,
    )

def main():
    parser = build_argparser()
    args = parser.parse_args()
    
    # 确定数据集类型：有--is-pbc时使用PBC，否则使用GenAI
    is_genai = not args.is_pbc
    
    cfg = cfg_from_args(args)

    if cfg.debug:
        print("=== PM+ES 单对图片处理开始 (详细模式) ===")
        dataset_type_str = "GENAI" if is_genai else "PBC"
        print(f"数据集类型: {dataset_type_str}")
        print(f"模型路径: {args.model_path}")
        print(f"输出目录: {args.out_dir}")
        print("\n--- 详细配置 ---")
        for k,v in cfg.to_dict().items():
            print(f"{k}: {v}")
    else:
        print("=== PM+ES 单对图片处理开始 ===")

    try:
        # 检查数据集路径
        if is_genai:
            # 检查GenAI数据集路径
            for path, name in [(args.csv_path, "CSV文件"), 
                               (args.lineart_dir, "线稿目录"), 
                               (args.labels_dir, "标签目录")]:
                if not os.path.exists(path):
                    print(f"错误: {name}不存在: {path}")
                    return
        else:
            # 检查PBC数据集路径
            if not os.path.exists(args.pbc_root):
                print(f"错误: PBC数据集根目录不存在: {args.pbc_root}")
                return
        
        if not cfg.debug:
            dataset_type_str = "GENAI" if is_genai else "PBC"
            print(f"处理 {dataset_type_str} 数据集第 {args.pair_index + 1} 对图片...")
        
        region_seg_pair_from_dataset(
            is_genai=is_genai,
            csv_path=args.csv_path,
            lineart_dir=args.lineart_dir,
            labels_dir=args.labels_dir,
            pbc_root=args.pbc_root,
            model_path=args.model_path,
            cfg=cfg,
            pair_index=args.pair_index
        )
            
    except KeyboardInterrupt:
        print("\n用户中断了处理过程")
    except Exception as e:
        print(f"\n处理过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
