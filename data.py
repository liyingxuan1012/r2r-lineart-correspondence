import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import imageio.v2 as imageio


class LineArtDataset(Dataset):
    def __init__(self, 
                 csv_path,
                 root_dir_lineart,
                 root_dir_label,
                 transform_image=None,
                 patch_size=32,
                 label_img_resize_size=(512, 896)):
        self.csv_path = csv_path
        self.root_dir_lineart = root_dir_lineart
        self.root_dir_label = root_dir_label
        self.transform_image = transform_image
        self.patch_size = patch_size
        self.resize_size = label_img_resize_size  # (H, W)

        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dir_name = row["dir"]
                ref_name = row["reference"]
                tgt_name = row["target"]
                self.samples.append((dir_name, ref_name, tgt_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ===== 1. 加载线稿图像 =====
        dir_name, ref_name, tgt_name = self.samples[idx]
        ref_path = os.path.join(self.root_dir_lineart, dir_name, ref_name)
        tgt_path = os.path.join(self.root_dir_lineart, dir_name, tgt_name)

        ref_img = Image.open(ref_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')

        if self.transform_image:
            ref_img = self.transform_image(ref_img)  # (3, H, W)
            tgt_img = self.transform_image(tgt_img)  # (3, H, W)

        # ===== 2. 加载标签图，并 resize =====
        ref_label_name = ref_name.replace('.jpg', '.png') if '.jpg' in ref_name else ref_name
        tgt_label_name = tgt_name.replace('.jpg', '.png') if '.jpg' in tgt_name else tgt_name

        ref_label_path = os.path.join(self.root_dir_label, dir_name, ref_label_name)
        tgt_label_path = os.path.join(self.root_dir_label, dir_name, tgt_label_name)

        ref_label_img = Image.open(ref_label_path).convert('I')
        tgt_label_img = Image.open(tgt_label_path).convert('I')

        ref_label_img = ref_label_img.resize(self.resize_size[::-1], Image.NEAREST)  # (W, H)
        tgt_label_img = tgt_label_img.resize(self.resize_size[::-1], Image.NEAREST)

        ref_label_img = torch.from_numpy(np.array(ref_label_img)).long()  # (H, W)
        tgt_label_img = torch.from_numpy(np.array(tgt_label_img)).long()

        # ===== 3. 读取 region_map.json =====
        region_map_json = os.path.join(self.root_dir_label, dir_name, "region_map.json")
        with open(region_map_json, 'r', encoding='utf-8') as f:
            region_map_list = json.load(f)

        matched_record = next((r for r in region_map_list if r["reference"] == ref_name and r["target"] == tgt_name), None)
        ref2tgt_map = {
            int(k): v["match_region"]
            for k, v in matched_record["region_map"].items()
        } if matched_record else {}

        # ===== 4. Patch 分割并统计 majority label =====
        def get_patch_labels(label_img, threshold=0.55):
            H, W = label_img.shape
            ph, pw = self.patch_size, self.patch_size
            nh, nw = H // ph, W // pw
            patch_labels = []

            for py in range(nh):
                for px in range(nw):
                    y0, x0 = py * ph, px * pw
                    patch = label_img[y0:y0+ph, x0:x0+pw]
                    vals, counts = torch.unique(patch, return_counts=True)
                    total = counts.sum().item()
                    max_idx = torch.argmax(counts)
                    dominant_ratio = counts[max_idx].item() / total

                    if dominant_ratio >= threshold and vals[max_idx].item() != 0:  # 0 是背景
                        patch_labels.append(vals[max_idx].item())
                    else:
                        patch_labels.append(-1)

            return patch_labels, nh, nw

        patch_labels_ref, nh, nw = get_patch_labels(ref_label_img)
        patch_labels_tgt, nh_t, nw_t = get_patch_labels(tgt_label_img)

        N = nh * nw
        M = nh_t * nw_t

        # ===== 5. 构建 GT matrix =====
        gt_matrix = torch.zeros((N + M, N + M), dtype=torch.float32)

        # 图1内部相似性
        for i in range(N):
            for j in range(N):
                if patch_labels_ref[i] != -1 and patch_labels_ref[j] != -1:
                    if patch_labels_ref[i] == patch_labels_ref[j]:
                        gt_matrix[i, j] = 1.0

        # 图2内部相似性
        for i in range(M):
            for j in range(M):
                if patch_labels_tgt[i] != -1 and patch_labels_tgt[j] != -1:
                    if patch_labels_tgt[i] == patch_labels_tgt[j]:
                        gt_matrix[N + i, N + j] = 1.0

        # 跨图像匹配
        for i in range(N):
            ref_rid = patch_labels_ref[i]
            if ref_rid != -1 and ref_rid in ref2tgt_map:
                tgt_rid = ref2tgt_map[ref_rid]
                if tgt_rid != -1:
                    for j in range(M):
                        if patch_labels_tgt[j] == tgt_rid and patch_labels_tgt[j] != -1:
                            gt_matrix[i, N + j] = 1.0
                            gt_matrix[N + j, i] = 1.0

        return ref_img, tgt_img, gt_matrix, ref_path, tgt_path, patch_labels_ref, patch_labels_tgt


class PBCLineArtDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 transform_image=None,
                 patch_size=16,
                 label_img_resize_size=(512, 512)):
        self.root_dir = root_dir
        self.transform_image = transform_image
        self.patch_size = patch_size
        self.resize_size = label_img_resize_size

        self.samples = []  # list of (dir_name, ref_name, tgt_name)

        for dir_name in sorted(os.listdir(root_dir)):
            char_path = os.path.join(root_dir, dir_name, 'line')
            if not os.path.isdir(char_path):
                continue
            filenames = sorted([f for f in os.listdir(char_path) if f.endswith('.png')])
            for i in range(len(filenames) - 7):
                ref_name = filenames[i]
                tgt_name = filenames[i + 7]
                self.samples.append((dir_name, ref_name, tgt_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ===== 1. 加载线稿图像 =====
        dir_name, ref_name, tgt_name = self.samples[idx]
        ref_path = os.path.join(self.root_dir, dir_name, "line", ref_name)
        tgt_path = os.path.join(self.root_dir, dir_name, "line", tgt_name)

        ref_img = Image.open(ref_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')

        if self.transform_image:
            ref_img = self.transform_image(ref_img)
            tgt_img = self.transform_image(tgt_img)

        # ===== 2. 加载标签图（segment ID） =====
        def load_seg_label(img_name):
            seg_path = os.path.join(self.root_dir, dir_name, "seg", img_name)
            seg_img = imageio.imread(seg_path)
            seg_id_map = seg_img[:, :, 2]
            label_img = Image.fromarray(seg_id_map.astype(np.uint8))
            label_img = label_img.resize(self.resize_size[::-1], Image.NEAREST)
            return torch.from_numpy(np.array(label_img)).long()

        ref_label_img = load_seg_label(ref_name)
        tgt_label_img = load_seg_label(tgt_name)

        # ===== 3. 读取 region_map.json =====
        region_map_json = os.path.join(self.root_dir, dir_name, "seg", "region_map.json")
        with open(region_map_json, 'r', encoding='utf-8') as f:
            region_map_list = json.load(f)

        matched_record = next((r for r in region_map_list if r["reference"] == ref_name and r["target"] == tgt_name), None)
        ref2tgt_map = {
            int(k): v["match_region"]
            for k, v in matched_record["region_map"].items()
        } if matched_record else {}

        # ===== 4. Patch 分割并统计 majority label =====
        def get_patch_labels(label_img, threshold=0.55):
            H, W = label_img.shape
            ph, pw = self.patch_size, self.patch_size
            nh, nw = H // ph, W // pw
            patch_labels = []

            for py in range(nh):
                for px in range(nw):
                    y0, x0 = py * ph, px * pw
                    patch = label_img[y0:y0+ph, x0:x0+pw]
                    vals, counts = torch.unique(patch, return_counts=True)
                    total = counts.sum().item()
                    max_idx = torch.argmax(counts)
                    dominant_ratio = counts[max_idx].item() / total

                    if dominant_ratio >= threshold and vals[max_idx].item() != 1:  # 1 = 背景
                        patch_labels.append(vals[max_idx].item())
                    else:
                        patch_labels.append(-1)
            return patch_labels, nh, nw

        patch_labels_ref, nh, nw = get_patch_labels(ref_label_img)
        patch_labels_tgt, nh_t, nw_t = get_patch_labels(tgt_label_img)

        N = nh * nw
        M = nh_t * nw_t
        gt_matrix = torch.zeros((N + M, N + M), dtype=torch.float32)

        for i in range(N):
            for j in range(N):
                if patch_labels_ref[i] != -1 and patch_labels_ref[j] != -1:
                    if patch_labels_ref[i] == patch_labels_ref[j]:
                        gt_matrix[i, j] = 1.0

        for i in range(M):
            for j in range(M):
                if patch_labels_tgt[i] != -1 and patch_labels_tgt[j] != -1:
                    if patch_labels_tgt[i] == patch_labels_tgt[j]:
                        gt_matrix[N + i, N + j] = 1.0

        for i in range(N):
            ref_rid = patch_labels_ref[i]
            if ref_rid != -1 and ref_rid in ref2tgt_map:
                tgt_rids = ref2tgt_map[ref_rid] if isinstance(ref2tgt_map[ref_rid], list) else [ref2tgt_map[ref_rid]]
                for tgt_rid in tgt_rids:
                    if tgt_rid == -1:
                        continue
                    for j in range(M):
                        if patch_labels_tgt[j] == tgt_rid and patch_labels_tgt[j] != -1:
                            gt_matrix[i, N + j] = 1.0
                            gt_matrix[N + j, i] = 1.0

        return ref_img, tgt_img, gt_matrix, ref_path, tgt_path, patch_labels_ref, patch_labels_tgt
