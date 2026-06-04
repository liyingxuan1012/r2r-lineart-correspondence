import os
import json
import csv
import argparse
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from data import LineArtDataset, PBCLineArtDataset

# 复用 region_seg.py 中的核心实现
from test_region_single import (
    PMESConfig, PatchMeta,
    load_model, get_model_probs, slice_intra_blocks,
    region_seg_single, patch_to_region_map,
    compute_bidirectional_matching_metrics,
    filter_valid_matches_for_visualization,
    create_combined_visualization_with_captions,
    load_gt_region_matches, load_pbc_gt_label,
    load_pbc_gt_region_matches,
    save_image, load_image_and_label, rgb2gray01,
    group_gt_matches
)


# ---------------- helpers ----------------
def load_pairs_from_csv(csv_path: str) -> List[Tuple[str, str, str]]:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        pairs = []
        for row in reader:
            d   = str(row.get('dir', '')).strip()
            ref = str(row.get('reference', '')).strip()
            tgt = str(row.get('target', '')).strip()
            if d and ref and tgt:
                pairs.append((d, ref, tgt))
        return pairs

def compute_average_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_metrics:
        return {}
    ref_metrics = [m['ref'] for m in all_metrics]
    tgt_metrics = [m['tgt'] for m in all_metrics]
    cross_metrics = [m['cross_matching'] for m in all_metrics]

    def avg(ms: List[Dict[str, float]]) -> Dict[str, float]:
        keys = ms[0].keys()
        return {k: float(np.mean([m[k] for m in ms])) for k in keys}

    return {
        'ref_average': avg(ref_metrics),
        'tgt_average': avg(tgt_metrics),
        'overall_average': avg(ref_metrics + tgt_metrics),
        'cross_matching_average': avg(cross_metrics),
        'total_pairs': len(all_metrics),
        'total_images': len(all_metrics) * 2
    }

# -------------- dataset setup --------------
def setup_dataset(is_genai: bool, csv_path: str, lineart_dir: str, labels_dir: str,
                  pbc_root: str, patch_size: int, seed: int):
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
        return dataset, resize_hw, 'csv'
    else:
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
        val_size = 300
        train_size = len(full_dataset) - val_size
        gen = torch.Generator().manual_seed(seed)
        _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=gen)
        return val_dataset, resize_hw, 'pbc'

# -------------- PBC batch loop --------------
def process_pbc_dataset(dataset, model, meta, cfg: PMESConfig, max_pairs=None):
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    total = len(dl) if max_pairs is None else min(max_pairs, len(dl))

    all_metrics, failed = [], []
    for i, batch in enumerate(tqdm(dl, total=total, desc="Processing PBC pairs", unit="pair")):
        if i >= total:
            break
        try:
            ref_img, tgt_img, _gt_matrix, ref_path, tgt_path, *_ = batch

            def denorm(t):
                x = t.squeeze(0).permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                x = (x * std + mean) * 255
                return np.clip(x, 0, 255).astype(np.uint8)

            ref_img_np = denorm(ref_img)
            tgt_img_np = denorm(tgt_img)

            ref_gt = load_pbc_gt_label(ref_path[0], cfg.resize_hw, cfg.bg_label)
            tgt_gt = load_pbc_gt_label(tgt_path[0], cfg.resize_hw, cfg.bg_label)

            prob_mat = get_model_probs(model, ref_img_np, tgt_img_np, cfg)
            ref_ref_probs, tgt_tgt_probs = slice_intra_blocks(prob_mat, meta)

            ref_gray = rgb2gray01(ref_img_np)
            tgt_gray = rgb2gray01(tgt_img_np)

            ref_metrics, ref_seg = region_seg_single(ref_img_np, ref_gray, ref_gt,
                                                     ref_ref_probs, meta, cfg,
                                                     out_dir="", prefix="ref", palette_seed=0)
            tgt_metrics, tgt_seg = region_seg_single(tgt_img_np, tgt_gray, tgt_gt,
                                                     tgt_tgt_probs, meta, cfg,
                                                     out_dir="", prefix="tgt", palette_seed=1000)

            cross_probs = prob_mat[0, :meta.N, meta.N:].numpy()
            ref_p2r = patch_to_region_map(ref_seg, meta)
            tgt_p2r = patch_to_region_map(tgt_seg, meta)

            gt_matches = load_pbc_gt_region_matches(ref_path[0], tgt_path[0], None)
            gt_group_count = len(group_gt_matches(gt_matches))

            mp, mr, matches, avg_valid = compute_bidirectional_matching_metrics(
                ref_seg, tgt_seg, ref_gt, tgt_gt, cross_probs, ref_p2r, tgt_p2r, gt_matches, cfg
            )
            gt_group_count = len(group_gt_matches(gt_matches))

            # 仍保存总可视化（如果不想保存，运行时 --no-save-intermediates 并改 region_seg.py 或 out_dir 非空）
            if cfg.save_intermediates:
                valid_vis = filter_valid_matches_for_visualization(matches, ref_seg, tgt_seg, ref_gt, tgt_gt, cfg)
                vis = create_combined_visualization_with_captions(
                    ref_img_np, tgt_img_np, ref_gt, tgt_gt,
                    ref_seg, tgt_seg, valid_vis, matches, gt_matches, cfg,
                    avg_valid_matches=avg_valid,
                    gt_group_count=gt_group_count
                )
                os.makedirs(cfg.out_dir, exist_ok=True)
                pid = f"PBC_pair_{i+1}"
                save_image(os.path.join(cfg.out_dir, f"{pid}_match_vis.png"), vis)

            pm = {
                "ref": asdict(ref_metrics),
                "tgt": asdict(tgt_metrics),
                "cross_matching": {
                    "precision_purity": mp,
                    "recall_coverage": mr,
                    "num_pred_matches": len(matches),
                    "num_valid_matches": avg_valid,
                    "num_gt_matches": gt_group_count
                }
            }
            
            if cfg.save_intermediates:
                with open(os.path.join(cfg.out_dir, f"PBC_pair_{i+1}_pair_metrics.json"), 'w') as f:
                    json.dump(pm, f, indent=2)
            all_metrics.append(pm)

        except Exception as e:
            failed.append((f"PBC_pair_{i+1}", str(e)))
    return all_metrics, failed

# -------------- GenAI pair loop --------------
def region_seg_pair_with_matching(ref_img_path, tgt_img_path,
                                  ref_gt_path, tgt_gt_path,
                                  model, meta, cfg: PMESConfig,
                                  out_dir: str, pair_id: str,
                                  csv_path: str = None, labels_dir: str = None,
                                  pair_index: int = None, is_genai: bool = True):

    ref_rgb, ref_gray, ref_gt = load_image_and_label(ref_img_path, ref_gt_path, cfg)
    tgt_rgb, tgt_gray, tgt_gt = load_image_and_label(tgt_img_path, tgt_gt_path, cfg)

    prob_mat = get_model_probs(model, ref_rgb, tgt_rgb, cfg)
    ref_ref_probs, tgt_tgt_probs = slice_intra_blocks(prob_mat, meta)

    ref_metrics, ref_seg = region_seg_single(ref_rgb, ref_gray, ref_gt,
                                             ref_ref_probs, meta, cfg,
                                             out_dir="", prefix="ref", palette_seed=0)
    tgt_metrics, tgt_seg = region_seg_single(tgt_rgb, tgt_gray, tgt_gt,
                                             tgt_tgt_probs, meta, cfg,
                                             out_dir="", prefix="tgt", palette_seed=1000)

    cross_probs = prob_mat[0, :meta.N, meta.N:].numpy()
    ref_p2r = patch_to_region_map(ref_seg, meta)
    tgt_p2r = patch_to_region_map(tgt_seg, meta)

    gt_matches = []
    if is_genai and csv_path and labels_dir and pair_index is not None:
        try:
            gt_matches = load_gt_region_matches(csv_path, labels_dir, pair_index)
        except Exception:
            pass

    mp, mr, matches, avg_valid = compute_bidirectional_matching_metrics(
        ref_seg, tgt_seg, ref_gt, tgt_gt, cross_probs, ref_p2r, tgt_p2r, gt_matches, cfg
    )

    # ==== 统计GT group数量 ====
    gt_group_count = len(group_gt_matches(gt_matches))

    if cfg.save_intermediates:
        valid_vis = filter_valid_matches_for_visualization(matches, ref_seg, tgt_seg, ref_gt, tgt_gt, cfg)
        vis = create_combined_visualization_with_captions(
            ref_rgb, tgt_rgb, ref_gt, tgt_gt,
            ref_seg, tgt_seg, valid_vis, matches, gt_matches, cfg, avg_valid, gt_group_count
        )
        os.makedirs(out_dir, exist_ok=True)
        save_image(os.path.join(out_dir, f"{pair_id}_match_vis.png"), vis)

    pair_metrics = {
        "ref": asdict(ref_metrics),
        "tgt": asdict(tgt_metrics),
        "cross_matching": {
            "precision_purity": mp,
            "recall_coverage": mr,
            "num_pred_matches": len(matches),
            "num_valid_matches": avg_valid,
            "num_gt_matches": len(group_gt_matches(gt_matches))
        }
    }
    if cfg.save_intermediates:
        with open(os.path.join(out_dir, f"{pair_id}_pair_metrics.json"), 'w') as f:
            json.dump(pair_metrics, f, indent=2)

    return pair_metrics

# -------------- batch main --------------
def region_seg_batch(csv_path: str, lineart_dir: str, labels_dir: str,
                     model_path: str, cfg: PMESConfig, is_genai: bool = True,
                     pbc_root: str = None):
    dataset, resize_hw, mode = setup_dataset(
        is_genai, csv_path, lineart_dir, labels_dir, pbc_root,
        cfg.patch_size, cfg.seed
    )

    cfg.bg_label = 0 if is_genai else 1
    cfg.resize_hw = resize_hw

    meta = PatchMeta.from_hwP(cfg.resize_hw, cfg.patch_size)
    model = load_model(model_path, meta, cfg.device, patch_size=cfg.patch_size)

    if mode == 'pbc':
        all_metrics, failed_pairs = process_pbc_dataset(
            dataset, model, meta, cfg, max_pairs=getattr(cfg, 'max_pairs', None)
        )
    else:
        pairs = load_pairs_from_csv(csv_path)
        total = len(pairs)
        all_metrics, failed_pairs = [], []
        for i, (dir_name, ref_name, tgt_name) in enumerate(tqdm(pairs, total=total, desc="Processing GENAI pairs", unit="pair")):
            try:
                ref_img_path = os.path.join(lineart_dir, dir_name, ref_name)
                tgt_img_path = os.path.join(lineart_dir, dir_name, tgt_name)
                ref_gt_path  = os.path.join(labels_dir,  dir_name, ref_name)
                tgt_gt_path  = os.path.join(labels_dir,  dir_name, tgt_name)

                for pth in [ref_img_path, tgt_img_path, ref_gt_path, tgt_gt_path]:
                    if not os.path.exists(pth):
                        raise FileNotFoundError(pth)

                m = region_seg_pair_with_matching(
                    ref_img_path, tgt_img_path, ref_gt_path, tgt_gt_path,
                    model, meta, cfg, cfg.out_dir, dir_name,
                    csv_path, labels_dir, i, is_genai
                )
                all_metrics.append(m)
            except Exception as e:
                failed_pairs.append((dir_name, str(e)))

    # --------- summary ----------
    if all_metrics:
        avg = compute_average_metrics(all_metrics)
        # 保存结果
        res = {
            'average_metrics': avg,
            'failed_pairs': failed_pairs,
            'config': cfg.__dict__,
            'dataset_type': 'GENAI' if is_genai else 'PBC'
        }
        os.makedirs(cfg.out_dir, exist_ok=True)
        results_path = os.path.join(cfg.out_dir, 'batch_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        # -------- 按你要求的格式打印 --------
        print("\n=== 平均评价结果 ===")
        print("--- 区域分割评价 ---")
        print("Ref图片平均指标:")
        ref_avg = avg['ref_average']
        print(f"  ARI: {ref_avg['ari']:.4f}, mIoU_PG: {ref_avg['miou_pg']:.4f}, mIoU_GP: {ref_avg['miou_gp']:.4f}, CR: {ref_avg['count_ratio']:.4f}")
        print("Tgt图片平均指标:")
        tgt_avg = avg['tgt_average']
        print(f"  ARI: {tgt_avg['ari']:.4f}, mIoU_PG: {tgt_avg['miou_pg']:.4f}, mIoU_GP: {tgt_avg['miou_gp']:.4f}, CR: {tgt_avg['count_ratio']:.4f}")
        print("总体平均指标:")
        ov = avg['overall_average']
        print(f"  ARI: {ov['ari']:.4f}, mIoU_PG: {ov['miou_pg']:.4f}, mIoU_GP: {ov['miou_gp']:.4f}, CR: {ov['count_ratio']:.4f}")
        print("--- 跨图匹配评价 ---")
        print("平均匹配指标:")
        cross = avg['cross_matching_average']
        print(f"  Match Precision: {cross['precision_purity']:.4f}")
        print(f"  Match Recall: {cross['recall_coverage']:.4f}")
        print(f"  Avg Predicted Matches: {cross['num_pred_matches']:.1f}")
        print(f"  Avg Valid Matches: {cross['num_valid_matches']:.1f}")
        print(f"  Avg GT Matches: {cross['num_gt_matches']:.1f}")

        if failed_pairs:
            print("\n失败的图片对:")
            for name, err in failed_pairs:
                print(f"  {name}: {err}")

        print(f"\n结果已保存到: {results_path}")
    else:
        print("所有图片对处理失败!")

    return all_metrics, failed_pairs

# -------------- CLI --------------
def build_argparser():
    p = argparse.ArgumentParser("PM+ES Region Segmentation (批量处理)")
    p.add_argument('--is-pbc', action='store_true')
    # GenAI
    p.add_argument('--csv-path', type=str, default=None, help='Path to the CSV file listing evaluation image pairs (required for GenAI mode)')
    p.add_argument('--lineart-dir', type=str, default=None, help='Root directory of line-art images (required for GenAI mode)')
    p.add_argument('--labels-dir', type=str, default=None, help='Root directory of label images (required for GenAI mode)')
    # PBC
    p.add_argument('--pbc-root', type=str, default=None, help='Root directory of PaintBucket-Character dataset (required for PBC mode)')
    # common
    p.add_argument('--model-path', type=str, default='models/lineart_transformer_200k_p32_best.pth')
    p.add_argument('--out-dir', type=str, default='results_region_batch_out')
    p.add_argument('--patch-size', type=int, default=32)
    p.add_argument('--neigh-mode', type=str, default='8', choices=['4', '8'])
    p.add_argument('--patch-thresh', type=float, default=0.72)
    p.add_argument('--edge-sigma', type=float, default=2.0)
    p.add_argument('--edge-gamma', type=float, default=0.9)
    p.add_argument('--edge-percentile', type=float, default=80.0)
    p.add_argument('--watershed-line', action='store_true')
    p.add_argument('--min-region-ratio', type=float, default=0.01)
    p.add_argument('--match-thresh', type=float, default=0.6)
    p.add_argument('--max-pairs', type=int, default=None)
    p.add_argument('--no-save-intermediates', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p

def cfg_from_args(args):
    cfg = PMESConfig(
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
    setattr(cfg, 'max_pairs', args.max_pairs)
    return cfg

def main():
    args = build_argparser().parse_args()
    is_genai = not args.is_pbc

    # sanity check
    if is_genai:
        for pth, name in [(args.csv_path, "CSV"), (args.lineart_dir, "lineart_dir"),
                          (args.labels_dir, "labels_dir"), (args.model_path, "model")]:
            if not os.path.exists(pth):
                print(f"错误: {name}不存在: {pth}")
                return
    else:
        for pth, name in [(args.pbc_root, "pbc_root"), (args.model_path, "model")]:
            if not os.path.exists(pth):
                print(f"错误: {name}不存在: {pth}")
                return

    cfg = cfg_from_args(args)
    print("=== PM+ES 批量处理开始 ===")
    print("数据集类型:", "GENAI" if is_genai else "PBC")
    print("模型路径:", args.model_path)
    print("输出目录:", args.out_dir)

    try:
        region_seg_batch(
            csv_path=args.csv_path,
            lineart_dir=args.lineart_dir,
            labels_dir=args.labels_dir,
            model_path=args.model_path,
            cfg=cfg,
            is_genai=is_genai,
            pbc_root=args.pbc_root
        )
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print("\n批量处理错误:", e)
        raise


if __name__ == "__main__":
    main()
