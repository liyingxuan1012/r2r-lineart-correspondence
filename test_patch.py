import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import argparse

from model import LineArtTransformerModel
from data import LineArtDataset


def compute_topk_counts(score_matrix, gt_matrix, topk=1, is_cross=True):
    if is_cross:
        B, N, M = score_matrix.shape
        gt_sub = gt_matrix[:, :N, N:]
        pred_score = torch.sigmoid(score_matrix)
    else:
        B, N, _ = score_matrix.shape
        gt_sub = gt_matrix
        pred_score = torch.sigmoid(score_matrix).clone()
        mask = torch.eye(N, dtype=torch.bool, device=score_matrix.device)
        pred_score.masked_fill_(mask.unsqueeze(0), -1e4)

    topk_idx = torch.topk(pred_score, k=topk, dim=-1).indices
    one_hot = torch.zeros_like(gt_sub, dtype=torch.bool)
    one_hot.scatter_(-1, topk_idx, True)

    valid_mask = gt_sub.sum(dim=-1) > 0
    correct_mask = (one_hot & gt_sub.bool()).any(dim=-1)

    total_correct = int((correct_mask & valid_mask).sum().item())
    total_valid = int(valid_mask.sum().item())
    return total_correct, total_valid


def compute_pr_stats(pred_scores, gt_labels, thresholds):
    tp_counts = np.zeros_like(thresholds)
    pred_counts = np.zeros_like(thresholds)
    pos_total = gt_labels.sum()
    for i, t in enumerate(thresholds):
        pred_mask = (pred_scores >= t)
        tp_counts[i] = np.logical_and(pred_mask, gt_labels).sum()
        pred_counts[i] = pred_mask.sum()
    return tp_counts, pred_counts, pos_total


def get_symmetric_cross_score(pred_matrix):
    B, S, _ = pred_matrix.shape
    N = S // 2
    ref2tgt = pred_matrix[:, :N, N:]
    tgt2ref = pred_matrix[:, N:, :N].transpose(1, 2)
    return (ref2tgt + tgt2ref) / 2.0


def evaluate_pr_curve(tp_counts, pred_counts, pos_total):
    precs = tp_counts / (pred_counts + 1e-8)
    recs = tp_counts / (pos_total + 1e-8)
    f1s = 2 * precs * recs / (precs + recs + 1e-8)
    best_idx = int(np.argmax(f1s))
    return f1s[best_idx], best_idx, precs, recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file listing evaluation image pairs')
    parser.add_argument('--root_lineart', type=str, required=True, help='Root directory of line-art images')
    parser.add_argument('--root_label', type=str, required=True, help='Root directory of label images')
    parser.add_argument('--model_path', type=str, default='lineart_transformer_best.pth')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 设置随机种子，保证可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # === 模型参数 ===
    # 计算单张图patch数
    num_patches = (512 // args.patch_size) * (896 // args.patch_size)
    
    # 初始化模型
    model = LineArtTransformerModel(
        embed_dim=768,
        num_heads=12,
        num_layers=4,
        num_patches=num_patches
    ).to(device).half()

    raw_state = torch.load(args.model_path, map_location=device)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in raw_state.items()})
    model.eval()

    # === 数据加载 ===
    transform_image = T.Compose([
        T.Resize((512, 896)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    # 构建完整数据集
    full_dataset = LineArtDataset(
        csv_path=args.csv_path,
        root_dir_lineart=args.root_lineart,
        root_dir_label=args.root_label,
        transform_image=transform_image,
        patch_size=args.patch_size,
        label_img_resize_size=(512, 896)
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # === 准备阈值统计数组 ===
    thresholds = np.arange(0.5, 1.0, 0.01)
    tp_cross, pred_cross, pos_cross = np.zeros_like(thresholds), np.zeros_like(thresholds), 0
    tp_ref, pred_ref, pos_ref = np.zeros_like(thresholds), np.zeros_like(thresholds), 0
    tp_tgt, pred_tgt, pos_tgt = np.zeros_like(thresholds), np.zeros_like(thresholds), 0
    total1 = total5 = valid1 = valid5 = 0
    total1_ref = total5_ref = valid1_ref = valid5_ref = 0
    total1_tgt = total5_tgt = valid1_tgt = valid5_tgt = 0

    # === 推理 & 统计 ===
    for ref_img, tgt_img, gt_matrix, _, _ in tqdm(val_loader, desc="Evaluating"):
        ref_img = ref_img.to(device).half()
        tgt_img = tgt_img.to(device).half()
        gt_matrix = gt_matrix.to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            pred_matrix = model(ref_img, tgt_img)

        B, S, _ = pred_matrix.shape
        N = S // 2
        score_sym = get_symmetric_cross_score(pred_matrix)

        # Cross
        pred_cross_np = torch.sigmoid(score_sym).flatten().cpu().numpy()
        gt_cross_np = gt_matrix[:, :N, N:].flatten().cpu().numpy()
        pos_cross += gt_cross_np.sum()
        c, p, _ = compute_pr_stats(pred_cross_np, gt_cross_np, thresholds)
        tp_cross += c
        pred_cross += p

        # Internal Ref
        pred_ref_np = torch.sigmoid(pred_matrix[:, :N, :N]).flatten().cpu().numpy()
        gt_ref_np = gt_matrix[:, :N, :N].flatten().cpu().numpy()
        pos_ref += gt_ref_np.sum()
        c, p, _ = compute_pr_stats(pred_ref_np, gt_ref_np, thresholds)
        tp_ref += c
        pred_ref += p

        # Internal Tgt
        pred_tgt_np = torch.sigmoid(pred_matrix[:, N:, N:]).flatten().cpu().numpy()
        gt_tgt_np = gt_matrix[:, N:, N:].flatten().cpu().numpy()
        pos_tgt += gt_tgt_np.sum()
        c, p, _ = compute_pr_stats(pred_tgt_np, gt_tgt_np, thresholds)
        tp_tgt += c
        pred_tgt += p

        # Cross Top-k
        c1, v1 = compute_topk_counts(score_sym, gt_matrix, topk=1, is_cross=True)
        c5, v5 = compute_topk_counts(score_sym, gt_matrix, topk=5, is_cross=True)
        total1 += c1; valid1 += v1
        total5 += c5; valid5 += v5

        # Internal Top-k Ref
        c1r, v1r = compute_topk_counts(pred_matrix[:, :N, :N], gt_matrix[:, :N, :N], topk=1, is_cross=False)
        c5r, v5r = compute_topk_counts(pred_matrix[:, :N, :N], gt_matrix[:, :N, :N], topk=5, is_cross=False)
        total1_ref += c1r; valid1_ref += v1r
        total5_ref += c5r; valid5_ref += v5r

        # Internal Top-k Tgt
        c1t, v1t = compute_topk_counts(pred_matrix[:, N:, N:], gt_matrix[:, N:, N:], topk=1, is_cross=False)
        c5t, v5t = compute_topk_counts(pred_matrix[:, N:, N:], gt_matrix[:, N:, N:], topk=5, is_cross=False)
        total1_tgt += c1t; valid1_tgt += v1t
        total5_tgt += c5t; valid5_tgt += v5t

    print("\n=== Cross-Image Patch Matching ===")
    best_f1_c, best_idx, precs_c, recs_c = evaluate_pr_curve(tp_cross, pred_cross, pos_cross)
    print(f"Best F1: {best_f1_c:.4f} | Threshold: {thresholds[best_idx]:.2f} | Prec: {precs_c[best_idx]:.4f} | Recall: {recs_c[best_idx]:.4f}")
    print(f"Top-1 Accuracy: {total1 / valid1:.4f} | Top-5 Accuracy: {total5 / valid5:.4f}")

    print("\n=== Internal Patch Matching (Ref) ===")
    best_f1_r, best_idx, precs_r, recs_r = evaluate_pr_curve(tp_ref, pred_ref, pos_ref)
    print(f"Best F1: {best_f1_r:.4f} | Threshold: {thresholds[best_idx]:.2f} | Prec: {precs_r[best_idx]:.4f} | Recall: {recs_r[best_idx]:.4f}")
    print(f"Top-1 Accuracy: {total1_ref / valid1_ref:.4f} | Top-5 Accuracy: {total5_ref / valid5_ref:.4f}")

    print("\n=== Internal Patch Matching (Tgt) ===")
    best_f1_t, best_idx, precs_t, recs_t = evaluate_pr_curve(tp_tgt, pred_tgt, pos_tgt)
    print(f"Best F1: {best_f1_t:.4f} | Threshold: {thresholds[best_idx]:.2f} | Prec: {precs_t[best_idx]:.4f} | Recall: {recs_t[best_idx]:.4f}")
    print(f"Top-1 Accuracy: {total1_tgt / valid1_tgt:.4f} | Top-5 Accuracy: {total5_tgt / valid5_tgt:.4f}")

    # === Plot all curves ===
    plt.figure(figsize=(6, 5))
    plt.plot(recs_c, precs_c, marker='o', label=f'Cross (F1={best_f1_c:.4f})')
    plt.plot(recs_r, precs_r, marker='^', label=f'Ref (F1={best_f1_r:.4f})')
    plt.plot(recs_t, precs_t, marker='s', label=f'Tgt (F1={best_f1_t:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Patch-Level PR Curves')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve.png")
    plt.close()


if __name__ == '__main__':
    main()
