import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import argparse

from model import LineArtTransformerModel
from data import LineArtDataset, PBCLineArtDataset


# -------------------------  公共工具  -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------  评估函数  -------------------------
@torch.no_grad()
def evaluate_cross_img_patch_pairs_filtered(
        y_scores, y_true,
        tag: str, save_dir: str,
        save_plot: bool = True,
        fname_plot: str = "pr_curve_cross.png"):
    """
    交叉图像 Patch-level 评估 & PR 曲线保存
    """
    ensure_dir(save_dir)

    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # —— 保存 PR 数据 ——
    np.savez_compressed(
        os.path.join(save_dir, f"{tag}_cross_pr.npz"),
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        ap=ap
    )

    # —— 绘图 ——
    if save_plot:
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f'{tag} (AP={ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Cross-Image Patch-Level PR Curve')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname_plot))
        plt.close()

    # 其它指标
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    binary_best = (y_scores > best_thresh).astype(int)
    return {
        "AP": ap,
        "Best F1": f1_scores[best_idx],
        "Best Threshold": best_thresh,
        "Precision@Best": precision[best_idx],
        "Recall@Best": recall[best_idx],
    }


@torch.no_grad()
def evaluate_internal_patch_pairs(
        all_ref_scores, all_ref_labels,
        all_tgt_scores, all_tgt_labels,
        tag: str, save_dir: str,
        save_plot: bool = True,
        fname_plot: str = "pr_curve_internal.png"):

    ensure_dir(save_dir)

    y_scores_ref = np.concatenate(all_ref_scores)
    y_true_ref = np.concatenate(all_ref_labels)
    y_scores_tgt = np.concatenate(all_tgt_scores)
    y_true_tgt = np.concatenate(all_tgt_labels)

    def eval_single(y_true, y_score, suffix):
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        # —— 保存 PR 数据 ——
        np.savez_compressed(
            os.path.join(save_dir, f"{tag}_intra_{suffix}_pr.npz"),
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            ap=ap
        )
        # 其它指标
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        return {
            "AP": ap,
            "Best F1": f1_scores[best_idx],
            "Best Threshold": best_thresh,
            "Precision@Best": precision[best_idx],
            "Recall@Best": recall[best_idx]
        }, (precision, recall)

    # Ref / Tgt
    result_ref, (prec_ref, rec_ref) = eval_single(y_true_ref, y_scores_ref, suffix="ref")
    result_tgt, (prec_tgt, rec_tgt) = eval_single(y_true_tgt, y_scores_tgt, suffix="tgt")

    # —— 绘图 ——
    if save_plot:
        plt.figure(figsize=(6, 5))
        plt.plot(rec_ref, prec_ref, label=f'{tag}-Ref (AP={result_ref["AP"]:.4f})')
        plt.plot(rec_tgt, prec_tgt, label=f'{tag}-Tgt (AP={result_tgt["AP"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Internal Patch-Level PR Curve')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname_plot))
        plt.close()

    return result_ref, result_tgt


def compute_topk_accuracy(score_matrix, gt_matrix, topk=1):
    B, N, M = score_matrix.shape
    assert B == 1
    pred_score = torch.sigmoid(score_matrix[0])     # (N, M)
    gt_sub = gt_matrix[0, :N, N:]                   # (N, M)
    _, topk_indices = torch.topk(pred_score, k=topk, dim=1)
    correct = 0
    total = 0
    for i in range(N):
        if gt_sub[i].sum() > 0:
            gt_match_indices = (gt_sub[i] > 0).nonzero(as_tuple=True)[0]
            if any(j in gt_match_indices for j in topk_indices[i]):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def get_symmetric_cross_score(pred_matrix):
    B, S, _ = pred_matrix.shape
    N = S // 2
    ref2tgt = pred_matrix[:, :N, N:]                 # (B, N, M)
    tgt2ref = pred_matrix[:, N:, :N].transpose(1, 2) # (B, N, M)
    return (ref2tgt + tgt2ref) / 2.0


# -------------------------  主程序  -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=None, help='Path to the CSV file listing evaluation image pairs (required for GenAI mode)')
    parser.add_argument('--root_lineart', type=str, default=None, help='Root directory of line-art images (required for GenAI mode)')
    parser.add_argument('--root_label', type=str, default=None, help='Root directory of label images (required for GenAI mode)')
    parser.add_argument('--pbc_root', type=str, default=None, help='Root directory of PaintBucket-Character dataset (required for PBC mode)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='GenAI',
                        help='标签前缀，用于保存文件命名，如 GenAI / PBC')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='保存 PR 数据及曲线的文件夹')
    parser.add_argument('--is-pbc', action='store_true',
                        help='使用 PBC 数据集 (默认使用 GenAI)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # === 数据集 ===
    IS_GENAI = not args.is_pbc
    if IS_GENAI:
        resize_hw = (512, 896)
        transform = T.Compose([
            T.Resize(resize_hw),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        dataset = LineArtDataset(
            csv_path=args.csv_path,
            root_dir_lineart=args.root_lineart,
            root_dir_label=args.root_label,
            transform_image=transform,
            patch_size=args.patch_size,
            label_img_resize_size=resize_hw
        )
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=args.num_workers)
    else:
        resize_hw = (512, 512)
        transform = T.Compose([
            T.Resize(resize_hw),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        if args.pbc_root is None:
            raise ValueError("--pbc_root is required when using --is-pbc mode")
        dataset = PBCLineArtDataset(
            root_dir=args.pbc_root,
            transform_image=transform,
            patch_size=args.patch_size,
            label_img_resize_size=resize_hw
        )
        val_size = 300
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=args.num_workers)

    # === 模型 ===
    # 计算单张图patch数
    pr = resize_hw[0] // args.patch_size
    pc = resize_hw[1] // args.patch_size
    num_patches = pr * pc
    
    # 初始化模型
    model = LineArtTransformerModel(
        embed_dim=768,
        num_heads=12,
        num_layers=4,
        num_patches=num_patches,
        patch_size=args.patch_size
    ).to(device)

    raw_state = torch.load(args.model_path, map_location=device)
    new_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    
    # 跳过位置编码参数（模型会自动根据输入尺寸调整）
    pos_embed_keys = ['pos_embed', 'encoder.pos_embedding']
    new_state = {k: v for k, v in new_state.items() if k not in pos_embed_keys}
    
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # === 累积指标 ===
    top1_accs, top5_accs = [], []
    all_ref_scores, all_ref_labels = [], []
    all_tgt_scores, all_tgt_labels = [], []
    all_cross_scores, all_cross_labels = [], []

    for ref_img, tgt_img, gt_matrix, _, _, patch_labels_ref, patch_labels_tgt in tqdm(val_loader, desc="Evaluating"):
        ref_img, tgt_img, gt_matrix = ref_img.to(device), tgt_img.to(device), gt_matrix.to(device)
        
        with torch.no_grad():
            pred_matrix = model(ref_img, tgt_img)

        B, S, _ = pred_matrix.shape
        N = S // 2

        score_sym = get_symmetric_cross_score(pred_matrix)

        # —— Cross-image scores ——（过滤 -1 即背景）
        valid_ref = [i for i in range(N) if patch_labels_ref[i] != -1]
        valid_tgt = [j for j in range(N) if patch_labels_tgt[j] != -1]
        for i in valid_ref:
            for j in valid_tgt:
                all_cross_scores.append(score_sym[0, i, j].item())
                all_cross_labels.append(gt_matrix[0, i, N + j].item())

        # —— Top-k Acc ——
        top1_accs.append(compute_topk_accuracy(score_sym, gt_matrix, topk=1))
        top5_accs.append(compute_topk_accuracy(score_sym, gt_matrix, topk=5))

        # —— Intra-image ——（同样过滤背景）
        valid_mask_ref = torch.tensor(patch_labels_ref) != -1
        valid_mask_tgt = torch.tensor(patch_labels_tgt) != -1

        pred_ref_sub = torch.sigmoid(pred_matrix[0, :N, :N])
        gt_ref_sub = gt_matrix[0, :N, :N]
        pred_tgt_sub = torch.sigmoid(pred_matrix[0, N:, N:])
        gt_tgt_sub = gt_matrix[0, N:, N:]

        all_ref_scores.append(
            pred_ref_sub[valid_mask_ref][:, valid_mask_ref].flatten().cpu().numpy())
        all_ref_labels.append(
            gt_ref_sub[valid_mask_ref][:, valid_mask_ref].flatten().cpu().numpy())
        all_tgt_scores.append(
            pred_tgt_sub[valid_mask_tgt][:, valid_mask_tgt].flatten().cpu().numpy())
        all_tgt_labels.append(
            gt_tgt_sub[valid_mask_tgt][:, valid_mask_tgt].flatten().cpu().numpy())

    # === 保存&打印结果 ===
    cross_result = evaluate_cross_img_patch_pairs_filtered(
        all_cross_scores, all_cross_labels,
        tag=args.dataset_name,
        save_dir=args.save_dir)

    result_ref, result_tgt = evaluate_internal_patch_pairs(
        all_ref_scores, all_ref_labels,
        all_tgt_scores, all_tgt_labels,
        tag=args.dataset_name,
        save_dir=args.save_dir)

    print("\n=== Cross-Image Patch Matching Evaluation ===")
    for k, v in cross_result.items():
        print(f"{k}: {v:.4f}")
    print(f"Top-1 Accuracy: {np.mean(top1_accs):.4f}")
    print(f"Top-5 Accuracy: {np.mean(top5_accs):.4f}")

    print("\n=== Internal Patch Matching Evaluation ===")
    print(f"[Ref]  AP: {result_ref['AP']:.4f} | F1: {result_ref['Best F1']:.4f} | "
          f"Thre: {result_ref['Best Threshold']:.4f} | "
          f"Prec: {result_ref['Precision@Best']:.4f} | Recall: {result_ref['Recall@Best']:.4f}")
    print(f"[Tgt]  AP: {result_tgt['AP']:.4f} | F1: {result_tgt['Best F1']:.4f} | "
          f"Thre: {result_tgt['Best Threshold']:.4f} | "
          f"Prec: {result_tgt['Precision@Best']:.4f} | Recall: {result_tgt['Recall@Best']:.4f}")


if __name__ == '__main__':
    main()
