import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging

from data import PBCLineArtDataset
from model import LineArtTransformerModel


def sampled_masked_clip_loss(score_matrix: torch.Tensor, gt_matrix: torch.Tensor,
                             num_pos: int = 256, num_neg: int = 64, temperature=0.07, epsilon=1e-8):
    """
    稀疏监督下的对比损失：采样部分正样本对，每对配K个负样本
    """
    B, S, _ = score_matrix.shape
    total_loss = 0.0
    valid_batch = 0

    for b in range(B):
        pos_indices = torch.nonzero(gt_matrix[b] > 0, as_tuple=False)  # 所有正样本对 (i, j)
        if len(pos_indices) == 0:
            continue  # 跳过空 supervision

        # 随机采样 num_pos 个正样本
        num_samples = min(len(pos_indices), num_pos)
        sampled = pos_indices[torch.randperm(len(pos_indices))[:num_samples]]

        logits = score_matrix[b] / temperature
        logits = logits - logits.max(dim=-1, keepdim=True).values  # 数值稳定性

        loss_per_pair = []

        for idx in range(sampled.shape[0]):
            i, j = sampled[idx]  # ref i → tgt j

            # 从除 j 以外的位置采 num_neg 个负样本
            all_neg = torch.arange(S, device=score_matrix.device)
            neg_candidates = all_neg[all_neg != j]
            neg_sample = neg_candidates[torch.randperm(len(neg_candidates))[:num_neg]]
            all_sampled = torch.cat([j.unsqueeze(0), neg_sample])  # 正样本 + 负样本

            # 对第 i 行做局部 softmax
            selected_scores = logits[i, all_sampled]  # shape: (1 + num_neg,)
            log_probs = torch.log_softmax(selected_scores, dim=0)
            loss = -log_probs[0]  # 只关心正样本的 log prob

            loss_per_pair.append(loss)

        if loss_per_pair:
            total_loss += torch.stack(loss_per_pair).mean()
            valid_batch += 1

    if valid_batch == 0:
        return torch.tensor(0.0, requires_grad=True, device=score_matrix.device)
    return total_loss / valid_batch

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_pos_count = 0
    total_sampled = 0

    for ref_img, tgt_img, gt_matrix, _, _ in tqdm(dataloader, desc="Training", leave=False):
        ref_img = ref_img.to(device)
        tgt_img = tgt_img.to(device)
        gt_matrix = gt_matrix.to(device)

        optimizer.zero_grad()
        pred_matrix = model(ref_img, tgt_img)

        loss = sampled_masked_clip_loss(pred_matrix, gt_matrix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 统计当前 batch 有效 supervision 数
        with torch.no_grad():
            B = gt_matrix.shape[0]
            pos_this_batch = sum((gt_matrix[b] > 0).sum().item() for b in range(B))
            total_pos_count += pos_this_batch
            total_sampled += B * gt_matrix.shape[1] * gt_matrix.shape[2]

    avg_loss = total_loss / len(dataloader)
    avg_pos = total_pos_count / total_sampled
    print(f"  [Supervision] Avg positive match ratio: {avg_pos:.6f}")

    return avg_loss

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_pos_count = 0
    total_sampled = 0

    with torch.no_grad():
        for idx, (ref_img, tgt_img, gt_matrix, ref_name, tgt_name) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            ref_img = ref_img.to(device)
            tgt_img = tgt_img.to(device)
            gt_matrix = gt_matrix.to(device)

            pred_matrix = model(ref_img, tgt_img)
            loss = sampled_masked_clip_loss(pred_matrix, gt_matrix)
            total_loss += loss.item()

            B = gt_matrix.shape[0]
            total_sampled += B * gt_matrix.shape[1] * gt_matrix.shape[2]
            total_pos_count += sum((gt_matrix[b] > 0).sum().item() for b in range(B))

    avg_loss = total_loss / len(dataloader)
    avg_pos = total_pos_count / total_sampled
    print(f"  [Validation] Avg positive match ratio: {avg_pos:.6f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of PaintBucket-Character dataset')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transform_image = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    full_root = os.path.join(args.root_dir, 'train', 'PaintBucket_Char')
    full_dataset = PBCLineArtDataset(
        root_dir=full_root,
        transform_image=transform_image,
        patch_size=args.patch_size,
        label_img_resize_size=(512, 512)
    )
    val_size = 300
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    num_patches = (512 // args.patch_size) ** 2

    model = LineArtTransformerModel(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_patches=num_patches
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # train_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    steps_per_epoch = math.ceil(train_size / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    min_lr_factor = 0.01
    warmup_steps = int(total_steps * 0.1)
    hold_steps = int(total_steps * 0.1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + hold_steps:
            return 1.0
        else:
            decay_steps = total_steps - warmup_steps - hold_steps
            decay_progress = (current_step - warmup_steps - hold_steps) / float(max(1, decay_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return cosine_decay * (1.0 - min_lr_factor) + min_lr_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses = []
    val_losses = []
    logging.basicConfig(filename='training_pbc_clip.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    best_val_loss = float('inf')
    best_model_path = "lineart_transformer_pbc_clip_best.pth"
    model_path = "lineart_transformer_pbc_clip_last.pth"

    for epoch in range(args.epochs):
        print(f"=== Epoch: {epoch+1}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log = f"[Epoch {epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        print(log)
        logging.info(log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model to {best_model_path}")

    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved final model to {model_path}")

    plt.figure()
    plt.plot(range(1, args.epochs+1), train_losses, label="Train")
    plt.plot(range(1, args.epochs+1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve_pbc_clip.png")


if __name__ == '__main__':
    main()
