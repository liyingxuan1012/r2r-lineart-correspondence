import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import LineArtDataset
from model import LineArtTransformerModel


# ---- 更稳妥的 SDPA 加速开关 ----
try:
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    # 某些版本没有 disable_math_sdp
    if hasattr(torch.backends.cuda, "disable_math_sdp"):
        torch.backends.cuda.disable_math_sdp(True)
except Exception as e:
    print(f"[WARN] SDPA backend selection skipped: {e}")


# ========= 缓存 1×(2S×2S) 单位阵，按 (device, S2) 复用 =========
_EYE_CACHE = {}

def _get_eye(device: torch.device, S2: int) -> torch.Tensor:
    """
    返回形状 (1, 2S, 2S) 的 bool 单位阵，用于排除自对角；按 (device, S2) 缓存。
    """
    key = (device, int(S2))
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(S2, device=device, dtype=torch.bool).unsqueeze(0)  # (1,2S,2S)
        _EYE_CACHE[key] = eye
    return eye


##################################################
# 训练与验证
##################################################
def _mark_cudagraph_step_if_available():
    """在每个迭代开头标记一个新的 cudagraph step（若可用）。"""
    try:
        # PyTorch 2.3+
        torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        try:
            # 某些版本接口在 inductor.utils 下
            import torch._inductor.utils as _iu
            if hasattr(_iu, "cudagraph_mark_step_begin"):
                _iu.cudagraph_mark_step_begin()
        except Exception:
            pass


def train_one_epoch(model, dataloader, optimizer, scheduler, device, is_main):
    model.train()
    total_loss = 0.0
    for ref_img, tgt_img, gt_matrix, *_ in tqdm(
        dataloader, desc="Training", leave=False, disable=not is_main
    ):
        _mark_cudagraph_step_if_available()  # ★ 关键：每 iter 开头标记

        ref_img = ref_img.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        tgt_img = tgt_img.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        gt_mask = gt_matrix.to(device, dtype=torch.bool, non_blocking=True)
        del gt_matrix

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 直接从特征上计算对比损失，避免构整块(2S x 2S)相似度矩阵
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            feat_ref, feat_tgt = model(ref_img, tgt_img, return_feats=True)
            loss = sampled_masked_clip_loss_from_feats(
                feat_ref, feat_tgt, gt_mask,
                num_pos=96, num_neg=256, temperature=0.10, exclude_self=True
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, device, is_main):
    model.eval()
    local_loss_sum = torch.tensor(0.0, device=device)
    local_count    = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for ref_img, tgt_img, gt_matrix, *_ in tqdm(
            dataloader, desc="Validating", leave=False, disable=not is_main
        ):
            _mark_cudagraph_step_if_available()  # ★ 验证也标记

            ref_img = ref_img.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            tgt_img = tgt_img.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            gt_mask = gt_matrix.to(device, dtype=torch.bool, non_blocking=True)
            del gt_matrix

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                feat_ref, feat_tgt = model(ref_img, tgt_img, return_feats=True)
                loss = sampled_masked_clip_loss_from_feats(
                    feat_ref, feat_tgt, gt_mask,
                    num_pos=96, num_neg=256, temperature=0.10, exclude_self=True
                )

            bs = ref_img.size(0)
            local_loss_sum += loss.detach() * bs
            local_count    += bs

    if dist.is_initialized():
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count,    op=dist.ReduceOp.SUM)

    global_mean = (local_loss_sum / local_count).item() if local_count.item() > 0 else float("inf")
    return global_mean


##################################################
# 损失函数
##################################################
def sampled_masked_clip_loss_from_feats(
    feat_ref: torch.Tensor,
    feat_tgt: torch.Tensor,
    gt_mask: torch.Tensor,   # (B,2S,2S) bool
    num_pos: int = 256,
    num_neg: int = 96,
    temperature: float = 0.07,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    向量化版 InfoNCE（采样式）：
    - 先在 (B, 2S, 2S) 的正样本掩码上用随机 top-k 采样得到 (B, k) 个正对；
    - 构造 (B, k, D) 与 (B, D, 2S) 的批量乘法得到 (B, k, 2S) 相似度；
    - 与 (B, k, 1+num_neg) 的列索引（正列 + 负列）做 gather 得到 logits；
    - 对每个样本的 k 行做 log_softmax，取正列的 -log p 作为损失；
    - 对于正样本不足 k 的 batch，用有效掩码统计平均。
    """
    B, S, D = feat_ref.shape
    device = feat_ref.device
    S2 = 2 * S
    k = num_pos

    # 拼接特征 (B, 2S, D)
    F = torch.cat([feat_ref, feat_tgt], dim=1)  # (B, 2S, D)

    # 正样本掩码 (B, 2S, 2S)
    pos_mask = gt_mask  # 已是 bool
    if exclude_self:
        eye = _get_eye(device, S2)   # (1,2S,2S) from cache
        pos_mask = pos_mask & (~eye)

    # 若没有任何正样本，直接返回可反传的 0
    if not pos_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # —— 采样正样本：在正样本掩码上做随机 top-k —— #
    M = S2 * S2
    rand_scores = torch.rand((B, M), device=device)
    pos_flat = pos_mask.view(B, M)
    rand_scores = rand_scores.masked_fill(~pos_flat, float('-inf'))

    # 取每个 batch 的 k 个候选（不足 k 时，topk 里会包含 -inf，我们后面用有效掩码剔除）
    topk_idx = torch.topk(rand_scores, k=k, dim=1, largest=True, sorted=False).indices  # (B, k)

    # 有效性掩码：标记 topk 中哪些位置确为正样本
    valid_mask = torch.gather(pos_flat, 1, topk_idx)  # (B, k), bool

    # 将扁平索引还原为 (i, j)
    i_idx = topk_idx // S2                     # (B, k)
    j_idx = topk_idx % S2                      # (B, k)

    # —— 选取被采样的行 (B, k, D) —— #
    F_rows = torch.gather(F, dim=1, index=i_idx.unsqueeze(-1).expand(B, k, D))  # (B, k, D)

    # —— 计算 (B, k, 2S) 的相似度矩阵行块 —— #
    scores = torch.bmm(F_rows, F.transpose(1, 2))  # (B, k, 2S)
    scores = scores / temperature
    scores = scores - scores.max(dim=-1, keepdim=True).values  # 数值稳定

    # —— 构造负样本列索引，避开正列 j_idx —— #
    if num_neg > 0:
        neg = torch.randint(0, S2 - 1, (B, k, num_neg), device=device)
        j_exp = j_idx.unsqueeze(-1)  # (B, k, 1)
        neg = neg + (neg >= j_exp).to(neg.dtype)  # 跳过 j
        col_idx = torch.cat([j_exp, neg], dim=-1)  # (B, k, 1+num_neg)
    else:
        col_idx = j_idx.unsqueeze(-1)  # (B, k, 1)

    selected = torch.gather(scores, dim=2, index=col_idx)  # (B, k, 1+num_neg)

    log_probs = torch.log_softmax(selected.float(), dim=-1)  # 用 FP32 更稳
    loss_mat = -log_probs[..., 0]  # (B, k)

    # 只对有效的正样本对计入损失
    valid_mask = valid_mask.to(loss_mat.dtype)
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = (loss_mat * valid_mask).sum() / valid_count
    return loss


# 备用：从整块相似度矩阵计算（现在默认不用）
def sampled_masked_clip_loss(score_matrix: torch.Tensor, gt_matrix: torch.Tensor,
                             num_pos: int = 256, num_neg: int = 96, temperature=0.07, eps=1e-8):
    B, S2, _ = score_matrix.shape
    device = score_matrix.device

    logits = score_matrix / temperature
    logits = logits - logits.max(dim=-1, keepdim=True).values

    losses = []
    for b in range(B):
        pos = torch.nonzero(gt_matrix[b] > 0, as_tuple=False)
        if pos.numel() == 0:
            continue

        n = min(pos.size(0), num_pos)
        perm = torch.randperm(pos.size(0), device=device)[:n]
        pos_samp = pos[perm]
        i_idx = pos_samp[:, 0]
        j_idx = pos_samp[:, 1]

        neg = torch.randint(0, S2 - 1, (n, num_neg), device=device)
        j_col = j_idx.unsqueeze(1)
        neg = neg + (neg >= j_col).to(neg.dtype)
        col_idx = torch.cat([j_col, neg], dim=1)

        chosen_rows = logits[b].index_select(0, i_idx)
        selected = torch.gather(chosen_rows, 1, col_idx)

        log_probs = torch.log_softmax(selected, dim=1)
        loss_b = -log_probs[:, 0]
        loss_b = loss_b[~torch.isnan(loss_b)]
        if loss_b.numel() > 0:
            losses.append(loss_b.mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


##################################################
# 主函数
##################################################
def main():
    # ==== DDP 基础 ====
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = (dist.get_rank() == 0)

    # ==== 性能开关 ====
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")

    # ==== 参数 ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file listing training image pairs')
    parser.add_argument('--root_lineart', type=str, required=True, help='Root directory of line-art images')
    parser.add_argument('--root_label', type=str, required=True, help='Root directory of label images')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compile', action='store_true', help='enable torch.compile for model')
    args = parser.parse_args()

    # ==== 随机种子 ====
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ==== 预处理 ====
    transform_image = T.Compose([
        T.Resize((512, 896)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # ==== 数据集 ====
    noisy_full = LineArtDataset(
        csv_path=args.csv_path,
        root_dir_lineart=args.root_lineart,
        root_dir_label=args.root_label,
        transform_image=transform_image,
        patch_size=args.patch_size,
        label_img_resize_size=(512, 896)
    )
    n_total = len(noisy_full)
    n_train = int(n_total * 0.9)
    n_val   = n_total - n_train
    train_dataset, val_dataset = random_split(
        noisy_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # ==== DataLoader ====
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, sampler=train_sampler,
        pin_memory=True, persistent_workers=True, prefetch_factor=6,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, sampler=val_sampler,
        pin_memory=True, persistent_workers=True, prefetch_factor=6,
        drop_last=False
    )

    # ==== 模型 ====
    model = LineArtTransformerModel(
        embed_dim=768, num_heads=12, num_layers=4,
        patch_size=args.patch_size,
        attention='linear',
        loftr_attn_dropout=0.1, loftr_ffn_dropout=0.1,
    ).to(device)

    model = model.to(memory_format=torch.channels_last)

    # ---- compile（禁用 cudagraphs）----
    if args.compile:
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"  # 强制禁用
        try:
            model = torch.compile(
                model,
                fullgraph=False,
                options={"triton.cudagraphs": False}  # 同时禁用 triton 的 cudagraphs
            )
            if is_main:
                print("[INFO] torch.compile enabled with cudagraphs disabled.")
        except Exception as e:
            if is_main:
                print(f"[WARN] torch.compile failed, fallback to eager: {e}")

    # ---- DDP 包裹放在 compile 之后 ----
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)

    # ==== 优化器 ====
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999),
            weight_decay=0.05, fused=True
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999),
            weight_decay=0.05
        )

    # ==== 学习率调度 ====
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    min_lr_factor = 0.02
    warmup_steps = int(total_steps * 0.10)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        decay_steps = total_steps - warmup_steps
        decay_progress = (current_step - warmup_steps) / float(max(1, decay_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return cosine_decay * (1.0 - min_lr_factor) + min_lr_factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ==== 日志 ====
    if is_main:
        logging.basicConfig(
            filename='training_200k_p32.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    # ==== 训练循环 ====
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = "models/lineart_transformer_200k_p32_best.pth"
    model_path = "models/lineart_transformer_200k_p32_last.pth"
    train_losses, val_losses = [], []

    ##################################################
    # 早停器（Early Stopping）
    ##################################################
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float("inf")
            self.wait = 0
            self.should_stop = False

        def step(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.should_stop = True

    early_stopper = EarlyStopping(patience=5, min_delta=1e-4)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if is_main:
            print(f"=== Epoch: {epoch+1}/{args.epochs} ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, is_main)
        val_loss   = validate_one_epoch(model, val_loader, device, is_main)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if is_main:
            msg = f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.8f}"
            print(msg); logging.info(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.module.state_dict(), best_model_path)
                logging.info(f"Best model saved to {best_model_path} at epoch {best_epoch}")

            # 早停判定
            early_stopper.step(val_loss)
            if early_stopper.should_stop:
                logging.info(f"Early stopping at epoch {epoch+1}")
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

    if is_main:
        torch.save(model.module.state_dict(), model_path)
        logging.info(f"Model saved to {model_path} at epoch {epoch+1}")
        logging.info(f"Training finished. Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}")

        # 画图
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure()
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, val_losses,   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve_200k.png")


if __name__ == '__main__':
    main()
