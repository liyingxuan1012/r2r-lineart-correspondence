import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from loftr_module.transformer import LocalFeatureTransformer


def _infer_hw_from_tokens(n_tokens: int) -> tuple[int, int]:
    """
    从 token 数推断一个 (H, W)；用于原始 ViT 位置编码的网格尺寸推断。
    （仅在初始化/插值时用，不再依赖它来推断运行时网格）
    """
    H = int(round(n_tokens ** 0.5))
    while n_tokens % H != 0:
        H -= 1
    W = n_tokens // H
    return H, W


def _resize_pos_embed_2d(pos_embed: torch.Tensor,  # (1, 1+H0*W0, C)
                         new_hw: tuple[int, int]) -> torch.Tensor:
    """
    把 ViT 的 2D 网格位置编码从 (H0,W0) 双三次插值到目标 (H1,W1)，保留 cls_token。
    """
    B, N, C = pos_embed.shape
    assert B == 1 and N >= 1
    cls_tok = pos_embed[:, :1, :]        # (1,1,C)
    grid    = pos_embed[:, 1:, :]        # (1,H0*W0,C)

    H0, W0 = _infer_hw_from_tokens(grid.shape[1])
    H1, W1 = new_hw

    grid = grid.reshape(1, H0, W0, C).permute(0, 3, 1, 2)        # (1,C,H0,W0)
    grid = F.interpolate(grid, size=(H1, W1), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, H1 * W1, C)       # (1,H1*W1,C)
    return torch.cat([cls_tok, grid], dim=1)                     # (1,1+H1*W1,C)


class LineArtTransformerModel(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 4,
                 num_patches: int = 448,   # 兼容旧接口，不依赖
                 patch_size: int = 32,     # 16 或 32
                 attention: str = 'linear',
                 loftr_attn_dropout: float = 0.1,
                 loftr_ffn_dropout: float = 0.1):
        super().__init__()

        assert patch_size in (16, 32), "patch_size must be 16 or 32"
        self.patch_size = patch_size
        self.pool_factor = 2 if patch_size == 32 else 1  # ps=32 时做 2x2 池化（32×56→16×28）

        # ---- ViT-B/16 backbone ----
        vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.conv_proj = vit_model.conv_proj        # stride=16 的 patch embed → (B,768,H/16,W/16)
        self.encoder   = vit_model.encoder          # Transformer encoder（内部会用 encoder.pos_embedding）
        self.cls_token = vit_model.class_token      # (1,1,768)

        # 保存一份“原始”位置编码用于插值；不作为持久化参数保存到 state_dict（避免多卡重复）
        self.register_buffer("_orig_pos_embed", vit_model.encoder.pos_embedding.detach().clone(),
                             persistent=False)
        # 先用原尺寸注册一个可学习参数；后续会按实际 (Hp,Wp) 重设
        self.pos_embed = nn.Parameter(self._orig_pos_embed.clone())
        # 让 encoder 使用我们这份参数
        self.encoder.pos_embedding = self.pos_embed

        # ---- LoFTR 交互层 ----
        self.loftr = LocalFeatureTransformer({
            'd_model': embed_dim,
            'nhead': num_heads,
            'layer_names': ['self', 'cross'] * num_layers,
            'attention': attention,               # 'linear' or 'full'
            'attn_dropout': loftr_attn_dropout,
            'ffn_dropout': loftr_ffn_dropout,
        })

        # 标记：下一次前向时需要按实际网格刷新 pos_embed（首次必然为 True）
        self._pos_need_update = True
        self._cached_hw = None  # 记录上次的 (Hp,Wp)，避免每步都重建参数

    # ======= 关键：按实际 (Hp,Wp) 设置 encoder.pos_embedding ======= #
    def _set_pos_embed_to_hw(self, Hp: int, Wp: int, device: torch.device):
        need_len = 1 + (Hp * Wp)
        have_len = self.pos_embed.shape[1]
        need_shape = (1, need_len, self.pos_embed.shape[-1])

        # 若尺寸完全一致，且缓存也一致，则不改动
        if have_len == need_len and self._cached_hw == (Hp, Wp) and not self._pos_need_update:
            return

        # 从 _orig_pos_embed 插值生成新位置编码参数，并替换到 encoder
        with torch.no_grad():
            new_pos = _resize_pos_embed_2d(self._orig_pos_embed, (Hp, Wp)).to(device)
        # 将 tensor 包装成可学习参数，替换引用
        self.pos_embed = nn.Parameter(new_pos)
        self.encoder.pos_embedding = self.pos_embed

        # 更新缓存与标记
        self._cached_hw = (Hp, Wp)
        self._pos_need_update = False

    # ============================================================

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x (B,3,H,W) 例如 512×896
        输出: (B, S, 768)
             - ps=16: S = 32×56 = 1792
             - ps=32: S = 16×28 = 448
        """
        # ViT-B/16 patch embed → (B,768, H/16, W/16)，如 (B,768,32,56)
        x = self.conv_proj(x)
        B, E, Hp, Wp = x.shape

        # ps=32 时对特征图做 2×2 平均池化，把 32×56 → 16×28
        if self.pool_factor == 2:
            # 使用更稳定的 reshape→mean 方式（避免 stride 问题）
            x = x.reshape(B, E, Hp // 2, 2, Wp // 2, 2).mean(dim=(3, 5))
            Hp, Wp = Hp // 2, Wp // 2

        # 展平为序列 (B, S, 768)
        x = x.flatten(2).transpose(1, 2)  # (B, S, E) 这里的 S = Hp*Wp

        # —— 用真实 (Hp,Wp) 设置 ViT 的位置编码 —— #
        # 注意：这一步只在第一次或网格变动时执行，会替换 encoder.pos_embedding 的 nn.Parameter
        self._set_pos_embed_to_hw(Hp, Wp, device=x.device)

        # 拼接 cls_token → (B,1+S,768)
        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 过 ViT encoder；内部会做 x + pos_embedding
        x = self.encoder(x)[:, 1:, :]     # 去掉 cls → (B,S,768)
        return x

    def forward(self, img_ref, img_tgt, return_feats=False):
        feat_ref = self.forward_single(img_ref)
        feat_tgt = self.forward_single(img_tgt)
        
        feat_ref, feat_tgt = self.loftr(feat_ref, feat_tgt, mask0=None, mask1=None)
        feat_ref = F.normalize(feat_ref, p=2, dim=-1)
        feat_tgt = F.normalize(feat_tgt, p=2, dim=-1)
        
        if return_feats:
            return feat_ref, feat_tgt
    
        feat_all = torch.cat([feat_ref, feat_tgt], dim=1)
        score_matrix = torch.bmm(feat_all, feat_all.transpose(1, 2))
        return score_matrix
