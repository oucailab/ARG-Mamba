import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

# 目标注意力块：基于类别中心的上下文建模，增强类别特征区分度
class _ObjectAttentionBlock(nn.Module):
    '''
    A simplified version of the Object-Contextual Representation block.
    '''
    def __init__(self, in_channels, key_channels):
        super(_ObjectAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        )
        self.f_value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        
        # proxy is the class center features from the coarse segmentation
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        
        key = self.f_key(proxy).view(batch_size, self.key_channels, -1)
        
        value = self.f_value(proxy).view(batch_size, self.in_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.in_channels, *x.size()[2:])
        
        return context

# 类别上下文模块：利用粗分割结果生成类别中心，引导细粒度特征学习（训练时启用）
class ClassContextModule(nn.Module):
    def __init__(self, in_channels, key_channels, num_classes):
        super(ClassContextModule, self).__init__()
        self.object_context_block = _ObjectAttentionBlock(in_channels, key_channels)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x, coarse_pred):
        # coarse_pred: (B, C, H, W) - a coarse segmentation map (logits)
        # x: (B, D, H, W) - feature map from the decoder
        
        batch_size, _, h, w = coarse_pred.shape
        
        # 1. Generate class centers (proxy features) from the coarse prediction
        coarse_pred_softmax = F.softmax(coarse_pred, dim=1)
        coarse_pred_flat = coarse_pred_softmax.view(batch_size, coarse_pred.size(1), -1) # (B, C, H*W)
        
        x_flat = x.view(batch_size, x.size(1), -1) # (B, D, H*W)
        x_flat = x_flat.permute(0, 2, 1) # (B, H*W, D)
        
        # Calculate class centers by weighted average
        proxy_features = torch.matmul(coarse_pred_flat, x_flat) # (B, C, D)
        proxy_features = proxy_features.permute(0, 2, 1).unsqueeze(3) # (B, D, C, 1)
        
        # 2. Calculate object-contextual representation
        context = self.object_context_block(x, proxy_features)
        
        # 3. Fuse original features with context
        output = self.conv_bn_dropout(torch.cat([x, context], dim=1))
        
        return output

# 边界细化模块：专门优化分割边界，通过边界检测分支引导主分割分支细化
class BoundaryRefinementModule(nn.Module):
    """专门优化分割边界的模块"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 边界检测分支
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(num_classes, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 边界引导的细化分支
        self.refine_conv = nn.Sequential(
            nn.Conv2d(num_classes + 1, 64, 3, padding=1, bias=False),  # +1 for boundary
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, coarse_pred):
        # 检测边界
        boundary = self.boundary_conv(coarse_pred.detach()) # 使用 detach 避免影响主干的梯度
        
        # 使用边界信息细化预测
        refined_residual = self.refine_conv(torch.cat([coarse_pred, boundary], dim=1))
        
        # 在边界区域加强细化
        weight = 1 + 2 * boundary
        final_pred = coarse_pred + refined_residual * weight
        
        return final_pred, boundary

# 空洞空间金字塔池化：瓶颈层多尺度特征提取，通过不同膨胀率的卷积捕捉多尺度上下文
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()
        for r in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True)
                )
            )
        # image-level pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [conv(x) for conv in self.convs]
        pool = F.interpolate(self.image_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        feats.append(pool)
        x = torch.cat(feats, dim=1)
        return self.project(x)

# 结构化随机失活，缓解过拟合，保留特征图结构
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor

# 输入投影层，将原始 RGB / 深度图像转换为模型所需的特征图（Conv-BN-SiLU 组合）
class Stem(nn.Module):
    """Stem: Conv-BN-SiLU -> Conv-BN-SiLU，输出为 (B,H,W,C)。"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(out_channels // 2, 16)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return rearrange(x, 'b c h w -> b h w c')

# Visual State Transform
class VST(nn.Module):
    """视觉状态空间块（单模态）：Mamba + FFN 残差。"""
    def __init__(self, dim: int, drop_path: float = 0.0, ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim)
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        seq = rearrange(x, 'b h w d -> b (h w) d')
        seq = self.norm1(seq)
        seq = self.mamba(seq)
        seq = rearrange(seq, 'b (h w) d -> b h w d', h=h, w=w)
        x = x + self.drop_path1(seq)

        seq2 = rearrange(x, 'b h w d -> b (h w) d')
        seq2 = self.norm2(seq2)
        seq2 = self.ffn(seq2)
        seq2 = rearrange(seq2, 'b (h w) d -> b h w d', h=h, w=w)
        x = x + self.drop_path2(seq2)
        return x

# 多尺度窗口选择扫描：多尺度窗口 Mamba 组合，对不同尺寸窗口的特征分别建模后融合
class MSW_Mamba2D(nn.Module):
    def __init__(self, dim: int, window_sizes: Iterable[int]):
        super().__init__()
        self.window_sizes = tuple(sorted(window_sizes, reverse=True))
        self.mambas = nn.ModuleList([Mamba(d_model=dim) for _ in self.window_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        x_ch = rearrange(x, 'b h w d -> b d h w')
        outputs = []
        for ws, m in zip(self.window_sizes, self.mambas):
            if ws > 1:
                pooled = F.avg_pool2d(x_ch, kernel_size=ws, stride=ws, ceil_mode=True)
                ph, pw = pooled.shape[-2:]
                seq = rearrange(pooled, 'b d ph pw -> b (ph pw) d')
                seq = m(seq)
                feat = rearrange(seq, 'b (ph pw) d -> b d ph pw', ph=ph, pw=pw)
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            else:
                seq = rearrange(x, 'b h w d -> b (h w) d')
                seq = m(seq)
                feat = rearrange(seq, 'b (h w) d -> b d h w', h=h, w=w)  # 统一为 B,C,H,W
            outputs.append(feat)
        out = torch.stack(outputs, dim=0).mean(dim=0)
        return rearrange(out, 'b d h w -> b h w d')

# Local Boost State Module：增强局部特征表达，结合深度卷积和多尺度窗口 Mamba（MSW_Mamba2D）
class LBSM(nn.Module):
    """Local Emphatic State Space Module."""
    def __init__(self, dim: int, window_sizes: Tuple[int, ...] = (1, 2, 4, 8), drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.pre = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True)
        )
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm_dw = nn.BatchNorm2d(dim)
        self.msw = MSW_Mamba2D(dim, window_sizes)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm1(x)
        x_norm = self.pre(x_norm)
        x_norm = rearrange(x_norm, 'b h w d -> b d h w')
        x_norm = self.norm_dw(self.dw(x_norm))
        x_norm = rearrange(x_norm, 'b d h w -> b h w d')
        x_norm = self.msw(x_norm)
        x = shortcut + self.drop_path(x_norm)

        x_ffn = self.ffn(self.norm2(x))
        return x + self.drop_path(x_ffn)

# Cross-Modal Edge Pooling:计算RGB和深度的二阶统计信息，再结合深度图的边缘先验，生成模态权重
class CMEP(nn.Module):
    """跨模态二阶池化 + 深度边缘先验。"""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, dim // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.SiLU(inplace=True)
        )
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.s_conv_r = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.s_conv_d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def _grad(self, depth_feat: torch.Tensor) -> torch.Tensor:
        depth_map = depth_feat.mean(dim=-1, keepdim=True)
        depth_map = rearrange(depth_map, 'b h w c -> b c h w')
        gx = F.conv2d(depth_map, self.sobel_x, padding=1)
        gy = F.conv2d(depth_map, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return grad

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor):
        rgb = rearrange(rgb_feat, 'b h w d -> b d h w')
        depth = rearrange(depth_feat, 'b h w d -> b d h w')
        rgb = self.pool(self.conv(rgb))
        depth = self.pool(self.conv(depth))

        depth_edge = F.interpolate(self._grad(depth_feat), size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        depth_edge = self.edge_proj(depth_edge)

        b, c, h, w = rgb.shape
        rgb_flat = rearrange(rgb, 'b c h w -> b (h w) c')
        depth_flat = rearrange(depth, 'b c h w -> b (h w) c')
        edge_flat = rearrange(depth_edge, 'b c h w -> b (h w) c')

        M = torch.matmul(rgb_flat, depth_flat.transpose(1, 2)) / self.tau.clamp_min(1e-4)
        M = M + torch.matmul(edge_flat, edge_flat.transpose(1, 2))
        M = torch.tanh(M).unsqueeze(1)

        S_r = torch.sigmoid(self.s_conv_r(M)).squeeze(1)
        S_d = torch.sigmoid(self.s_conv_d(M)).squeeze(1)
        D_r = 1 - S_r
        D_d = 1 - S_d
        return S_r, S_d, D_r, D_d

# Selective Cross-Modal Interaction:基于权重，门控式地融合跨模态上下文
class SCMI(nn.Module):
    """选择性交互模块，带置信门控。"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_rgb = Mamba(d_model=dim)
        self.mamba_depth = Mamba(d_model=dim)
        self.gate_rgb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_depth = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat, D_r, D_d):
        b, h, w, d = rgb_feat.shape
        rgb_seq = rearrange(rgb_feat, 'b h w d -> b (h w) d')
        depth_seq = rearrange(depth_feat, 'b h w d -> b (h w) d')
        rgb_ctx = self.mamba_rgb(self.norm(depth_seq))
        depth_ctx = self.mamba_depth(self.norm(rgb_seq))

        side = int(math.isqrt(rgb_seq.shape[1]))
        D_r_up = F.interpolate(D_r.unsqueeze(1), size=(side, side), mode='nearest')
        D_d_up = F.interpolate(D_d.unsqueeze(1), size=(side, side), mode='nearest')
        g_r = rearrange(D_r_up, 'b 1 h w -> b (h w) 1')
        g_d = rearrange(D_d_up, 'b 1 h w -> b (h w) 1')

        rgb_ctx = rgb_ctx * g_r * self.gate_rgb(rgb_ctx)
        depth_ctx = depth_ctx * g_d * self.gate_depth(depth_ctx)

        rgb_ctx = rearrange(rgb_ctx, 'b (h w) d -> b h w d', h=side, w=side)
        depth_ctx = rearrange(depth_ctx, 'b (h w) d -> b h w d', h=side, w=side)

        if side != h:
            rgb_ctx = F.interpolate(rearrange(rgb_ctx, 'b h w d -> b d h w'),
                                    size=(h, w), mode='bilinear', align_corners=False)  # False
            rgb_ctx = rearrange(rgb_ctx, 'b d h w -> b h w d')
            depth_ctx = F.interpolate(rearrange(depth_ctx, 'b h w d -> b d h w'),
                                      size=(h, w), mode='bilinear', align_corners=False)  # False
            depth_ctx = rearrange(depth_ctx, 'b d h w -> b h w d')

        return rgb_feat + rgb_ctx, depth_feat + depth_ctx

# Selective Feature Boost:增强模态互补性
class SFB(nn.Module):
    """选择性增强模块：融合 RGB/深度增强特征，并输出模态置信。"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.mamba = Mamba(d_model=dim * 2)
        self.conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)
        self.confidence = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor, S_r: torch.Tensor, S_d: torch.Tensor):
        b, h, w, d = rgb_feat.shape
        # 先把低分辨率注意图上采样到 (h, w)
        S_r = F.interpolate(S_r.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
        S_d = F.interpolate(S_d.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        rgb_att = rgb_feat * S_r.unsqueeze(-1)
        depth_att = depth_feat * S_d.unsqueeze(-1)
        fused = torch.cat([rgb_att, depth_att], dim=-1)
        side = int(math.isqrt(fused.shape[1] if fused.ndim == 3 else h * w))

        seq = rearrange(fused, 'b h w c -> b (h w) c')
        seq = self.mamba(self.norm(seq))
        fused_feat = rearrange(seq, 'b (h w) c -> b h w c', h=h, w=w)

        fused_feat = rearrange(fused_feat, 'b h w c -> b c h w')
        conf_input = torch.cat([
            rearrange(rgb_feat, 'b h w d -> b d h w'),
            rearrange(depth_feat, 'b h w d -> b d h w')
        ], dim=1)
        conf = self.confidence(conf_input)
        fused_feat = self.conv(fused_feat)
        fused_feat = fused_feat * conf + conf_input[:, :fused_feat.shape[1]] * (1 - conf)
        return rearrange(fused_feat, 'b c h w -> b h w c')

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        
        self.out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

    def forward(self, rgb, depth):
        B, H, W, D = rgb.shape
        ws = self.window_size

        # 1. 动态计算需要填充的大小
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        
        # 对特征图进行右侧和底侧填充
        # (B, H, W, D) -> (B, D, H, W) for padding
        rgb_padded = F.pad(rearrange(rgb, 'b h w d -> b d h w'), (0, pad_w, 0, pad_h))
        depth_padded = F.pad(rearrange(depth, 'b h w d -> b d h w'), (0, pad_w, 0, pad_h))
        
        # 获取填充后的新尺寸
        _B, _D, H_pad, W_pad = rgb_padded.shape
        
        # 转换回 (B, H, W, D) 格式
        rgb_padded = rearrange(rgb_padded, 'b d h w -> b h w d')
        depth_padded = rearrange(depth_padded, 'b d h w -> b h w d')

        # 2. 将填充后的特征图划分为窗口
        rgb_windows = rearrange(rgb_padded, 'b (h ws1) (w ws2) d -> (b h w) (ws1 ws2) d', ws1=ws, ws2=ws)
        dep_windows = rearrange(depth_padded, 'b (h ws1) (w ws2) d -> (b h w) (ws1 ws2) d', ws1=ws, ws2=ws)

        # 3. 在窗口内计算 Q, K, V 和注意力
        q = self.q(rgb_windows).view(-1, ws*ws, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k(dep_windows).view(-1, ws*ws, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v(dep_windows).view(-1, ws*ws, self.num_heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)
        
        out = (att @ v).transpose(1,2).contiguous().view(-1, ws*ws, self.dim)
        
        # 4. 将窗口合并回填充后的特征图形状
        out = rearrange(out, '(b h w) (ws1 ws2) d -> b (h ws1) (w ws2) d', h=H_pad//ws, w=W_pad//ws, ws1=ws, ws2=ws)
        
        # 5. 裁剪掉填充的部分，恢复原始尺寸
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]
        
        out = self.out(out)
        
        # residual
        return rgb + out, depth + out

class ADMF(nn.Module):
    """Adaptive Dual-Modal Fusion."""
    def __init__(self, dim: int):
        super().__init__()
        self.csop = CMEP(dim)
        self.sim = SCMI(dim)
        self.sem = SFB(dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat):
        if rgb_feat.shape[1:3] != depth_feat.shape[1:3]:
            depth_feat = F.interpolate(
                rearrange(depth_feat, 'b h w d -> b d h w'),
                size=rgb_feat.shape[1:3],
                mode='bilinear',
                align_corners=False  # False
            )
            depth_feat = rearrange(depth_feat, 'b d h w -> b h w d')

        S_r, S_d, D_r, D_d = self.csop(rgb_feat, depth_feat)
        rgb_enh, depth_enh = self.sim(rgb_feat, depth_feat, D_r, D_d)
        fused = self.sem(rgb_enh, depth_enh, S_r, S_d)
        gate = self.fusion_gate(torch.cat([rgb_enh, depth_enh, fused], dim=-1))
        return fused * gate + rgb_enh * (1 - gate)


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv(x)
        return rearrange(x, 'b c h w -> b h w c')


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class DecoderBlock(nn.Module):
    """Up-sample + CBAM refine（参照论文图中的 DB）。"""
    def __init__(self, in_dim: int, skip_dim: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, skip_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(skip_dim),
            nn.SiLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(skip_dim * 2, skip_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(skip_dim, skip_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_dim),
            nn.SiLU(inplace=True)
        )
        self.cbam = CBAM(skip_dim)

    def forward(self, x, skip):
        x = rearrange(x, 'b h w c -> b c h w')
        skip = rearrange(skip, 'b h w c -> b c h w')
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)  # False
        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        x = self.cbam(x)
        return rearrange(x, 'b c h w -> b h w c')


class Head(nn.Module):
    """Segmentation head，输出 (B,C,H,W)。"""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        mid = max(in_dim // 2, 32)
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, num_classes, kernel_size=1)
        )

    def forward(self, x, out_size: Tuple[int, int] = None):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.proj(x)
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)  # False
        return x


class ARGMamba(nn.Module):
    """ARGMamba segmentation backbone，支持 RGB-D。"""
    def __init__(self,
                 num_classes: int,
                 dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
                 drop_path_rate: float = 0.1,
                 window_sizes: Tuple[int, ...] = (1, 2, 4, 8),
                 deep_supervision: bool = False,
                 aux_tasks: bool = False,
                 boundary_refine: bool = False,
                 use_context_module: bool = False):  # 新增
        super().__init__()
        self.num_classes = num_classes
        self.dims = dims
        self.deep_supervision = deep_supervision  # 新增
        self.aux_tasks = aux_tasks
        self.boundary_refine = boundary_refine # 新增
        self.use_context_module = use_context_module

        self.stem_rgb = Stem(3, dims[0])
        self.stem_depth = Stem(1, dims[0])

        dpr = torch.linspace(0, drop_path_rate, steps=len(dims) * 2).tolist()
        self.encoder_stages = nn.ModuleList()
        for i, dim in enumerate(dims):
            stage = nn.ModuleDict({
                "vssb_rgb": VST(dim, drop_path=dpr[2 * i]),
                "vssb_depth": VST(dim, drop_path=dpr[2 * i + 1]),
                "le_rgb": LBSM(dim, window_sizes=window_sizes, drop_path=dpr[2 * i]),
                "le_depth": LBSM(dim, window_sizes=window_sizes, drop_path=dpr[2 * i + 1]),
                "cross_attn": CrossModalAttention(dim),
                "afm": ADMF(dim),
                "downsample": Downsample(dim, dims[i + 1]) if i < len(dims) - 1 else None
            })
            self.encoder_stages.append(stage)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dims[3], dims[2]),
            DecoderBlock(dims[2], dims[1]),
            DecoderBlock(dims[1], dims[0])
        ])
        self.aspp = ASPP(dims[3], dims[3])          # 加在 bottleneck
        self.head_top = Head(dims[3], num_classes)
        self.head3 = Head(dims[2], num_classes)
        self.head2 = Head(dims[1], num_classes)
        self.head1 = Head(dims[0], num_classes)

        # 新增：如果启用类别上下文模块，则初始化
        if self.use_context_module:
            self.context_module = ClassContextModule(in_channels=dims[0], key_channels=dims[0]//2, num_classes=num_classes)
            # 创建一个额外的head用于生成粗略预测
            self.coarse_head = Head(dims[0], num_classes)


        if self.boundary_refine:
            self.br_module = BoundaryRefinementModule(num_classes)

        # 新增：如果启用辅助任务，则初始化对应的预测头
        if self.aux_tasks:
            # 辅助任务1：边缘检测头
            self.edge_head = nn.Sequential(
                nn.Conv2d(dims[0], 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            )
            # 辅助任务2：深度估计头
            self.depth_head = nn.Sequential(
                nn.Conv2d(dims[0], 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor = None):
        if depth is None:
            depth = torch.zeros(rgb.size(0), 1, rgb.size(2), rgb.size(3), device=rgb.device, dtype=rgb.dtype)

        x_rgb = self.stem_rgb(rgb)
        x_depth = self.stem_depth(depth)

        fused_feats = []
        for stage in self.encoder_stages:
            x_rgb = stage["vssb_rgb"](x_rgb)
            x_depth = stage["vssb_depth"](x_depth)
            x_rgb = stage["le_rgb"](x_rgb)
            x_depth = stage["le_depth"](x_depth)
            x_rgb, x_depth = stage["cross_attn"](x_rgb, x_depth)
            fused = stage["afm"](x_rgb, x_depth)
            fused_feats.append(fused)
            if stage["downsample"] is not None:
                downsampled_feat = stage["downsample"](fused)
                x_rgb = downsampled_feat
                x_depth = downsampled_feat

        size_in = rgb.shape[-2:]
        bottleneck_input = rearrange(fused_feats[-1], 'b h w c -> b c h w')
        bottleneck = self.aspp(bottleneck_input)
        bottleneck = rearrange(bottleneck, 'b c h w -> b h w c')
        p4 = self.head_top(fused_feats[-1], out_size=size_in)
        x = bottleneck 

        x = self.decoder_blocks[0](x, fused_feats[-2])
        p3 = self.head3(x, out_size=size_in)

        x = self.decoder_blocks[1](x, fused_feats[-3])
        p2 = self.head2(x, out_size=size_in)

        x = self.decoder_blocks[2](x, fused_feats[-4])
        # p1 = self.head1(x, out_size=size_in)

        if self.use_context_module and self.training: # 通常只在训练时使用以增加监督
            x_rearranged = rearrange(x, 'b h w c -> b c h w')
            # 1. 生成一个粗略的分割图
            coarse_p1 = self.coarse_head(x, out_size=x.shape[1:3])
            # 2. 使用粗略分割图和特征图来计算上下文特征
            x_context = self.context_module(x_rearranged, coarse_p1)
            # 3. 将上下文特征转换回 (B,H,W,C) 并与原始特征融合
            x_context = rearrange(x_context, 'b c h w -> b h w c')
            x = x + x_context # 使用残差连接

        p1 = self.head1(x, out_size=size_in)

        # 新增：如果启用了边界细化，则对最高分辨率的预测进行处理
        if self.boundary_refine:
            p1_refined, boundary_pred = self.br_module(p1)
            # 在深度监督模式下，用细化后的结果替换p1和final
            if self.deep_supervision:
                p1 = p1_refined
            else: # 非深度监督模式下，直接将细化结果作为输出
                p1 = p1_refined

        # 组织主任务的输出
        if self.deep_supervision:
            outputs = {
                "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                "final": p1
            }
        else:
            outputs = p1  # 只返回最高分辨率的输出

        # 如果在训练模式且启用了辅助任务，则计算并返回辅助输出
        if self.training and self.aux_tasks:
            # 从最高分辨率的解码器特征 x 生成辅助任务预测
            # x 的形状是 (B, H/2, W/2, C)，需要转换为 (B, C, H/2, W/2)
            aux_feat = rearrange(x, 'b h w c -> b c h w')
            edge_pred = self.edge_head(aux_feat)
            depth_pred = self.depth_head(aux_feat)
            
            # 将辅助预测上采样到输入尺寸
            edge_pred = F.interpolate(edge_pred, size=size_in, mode='bilinear', align_corners=False)
            depth_pred = F.interpolate(depth_pred, size=size_in, mode='bilinear', align_corners=False)
            
            # 如果同时启用了边界细化，可以考虑也返回边界预测用于监督
            if self.boundary_refine:
                 return outputs, edge_pred, depth_pred, boundary_pred

            return outputs, edge_pred, depth_pred
        
        # 否则，只返回主任务的输出
        return outputs
        
        


if __name__ == "__main__":
    device = torch.device("cuda")
    rgb = torch.randn(1, 3, 512, 512, device=device)
    depth = torch.randn(1, 1, 512, 512, device=device)
    model = ARGMamba(num_classes=6, dims=[32, 64, 128, 256], deep_supervision=True, aux_tasks=True).to(device)
    model.train()
    # 训练时的输出
    seg_preds, edge_pred, depth_pred = model(rgb, depth)
    
    print("--- Training Outputs ---")
    for k, v in seg_preds.items():
        print(f"Seg Preds '{k}': {v.shape}")
    print(f"Edge Pred: {edge_pred.shape}")
    print(f"Depth Pred: {depth_pred.shape}")

    model.eval() # 设置为评估模式
    # 评估/推理时的输出
    seg_preds_eval = model(rgb, depth)
    print("\n--- Eval Outputs ---")
    for k, v in seg_preds_eval.items():
        print(f"Seg Preds '{k}': {v.shape}")
