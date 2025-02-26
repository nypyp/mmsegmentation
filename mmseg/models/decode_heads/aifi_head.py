"""
AIFI Head implementation.

Reference:
    [paper name/link]
"""

# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Transformer modules."""

import math

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        
        # 只在attention计算前下采样
        x_small = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        # 计算attention
        pos_embed = self.build_2d_sincos_position_embedding(w//2, h//2, c)
        x_attn = super().forward(x_small.flatten(2).permute(0, 2, 1), 
                              pos=pos_embed.to(device=x.device, dtype=x.dtype))
        # 恢复原始分辨率
        x_attn = x_attn.permute(0, 2, 1).view([-1, c, h//2, w//2])
        x_out = F.interpolate(x_attn, size=(h, w), mode='bilinear', align_corners=False)
        
        return x_out

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class Spatial_Att(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.bn2 = nn.BatchNorm2d(feat_dim, affine=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, 1, c).contiguous()
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 3, 2, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.reshape(b, c, h, w).contiguous()
        return torch.sigmoid(x) * residual

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return torch.sigmoid(x) * residual

class NAMAttention(nn.Module):
    def __init__(self, channels, no_spatial: bool = False):
        super().__init__()
        self.channel_att = Channel_Att(channels)
        self.spatial_att = None
        self.no_spatial = no_spatial

    def forward(self, x):
        x = self.channel_att(x)
        if self.no_spatial is False:
            _, _, h, w = x.shape
            self.spatial_att = Spatial_Att(h * w).to(x.device)
            x = self.spatial_att(x)
        return x

@MODELS.register_module()
class AIFIHead(BaseDecodeHead):
    """AIFI Decoder Head.
    
    This head uses transformer-based attention to enhance feature representations.

    Args:
        c1_in_channels (int): Number of channels in the low-level feature map.
        c1_channels (int): Transform channels of low-level features.
        transformer_channels (int): Number of channels in transformer module.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout rate. Default: 0.1.
    """

    def __init__(self,
                 c1_in_channels: int,
                 c1_channels: int,
                 transformer_channels: int = 2048,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # 初始化NAM注意力模块（通道数匹配concat后的维度）
        self.nam_attention = NAMAttention(channels=self.channels * 2 + self.in_channels)

        # Main branch
        self.lateral_conv = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
            
        # AIFI transformer
        self.aifi = AIFI(
            c1=self.in_channels,
            cm=transformer_channels,
            num_heads=num_heads)

        # Global pooling branch  
        self.global_conv = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Feature fusion
        self.fusion_conv = ConvModule(
            self.channels * 2 + self.in_channels,
            self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Low-level feature transform
        self.c1_transform = ConvModule(
            c1_in_channels,
            c1_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Final fusion
        self.final_conv = nn.Sequential(
            ConvModule(
                c1_channels + self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
        )

    def _forward_feature(self, inputs: List[Tensor]) -> Tensor:
        """Forward function for feature computation."""
        x = self._transform_inputs(inputs)
        
        # Get input shape
        row, col = x.shape[2:]
        
        # Main branch
        feat_main = self.lateral_conv(x)
        
        # AIFI branch
        feat_aifi = self.aifi(x)
        
        # Global context branch
        feat_pool = F.adaptive_avg_pool2d(x, 1)  # 形状 [B, C, 1, 1]
        feat_conv = self.global_conv(feat_pool)  # 卷积操作在1x1特征图上进行
        feat_global = F.interpolate(feat_conv, (row, col), mode='bilinear', align_corners=True)  # 显式上采样

        # Feature fusion
        cat_feat = torch.cat([feat_main, feat_aifi, feat_global], dim=1)
        cat_feat = self.nam_attention(cat_feat)
        feat_fused = self.fusion_conv(cat_feat)

        return feat_fused

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Forward function."""
        output = self._forward_feature(inputs)
        
        # Transform and fuse low-level features
        low_level_feat = self.c1_transform(inputs[0])
        
        # Resize and concatenate features
        output = F.interpolate(
            output,
            size=low_level_feat.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        output = self.final_conv(
            torch.cat([low_level_feat, output], dim=1))
            
        # Final prediction
        output = self.cls_seg(output)
        
        
        return output
