"""
AIFI Head implementation.

Reference:
    [paper name/link]
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


class SpatialAttention(BaseModule):
    """Spatial attention module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 1, self.channels)
        x = self.bn(x)
        
        weights = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
        x = x.permute(0, 3, 2, 1)
        x = torch.mul(weights, x)
        x = x.reshape(identity.shape)
        
        return torch.sigmoid(x) * identity


class ChannelAttention(BaseModule):
    """Channel attention module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.bn(x)
        
        weights = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
        x = x.permute(0, 2, 3, 1)
        x = torch.mul(weights, x)
        x = x.permute(0, 3, 1, 2)
        
        return torch.sigmoid(x) * identity


class NAMAttention(BaseModule):
    """Non-local Adaptive Module Attention."""
    
    def __init__(self,
                 channels: int,
                 out_channels: Optional[int] = None,
                 use_spatial: bool = True) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial_attention = SpatialAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_attention(x)
        if self.use_spatial:
            out = self.spatial_attention(out)
        return out


class TransformerEncoderLayer(BaseModule):
    """Transformer encoder layer."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 dropout: float = 0.0,
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
            
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.activation = nn.GELU()

    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        # Self attention
        x_with_pos = x if pos is None else x + pos
        attn_out = self.self_attn(
            x_with_pos, x_with_pos, value=x)[0]
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN
        ffn_out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


class AIFI(BaseModule):
    """AIFI transformer module."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 feedforward_channels: int = 2048,
                 dropout: float = 0.0,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        self.encoder = TransformerEncoderLayer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        pos_embed = self._build_position_embedding(w, h, c)
        
        # (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)
        
        x = self.encoder(x, pos=pos_embed.to(x.device, x.dtype))
        
        # (B, H*W, C) -> (B, C, H, W)
        return x.permute(0, 2, 1).reshape(b, c, h, w)

    @staticmethod
    def _build_position_embedding(w: int,
                                h: int,
                                embed_dims: int,
                                temperature: float = 10000.) -> Tensor:
        """Build 2D sinusoidal position embedding."""
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        
        assert embed_dims % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D position embedding'
            
        pos_dim = embed_dims // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([
            torch.sin(out_w), torch.cos(out_w),
            torch.sin(out_h), torch.cos(out_h)
        ], 1)[None]


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

        # Main branch
        self.lateral_conv = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
            
        # AIFI transformer
        self.aifi = AIFI(
            embed_dims=self.in_channels,
            feedforward_channels=transformer_channels,
            num_heads=num_heads,
            dropout=dropout)

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
            nn.Dropout(dropout),
            ConvModule(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Dropout(0.1))

    def _forward_feature(self, inputs: List[Tensor]) -> Tensor:
        """Forward function for feature computation."""
        x = self._transform_inputs(inputs)
        
        # Main branch
        feat_main = self.lateral_conv(x)
        
        # AIFI branch
        feat_aifi = self.aifi(x)
        
        # Global context
        feat_global = self.global_conv(
            F.adaptive_avg_pool2d(x, 1).expand_as(x))

        # Feature fusion
        feat_fused = self.fusion_conv(
            torch.cat([feat_main, feat_aifi, feat_global], dim=1))

        return feat_fused

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Forward function."""
        output = self._forward_feature(inputs)
        
        # Transform and fuse low-level features
        low_level_feat = self.c1_transform(inputs[0])
        
        # Resize and concatenate features
        output = resize(
            output,
            size=low_level_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = self.final_conv(
            torch.cat([low_level_feat, output], dim=1))
            
        # Final prediction
        output = self.cls_seg(output)
        
        return output