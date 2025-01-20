import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch.nn.init import constant_, xavier_uniform_
from .decode_head import BaseDecodeHead


class NAMAttention(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True,h_w=512*512):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.Spatial_Att = Spatial_Att(h_w)
  
    def forward(self, x):
        #print(x.size())
        x_out1=self.Channel_Att(x)
        if not self.no_spatial:
            x_out1=self.Spatial_Att(x_out1)
        return x_out1

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
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
        """Add position embeddings if given."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
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


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads,cout=None):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        if cout is None:
            cout=c
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=cout, num_heads=num_heads)
        self.fc1 = nn.Linear(cout, cout, bias=False)
        self.fc2 = nn.Linear(cout, cout, bias=False)

    def forward(self, x):
        print(x.shape)
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        print(x.shape)
        return self.fc2(self.fc1(x)) + x
    
@MODELS.register_module()
class AIFIHead(BaseDecodeHead):
    def __init__(self, *args,dim_in=2048,dim_out=1024,bn_mom=0.1,rate=1,**kwargs):
        super().__init__(**kwargs)
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.__AIFI=AIFI(dim_in,1024,8)
        self.__tflayer=TransformerLayer(dim_in,8,256*3)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            # TransformerLayer(dim_out*6, 8, dim_out),
            nn.Conv2d(dim_out*2+dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # self.att = NAMAttention(dim_out*2+dim_in,no_spatial=True,h_w=1024)
        #self.att = ACmix(736,736)
        #self.att = GAMAttention(688,688)
        #self.strpool = StripPooling(dim_in,(16,16),nn.BatchNorm2d,{'mode': 'bilinear', 'align_corners': True})

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        [b, c, row, col] = x.size()
        x0=self.branch1(x)
        x1=self.__AIFI(x)
        #print(row,col)
        # x1=self.__tflayer(x1)
        #global_feature = self.strpool(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([x0,x1, global_feature], dim=1)
        # feature_cat = self.att(feature_cat)
        result = self.conv_cat(feature_cat)
        return result
    
    def forward(self, x):
        output = self._forward_feature(x)
        output = self.cls_seg(output)
        return output