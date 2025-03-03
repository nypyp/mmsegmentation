# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, MaxPool2d

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

__all__ = ['HGNetV2']

kaiming_normal_ = kaiming_normal_
zeros_ = constant_
ones_ = constant_


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 use_act=True,
                 use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding
            if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            bias=False)
        self.bn = BatchNorm2d(out_channels)
        if self.use_act:
            self.act = ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab)
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab)
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="same",
            use_lab=use_lab)
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="same",
            use_lab=use_lab)
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab)
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab)
        self.pool = MaxPool2d(
            kernel_size=2, stride=1, ceil_mode=True, padding="same")

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], 1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


class HG_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 layer_num=6,
                 identity=False,
                 light_block=True,
                 use_lab=False):
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        block_type = LightConvBNAct if light_block else ConvBNAct
        for i in range(layer_num):
            self.layers.append(
                block_type(in_channels=in_channels
                           if i == 0 else mid_channels,
                           out_channels=mid_channels,
                           stride=1,
                           kernel_size=kernel_size,
                           use_lab=use_lab))
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab)
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, axis=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x += identity
        return x


class HG_Stage(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num=6,
                 downsample=True,
                 light_block=True,
                 kernel_size=3,
                 use_lab=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab)

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


@MODELS.register_module()
class HGNetV2(BaseModule):
    """HGNetV2 backbone.

    Args:
        arch (str): Architecture of HGNetV2, from {'S', 'M', 'L', 'X', 'H'}.
        use_lab (bool): Whether to use LearnableAffineBlock. Default: False.
        return_idx (list): Stages to return. Default: [1, 2, 3].
    """

    arch_configs = {
        'S': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            }
        },
        'M': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            }
        },
        'L': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, False, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, False, True, 5, 6],
            }
        },
        'X': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            }
        },
        'H': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            }
        }
    }

    def __init__(self,
                 arch,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 norm_cfg=dict(type='BN', requires_grad=True),
                 **kwargs):
        super().__init__(**kwargs)
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[arch]['stem_channels']
        stage_config = self.arch_configs[arch]['stage_config']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
