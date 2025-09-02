# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/18 14:56
@Author  : Kend
@FileName: movinet
@Software: PyCharm
@modifier:
"""

"""
不再用 3D Conv，而是 2D backbone + 1D Temporal Conv。
    输入: [B, T, C, H, W]
    输出: [B, num_classes]
    
这里又有两种方式：
    方式 1：完全替换 backbone
        直接把 MoViNet 的 3D 卷积 backbone 替换成一个标准 2D backbone（如 MobileNetV2、EfficientNet-Lite、ResNet-18 等）
        缺点：和原始 MoViNet 差别大，可能需要重新训练，精度下降稍多。
    方式 2：改造 MoViNet 自己的 backbone（推荐）
        把 conv_type="2plus1d"（3D Conv 拆成 2D Conv + 1D Conv）彻底替换成：
        2D Conv（空间卷积）
        Temporal Conv（1D 卷积） 或 TSM
        这样你仍然保留 MoViNet 的 block 设计，但避免了 RKNN 不支持的 3D 卷积。
        相当于做一个 MoViNet-lite for RKNN，既能兼容端侧，又能保持较高精度。       

"""

from collections import OrderedDict
import torch
from torch.nn.modules.utils import _triple, _pair
import torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from torch import nn, Tensor



class MoViNet2DTemporal(nn.Module):
    def __init__(self, backbone_2d, num_classes=2, feature_dim=1280, t_kernel=3):
        super().__init__()
        self.backbone = backbone_2d   # 2D 卷积骨干（可以用 MoViNet 改成 2D 或者 MobileNetV2）
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,   # 特征维度
            out_channels=feature_dim,
            kernel_size=t_kernel,
            padding=t_kernel // 2,
            groups=feature_dim  # 深度可分离卷积，轻量化
        )
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        # 1. 把帧打平，送入 2D backbone
        x = x.view(B * T, C, H, W)          # [B*T, C, H, W]
        feat = self.backbone(x)             # [B*T, D]

        # 2. reshape 回 [B, T, D]
        D = feat.shape[-1]
        feat = feat.view(B, T, D)           # [B, T, D]

        # 3. 时间卷积 (1D Conv over T)
        feat = feat.transpose(1, 2)         # [B, D, T]
        feat = self.temporal_conv(feat)     # [B, D, T]
        feat = feat.mean(dim=-1)            # [B, D]

        # 4. 分类头
        out = self.fc(feat)                 # [B, num_classes]
        return out

