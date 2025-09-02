# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/18 14:56
@Author  : Kend
@FileName: block
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
              
MoViNet 默认的 block 里用的是 3D Conv 或 (2+1)D Conv。
这里要做的就是：在保持 MoViNet 主体结构的情况下，把这些 3D 卷积层替换成 2D Conv + Temporal Conv。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================
# Swish 激活函数
# ====================================================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



# ====================================================
# ConvBlock2DTemp: 2D卷积 + Temporal 1D卷积
# ====================================================
class Conv2DTemporalBlock(nn.Module):
    """
    用 2D Conv (空间卷积) + 1D Conv (时间卷积) 替代 3D Conv
    输入: x [B, C, T, H, W]
    输出: x [B, C_out, T, H_out, W_out]
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super().__init__()

        # ---- 空间卷积 (2D) ----
        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size[1:],   # (H, W)
            stride=stride[1:],
            padding=padding[1:],
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # ---- 时间卷积 (1D) ----
        self.temporal_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size[0],   # 时间维
            stride=stride[0],
            padding=padding[0],
            bias=False
        )
        self.bn_t = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # ---- 空间卷积 ----
        x = x.permute(0, 2, 1, 3, 4)        # [B, T, C, H, W]
        x = x.reshape(B*T, C, H, W)         # [B*T, C, H, W]
        x = self.relu(self.bn(self.spatial_conv(x)))  # [B*T, C_out, H', W']
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out)  # [B, T, C_out, H', W']
        x = x.permute(0, 2, 1, 3, 4)          # [B, C_out, T, H', W']

        # ---- 时间卷积 ----
        x = x.flatten(3, 4)  # [B, C_out, T, H'*W'] → [B, C_out, T, HW]
        x = x.mean(-1)       # 做空间池化 → [B, C_out, T]
        x = self.bn_t(self.temporal_conv(x))  # [B, C_out, T]
        x = x.unsqueeze(-1).unsqueeze(-1)     # [B, C_out, T, 1, 1]

        return x
