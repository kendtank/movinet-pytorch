# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:29
@Author  : Kend
@FileName: transforms
@Software: PyCharm
@modifier:
"""

"""
数据增强
1. 提升模型泛化能力
    增加数据多样性，防止过拟合
    提高模型对不同光照、色彩条件的鲁棒性
2. MoViNet训练的最佳实践
    官方Kinetics训练中也使用了数据增强
    有助于提升最终的准确率
"""


"""
视频帧预处理 - 等比例缩放 + 填充 + 数据增强
适用于小数据集 + 端侧部署
"""


import cv2
import torch
import numpy as np


class VideoTransformPad:
    def __init__(self, is_train=True, resize=224):
        """
        :param is_train: 是否训练阶段，训练阶段启用数据增强
        :param resize: 输出尺寸 H=W=resize
        """
        self.is_train = is_train
        self.resize = resize

        # ImageNet标准化
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, frame):
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # 等比例缩放
        scale = self.resize / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # 填充到 resize x resize
        top_pad = (self.resize - new_h) // 2
        bottom_pad = self.resize - new_h - top_pad
        left_pad = (self.resize - new_w) // 2
        right_pad = self.resize - new_w - left_pad
        frame_padded = cv2.copyMakeBorder(frame_resized, top_pad, bottom_pad, left_pad, right_pad,
                                          borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

        frame = frame_padded

        # 数据增强（训练阶段）
        if self.is_train:
            # 随机水平翻转
            if torch.rand(1).item() < 0.5:
                frame = cv2.flip(frame, 1)
            # 随机颜色抖动
            frame = frame.astype('float32') / 255.0
            factor = (torch.rand(3) - 0.5) * 0.4 + 1.0
            frame = (frame * factor.numpy()).clip(0,1)
            frame = (frame * 255).astype('uint8')

        # 转 Tensor & 归一化
        frame = torch.tensor(frame).permute(2,0,1).float() / 255.0
        frame = (frame - self.mean) / self.std

        return frame


"""
总结
建议进行适度的数据增强，重点是：
    保持时间一致性（对所有帧使用相同的增强参数）
    专注于颜色和光照变化的增强
    避免破坏动作语义的几何变换
    根据具体任务调整增强强度
    这样既能提升模型性能，又不会破坏视频数据的时间连续性特征。
"""