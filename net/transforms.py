# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:29
@Author  : Kend
@FileName: transforms
@Software: PyCharm
@modifier:
"""


"""数据增强方案"""

import torch
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class RandomResizeCrop:
    def __init__(self, size=(224, 224), scale=(0.5, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        scale = random.uniform(*self.scale)
        new_h = int(img.shape[1] * scale)
        new_w = int(img.shape[2] * scale)
        img = F.resize(img, (new_h, new_w))
        img = F.center_crop(img, self.size)
        return img


class ToTensor:
    def __call__(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0



# 推荐数据增强组合
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ]),
    transforms.RandomApply([
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0))
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
