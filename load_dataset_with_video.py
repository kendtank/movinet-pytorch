# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 10:52
@Author  : Kend
@FileName: load_dataset_with_video
@Software: PyCharm
@modifier:
"""

"""
不需要严格等比缩放。
视频分类任务更关注语义信息，而不是精确的长宽比。
cv2.resize() 使用默认的插值方式（INTER_LINEAR）即可，可以保证输入大小统一。
MoViNet 的预训练模型使用了 224x224 输入，保持这个尺寸即可。
"""


import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=256):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            for video in os.listdir(cls_folder):
                self.samples.append((os.path.join(cls_folder, video), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)

        # 对每一帧分别做 transform
        if self.transform:
            frames = [self.transform(frame) for frame in frames]  # List of (C, H, W)
            frames = torch.stack(frames)  # (T, C, H, W)

        # 调整为 MoViNet 输入格式 (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # 补帧
        while len(frames) < self.max_frames:
            frames.append(frames[-1])

        # 返回 List[np.ndarray]，每个元素是 (H, W, C)
        return frames



