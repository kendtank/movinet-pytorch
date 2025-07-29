# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 10:52
@Author  : Kend
@FileName: load_dataset_with_video
@Software: PyCharm
@modifier: 采用官方建议的方式训练: https://github.com/Atze00/MoViNet-pytorch
"""

"""
官方建议的方法：
    分段处理（Streaming）：
    将视频分为 n_clips 个片段
    每个片段包含 n_clip_frames 帧
    逐段输入模型处理
    流式缓冲：使用 model.clean_activation_buffers() 清理激活缓冲区
    梯度累积：对每个片段分别计算损失并反向传播，最后统一优化
为MoViNet优化的数据集加载器，支持流式处理和分段训练
"""



import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms


class StreamingVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_frames=16):
        """
        :param root_dir: 数据集视频目录
        :param transform: 视频帧的图像增强方法
        :param clip_frames: 每个clip包含的帧数
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clip_frames = clip_frames

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
        """
        返回视频路径和标签，实际帧加载在训练时进行
        """
        video_path, label = self.samples[idx]
        return video_path, label


def load_video_clip(video_path, start_frame, num_frames, transform=None):
    """
    从视频中加载指定起始位置的帧序列
    :param video_path: 视频文件路径
    :param start_frame: 起始帧索引
    :param num_frames: 需要加载的帧数
    :param transform: 图像变换
    :return: 处理后的帧张量 (C, T, H, W)
    """

    cap = cv2.VideoCapture(video_path)

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # 如果没有更多帧，使用最后一帧填充
            if frames:
                frame = frames[-1]
            else:
                break

        if frame is not None:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if not frames:
        # 如果没有读取到任何帧，返回零张量
        return torch.zeros(3, num_frames, 224, 224)

    # 对帧进行变换
    if transform:
        frames = [transform(frame) for frame in frames]
    else:
        # 默认转换为tensor并归一化
        frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]

    frames = torch.stack(frames)  # (T, C, H, W)
    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    return frames



class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=256):
        """
        :param root_dir: 数据集视频目录
        :param transform: 视频帧的图像增强方法
        :param max_frames: 从数据集视频中裁剪出来的最大帧数
        """
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
        # 这里使用的方式是连续的抽帧，而不是间隔抽帧
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



