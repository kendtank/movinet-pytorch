# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/29 15:58
@Author  : Kend
@FileName: dataset
@Software: PyCharm
@modifier: 采用官方建议的方式训练: https://github.com/Atze00/MoViNet-pytorch
"""


import os
import torch
from torch.utils.data import Dataset
import cv2



class StreamingVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: 数据集视频目录
        :param transform: 视频帧的图像增强方法
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []   # 样本列表, 提供给pytorch的DataLoader

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



""" 对具体的段视频做处理, 也是减少内存的核心所在 """
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



def get_video_frame_count(video_path):
    """获取视频总帧数"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def adaptive_frame_strategy(video_paths, target_frames=128):
    """
    根据视频长度自适应调整帧处理策略

    :param video_paths: batch中视频路径列表
    :param target_frames: 目标帧数
    :return: 实际处理的帧数和策略
    """
    frame_counts = [get_video_frame_count(path) for path in video_paths]
    min_frames = min(frame_counts)

    if min_frames <= target_frames:
        # 视频较短，使用全部帧或重复帧
        return min_frames, "full"  # 使用全部可用帧
    else:
        # 视频较长，使用滑动窗口策略
        return target_frames, "window"  # 使用滑动窗口


def load_video_clip_adaptive(video_path, strategy="window", total_frames=128,
                             n_clips=8, transform=None):
    """
    自适应加载视频片段

    :param video_path: 视频路径
    :param strategy: 处理策略 ("full" 或 "window")
    :param total_frames: 总帧数
    :param n_clips: 分段数
    :param transform: 变换函数
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if strategy == "full" or frame_count <= total_frames:
        # 短视频：使用全部帧，不足则重复
        frames = []
        for i in range(total_frames):
            idx = i % frame_count  # 循环读取
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    else:
        # 长视频：滑动窗口策略
        frames = []
        # 可以选择从随机位置开始，或者从开头开始
        start_offset = 0  # 可以改为随机: random.randint(0, frame_count - total_frames) # TODO: 需要测试

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    cap.release()

    # 处理帧数不足的情况
    while len(frames) < total_frames and frames:
        frames.append(frames[-1])  # 用最后一帧填充

    # 没有帧,直接返回零张量
    if not frames:
        return torch.zeros(3, total_frames, 224, 224)

    # 应用变换
    if transform:
        frames = [transform(frame) for frame in frames]
    else:
        frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]

    frames = torch.stack(frames)  # (T, C, H, W)
    return frames.permute(1, 0, 2, 3)  # (C, T, H, W)





""" 方式二: 对视频做处理, 减少内存 大于128取前128帧, 小于128 取全部, 最后一帧循环补全到128 """
def load_video_frames_flexible(video_path, target_total_frames=128, transform=None):
    """
    更加简单灵活的视频帧加载函数
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if frame_count <= target_total_frames:
        # 帧数不足，循环读取直到达到目标数量
        for i in range(target_total_frames):
            idx = i % frame_count
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    else:
        # 帧数充足，可以选择随机起始点
        max_start = frame_count - target_total_frames
        start_frame = torch.randint(0, max_start + 1, (1,)).item()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(target_total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    cap.release()

    # 确保帧数正确
    while len(frames) < target_total_frames and frames:
        frames.append(frames[-1])

    if not frames:
        return torch.zeros(3, target_total_frames, 224, 224)

    # 应用变换
    if transform:
        frames = [transform(frame) for frame in frames]
    else:
        frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]

    frames = torch.stack(frames)  # (T, C, H, W)
    return frames.permute(1, 0, 2, 3)  # (C, T, H, W)
