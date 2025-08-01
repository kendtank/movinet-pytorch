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


"""
处理混合长度视频数据集
    数据集非常混乱有2-20秒的视频（25fps = 50-500帧）(30fps = 60-600帧)
视频长度分类：
    短视频： < 64帧  →  需要特殊处理, 一次性加载完
    中等视频：64-192帧  →  标准分段处理
    长视频： > 192帧  →  随机采样处理 (随机采集, 再标准分段处理)
"""


VALID_VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv"]  # 支持的视频加载格式

def is_valid_video_file(filename):
    return os.path.splitext(filename)[-1].lower() in VALID_VIDEO_EXTS




""" 1: 视频长度信息 """
def get_video_info(video_path):
    """
    获取视频信息
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return frame_count, fps, duration


def classify_video_length(frame_count, target_total_frames=128):
    """
    根据帧数分类视频长度
    """
    if frame_count < target_total_frames * 0.5:  # < 64帧
        return "very_short"
    elif frame_count <= target_total_frames * 1.5:  # 64-192帧
        return "medium"
    else:   # > 192帧
        return "long"




""" 2: 自适应分段策略 """
def adaptive_clip_strategy(video_path, target_total_frames=128, base_clip_frames=16):
    """
    根据视频长度自适应调整分段策略
    """
    frame_count, fps, duration = get_video_info(video_path)
    length_category = classify_video_length(frame_count, target_total_frames)

    if length_category == "very_short":
        # 极短视频：循环播放确保足够的训练数据
        n_clips = min(8, (target_total_frames + base_clip_frames - 1) // base_clip_frames)
        n_clip_frames = min(base_clip_frames, frame_count)
        strategy = "loop_fill"
    elif length_category == "medium":
        # 中等视频：标准分段处理
        n_clips = min(8, frame_count // base_clip_frames)
        n_clip_frames = min(base_clip_frames, frame_count // n_clips)
        strategy = "standard"
    else:
        # 长视频：随机起始点采样
        n_clips = 8
        n_clip_frames = base_clip_frames
        strategy = "random_start"

    return n_clips, n_clip_frames, strategy, frame_count



def read_and_process_frame(cap, frame_idx, dsize=(224, 224)):
    """
    读取并处理单帧图像：resize、BGR转RGB
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    frame = cv2.resize(frame, tuple(dsize))   # 强制转换为元组
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame




""" 加载单个视频 """
def load_video_clip_adaptive_strategy(video_path, clip_idx, n_clip_frames, strategy, total_frame_count, dsize=(224, 224), logger=None):
    """
    根据策略加载视频片段（包含因果模型适配、补帧、异常处理）
    返回格式：List[np.ndarray]，帧序列长度 = n_clip_frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    # if logger and clip_idx == 0:  # 只打印第一个clip的信息:
    #     # 打印调试信息
    #     logger.info(f"[{strategy}] clip_idx: {clip_idx}, n_clip_frames: {n_clip_frames}, total_frames: {total_frame_count}")
    def append_frame(frame_idx):
        frame = read_and_process_frame(cap, frame_idx, dsize)
        if frame is not None:
            frames.append(frame)

    try:
        if strategy == "loop_fill":
            for i in range(n_clip_frames):
                frame_idx = (clip_idx * n_clip_frames + i) % total_frame_count
                append_frame(frame_idx)

        elif strategy == "standard":
            start_frame = clip_idx * n_clip_frames
            if start_frame >= total_frame_count:
                start_frame = max(0, total_frame_count - n_clip_frames)
            for i in range(n_clip_frames):
                frame_idx = start_frame + i
                if frame_idx < total_frame_count:
                    append_frame(frame_idx)

        elif strategy == "random_start":
            max_offset = max(0, total_frame_count - (8 * n_clip_frames))
            random_offset = torch.randint(0, max_offset + 1, (1,)).item() if max_offset > 0 else 0
            start_frame = random_offset + clip_idx * n_clip_frames
            for i in range(n_clip_frames):
                frame_idx = start_frame + i
                if frame_idx < total_frame_count:
                    append_frame(frame_idx)


    finally:

        cap.release()


    # 空帧处理：完全读取失败
    if not frames:
        return torch.zeros(3, n_clip_frames, 224, 224)

    # 补齐帧数（末尾重复, 前向填充没有测试过）
    while len(frames) < n_clip_frames:
        frames.append(frames[-1])  # 也可以用 frames.insert(0, frames[0]) 做前向补齐

    return frames

"""
| 视频长度 | 帧数   | 分类          | n\_clips  | n\_clip\_frames | 采样策略        | 采样帧区间示例                                                         |
| ------ | ---   | -----------   | -------- | ---------------  | ------------- | --------------------------------------------------------------- |
| 1      | 45    | very\_short   | 1        | 45               | loop\_fill    | clip0: \[0, 44]                                                 |
| 2      | 100   | medium        | 6        | 16               | standard      | clip0: \[0, 15], clip1: \[16, 31], ..., clip5: \[80, 95]        |
| 3      | 300   | long          | 8        | 16               | random\_start | clip0: \[50, 65], clip1: \[66, 81], ..., clip7: \[162, 177]（示例） |
"""






""" 4: 批量处理一致性保证 """
def get_batch_adaptive_params(video_paths, base_n_clips=8, base_n_clip_frames=16):
    """
    获取batch级别的自适应参数，确保一致性, 并返回视频长度分类（便于调试）
    """
    if not video_paths:
        return base_n_clips, base_n_clip_frames, "undefined"

    # 获取所有视频的信息
    video_info_list = [get_video_info(path) for path in video_paths]
    frame_counts = [info[0] for info in video_info_list]

    # 使用最小帧数来确定策略，确保所有视频都能处理
    min_frames = min(frame_counts)

    # 根据最小帧数确定策略
    length_category = classify_video_length(min_frames, 128)

    if length_category == "very_short":
        # 极短视频：确保至少有1个clip
        n_clips = 1
        n_clip_frames = min_frames
    elif length_category == "medium":
        # 中等视频：标准分段
        n_clips = min(8, min_frames // base_n_clip_frames)
        n_clip_frames = min(base_n_clip_frames, min_frames // n_clips) if n_clips > 0 else base_n_clip_frames
    else:
        # 长视频：固定参数
        n_clips = base_n_clips
        n_clip_frames = base_n_clip_frames

    # 确保参数有效
    n_clips = max(1, n_clips)
    n_clip_frames = max(1, n_clip_frames)

    return n_clips, n_clip_frames, length_category



class StreamingVideoDataset(Dataset):
    def __init__(self, root_dir):
        """
        :param root_dir: 数据集视频目录
        扫描目录，构建样本列表
        不实际加载视频数据，只保存路径
        """
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []  # 样本列表, 提供给pytorch的DataLoader

        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            for video in os.listdir(cls_folder):
                if is_valid_video_file(video):
                    self.samples.append((os.path.join(cls_folder, video), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回视频路径和标签，实际帧加载在训练时进行
        """
        video_path, label = self.samples[idx]
        return video_path, label