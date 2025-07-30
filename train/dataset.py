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
    数据集非常混乱有2-20秒的视频（25fps = 50-500帧）
视频长度分类：
    短视频： < 64帧  →  需要特殊处理
    中等视频：64-192帧  →  标准分段处理
    长视频： > 192帧  →  随机采样处理
"""



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



def load_video_clip_adaptive_strategy(video_path, clip_idx, n_clip_frames, strategy, total_frame_count, dsize=(224, 224)):
    """
    根据策略加载视频片段
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if strategy == "loop_fill":
        # 循环填充策略
        for i in range(n_clip_frames):
            frame_idx = (clip_idx * n_clip_frames + i) % total_frame_count
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, dsize)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    elif strategy == "standard":
        # 标准顺序策略
        start_frame = clip_idx * n_clip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(n_clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                frame = cv2.resize(frame, dsize)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    elif strategy == "random_start":
        # 随机起始点策略
        max_start = max(0, total_frame_count - (8 * n_clip_frames))
        random_offset = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        start_frame = random_offset + clip_idx * n_clip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(n_clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame is not None:
                frame = cv2.resize(frame, dsize)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    cap.release()


    # 处理帧数不足的情况
    while len(frames) < n_clip_frames and frames:
        frames.append(frames[-1])

    if not frames:
        return torch.zeros(3, n_clip_frames, 224, 224)

    return frames





""" 4: 批量处理一致性保证 """
def get_batch_adaptive_params(video_paths, base_n_clips=8, base_n_clip_frames=16):
    """
    获取batch级别的自适应参数，确保一致性
    """
    if not video_paths:
        return base_n_clips, base_n_clip_frames

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

    return n_clips, n_clip_frames



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
                self.samples.append((os.path.join(cls_folder, video), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回视频路径和标签，实际帧加载在训练时进行
        """
        video_path, label = self.samples[idx]
        return video_path, label





""" 5: 训练函数 训练一个epoch中一个batch的函数, """
def train_iter_streaming_adaptive(
        model,
        optimizer,
        data_loader,
        base_n_clips=8,
        base_n_clip_frames=16,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        transform=None
):
    """
    自适应流式训练处理不同长度视频
    """
    import torch.nn.functional as F

    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (video_paths, targets) in enumerate(data_loader):
        targets = targets.to(device)

        # 清理模型的激活缓冲区
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        optimizer.zero_grad()

        # 获取batch级别自适应参数，确保一致性
        batch_n_clips, batch_n_clip_frames = get_batch_adaptive_params(
            video_paths, base_n_clips, base_n_clip_frames
        )

        # 对每个clip进行处理
        clip_losses = []

        for clip_idx in range(batch_n_clips):
            clip_frames = []
            for i, video_path in enumerate(video_paths):
                # 为每个视频获取具体策略（但使用统一的clip参数）
                _, _, strategy, total_frames = adaptive_clip_strategy(
                    video_path, 128, batch_n_clip_frames
                )

                # 加载帧
                frames = load_video_clip_adaptive_strategy(
                    video_path, clip_idx, batch_n_clip_frames, strategy, total_frames
                )

                # 应用变换
                if transform and isinstance(frames, list):
                    frames = [transform(frame) for frame in frames]
                elif isinstance(frames, list):
                    frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]

                if isinstance(frames, list):
                    frames = torch.stack(frames).permute(1, 0, 2, 3)

                clip_frames.append(frames)

            clip_frames = torch.stack(clip_frames).to(device)

            # 前向传播
            output = model(clip_frames)

            # 计算loss并累积梯度
            loss = F.cross_entropy(output, targets) / batch_n_clips
            loss.backward()
            clip_losses.append(loss.item())

        # 更新参数
        optimizer.step()

        # 统计信息
        avg_loss = sum(clip_losses)
        total_loss += avg_loss

        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(targets).sum().item()

        total_samples += targets.size(0)

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, '
                  f'Clips: {batch_n_clips}x{batch_n_clip_frames}, '
                  f'Loss: {avg_loss:.4f}, '
                  f'Acc: {100. * correct / total_samples:.2f}%')

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = 100. * correct / total_samples

    return epoch_loss, epoch_acc
