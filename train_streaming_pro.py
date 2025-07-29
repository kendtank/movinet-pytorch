# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:28
@Author  : Kend
@FileName: train_streaming
@Software: PyCharm
@modifier:
"""


"""
视频长度不一致会影响流式训练的效果，因为不同长度的视频会导致clip数量不同
流式训练 - 支持可变长度视频
"""

"""
1. 自适应Clip策略
    adaptive_clip_strategy 函数根据视频长度动态调整clip参数
    确保短视频也能被有效处理
2. 视频帧数检测
    get_video_frame_count 函数获取每个视频的实际帧数
    作为调整策略的依据
3. 灵活的处理方式
    对于短视频：减少每clip帧数或减少clip数量
    对于长视频：使用标准参数处理
4. 处理策略
    最少保证4帧每clip
    根据batch中最短视频调整参数
    确保所有视频都能被完整处理
    训练能很好地处理2-20秒不等长度的视频，充分发挥MoViNet的流式处理优势。
"""


import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from load_dataset_with_video import StreamingVideoDataset
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg


def get_video_frame_count(video_path):
    """
    获取视频总帧数
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


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


def adaptive_clip_strategy(video_paths, target_total_clips=8, target_frames_per_clip=16):
    """
    根据视频长度自适应调整clip策略

    :param video_paths: batch中视频路径列表
    :param target_total_clips: 目标总clip数
    :param target_frames_per_clip: 目标每clip帧数
    :return: 实际使用的clip数和每clip帧数
    """
    # 获取batch中视频的帧数
    frame_counts = []
    for video_path in video_paths:
        frame_count = get_video_frame_count(video_path)
        frame_counts.append(frame_count)

    # 使用最小帧数作为基准（确保所有视频都能处理）
    min_frame_count = min(frame_counts) if frame_counts else target_total_clips * target_frames_per_clip

    # 调整策略
    if min_frame_count < target_frames_per_clip:
        # 视频太短，减少每clip帧数
        actual_frames_per_clip = max(4, min_frame_count)  # 至少4帧
        actual_total_clips = max(1, min_frame_count // actual_frames_per_clip)
    elif min_frame_count < target_total_clips * target_frames_per_clip:
        # 视频不够长，减少clip数
        actual_frames_per_clip = target_frames_per_clip
        actual_total_clips = max(1, min_frame_count // actual_frames_per_clip)
    else:
        # 视频足够长，使用默认参数
        actual_total_clips = target_total_clips
        actual_frames_per_clip = target_frames_per_clip

    return actual_total_clips, actual_frames_per_clip


def train_iter_streaming(model, optimizer, data_loader,
                         target_n_clips=8, target_n_clip_frames=16,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    使用流式处理方式训练MoViNet模型（支持可变长度视频）

    :param model: MoViNet模型
    :param optimizer: 优化器
    :param data_loader: 数据加载器
    :param target_n_clips: 目标视频分割的片段数
    :param target_n_clip_frames: 目标每个片段的帧数
    :param device: 训练设备
    """
    import torch.nn.functional as F

    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (video_paths, targets) in enumerate(data_loader):
        targets = targets.to(device)

        # 根据视频长度自适应调整clip策略
        n_clips, n_clip_frames = adaptive_clip_strategy(
            video_paths, target_n_clips, target_n_clip_frames
        )

        # 清理模型的激活缓冲区
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        optimizer.zero_grad()

        # 对每个clip进行处理
        clip_losses = []
        clip_outputs = []

        for clip_idx in range(n_clips):
            # 加载当前clip的帧
            clip_frames = []
            for i, video_path in enumerate(video_paths):
                frames = load_video_clip(
                    video_path,
                    start_frame=clip_idx * n_clip_frames,
                    num_frames=n_clip_frames,
                    transform=None
                )
                clip_frames.append(frames)

            # 合并batch中的clip帧
            clip_frames = torch.stack(clip_frames).to(device)  # (B, C, T, H, W)

            # 前向传播
            output = model(clip_frames)

            # 第一个clip计算完整输出，后续clip只计算loss
            if clip_idx == 0:
                clip_outputs.append(output)

            # 计算loss并累积梯度
            loss = F.cross_entropy(output, targets) / n_clips
            loss.backward()
            clip_losses.append(loss.item())

        # 更新参数
        optimizer.step()

        # 统计信息
        avg_loss = sum(clip_losses)
        total_loss += avg_loss

        if clip_outputs:
            pred = torch.argmax(clip_outputs[0], dim=1)
            correct += pred.eq(targets).sum().item()

        total_samples += targets.size(0)

        # 定期清理缓冲区
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, '
                  f'Clips: {n_clips}x{n_clip_frames}, '
                  f'Loss: {avg_loss:.4f}, '
                  f'Acc: {100. * correct / total_samples:.2f}%')

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = 100. * correct / total_samples

    return epoch_loss, epoch_acc


def evaluate_streaming(model, data_loader,
                       target_n_clips=8, target_n_clip_frames=16,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    使用流式处理方式评估MoViNet模型（支持可变长度视频）

    :param model: MoViNet模型
    :param data_loader: 数据加载器
    :param target_n_clips: 目标视频分割的片段数
    :param target_n_clip_frames: 目标每个片段的帧数
    :param device: 评估设备
    """
    import torch.nn.functional as F

    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for video_paths, targets in data_loader:
            targets = targets.to(device)

            # 根据视频长度自适应调整clip策略
            n_clips, n_clip_frames = adaptive_clip_strategy(
                video_paths, target_n_clips, target_n_clip_frames
            )

            # 清理模型的激活缓冲区
            if hasattr(model, 'clean_activation_buffers'):
                model.clean_activation_buffers()

            # 存储所有clip的输出用于集成
            clip_outputs = []

            for clip_idx in range(n_clips):
                # 加载当前clip的帧
                clip_frames = []
                for i, video_path in enumerate(video_paths):
                    frames = load_video_clip(
                        video_path,
                        start_frame=clip_idx * n_clip_frames,
                        num_frames=n_clip_frames,
                        transform=None
                    )
                    clip_frames.append(frames)

                # 合并batch中的clip帧
                clip_frames = torch.stack(clip_frames).to(device)

                # 前向传播
                output = model(clip_frames)
                clip_outputs.append(output)

                # 清理缓冲区
                if hasattr(model, 'clean_activation_buffers'):
                    model.clean_activation_buffers()

            # 集成所有clip的输出（平均）
            if clip_outputs:
                final_output = torch.stack(clip_outputs).mean(dim=0)
                loss = F.cross_entropy(final_output, targets)
                total_loss += loss.item()

                pred = torch.argmax(final_output, dim=1)
                correct += pred.eq(targets).sum().item()
                total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total_samples

    return avg_loss, accuracy


def train_streaming():
    # 参数配置
    data_root = 'dataset/train'
    val_root = 'dataset/val'
    batch_size = 2  # 建议使用较小的batch size
    num_epochs = 100
    learning_rate = 3e-4
    num_classes = 2  # 拆家/正常视频
    target_n_clips = 8  # 目标视频分割的片段数
    target_n_clip_frames = 16  # 目标每个片段的帧数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志和模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/movinet_a0_streaming_{timestamp}'
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集 (使用StreamingVideoDataset)
    train_dataset = StreamingVideoDataset(root_dir=data_root, transform=None, clip_frames=target_n_clip_frames)
    val_dataset = StreamingVideoDataset(root_dir=val_root, transform=None, clip_frames=target_n_clip_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 模型初始化
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=num_classes, conv_type="2plus1d", tf_like=True)
    model = model.to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_loss, train_acc = train_iter_streaming(
            model, optimizer, train_loader,
            target_n_clips=target_n_clips, target_n_clip_frames=target_n_clip_frames, device=device
        )

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 验证
        val_loss, val_acc = evaluate_streaming(
            model, val_loader,
            target_n_clips=target_n_clips, target_n_clip_frames=target_n_clip_frames, device=device
        )
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # TensorBoard 日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f'movinet_best.pth'))
            print(f"✅ Best model saved with accuracy: {best_acc:.4f}")

    writer.close()
    print("Streaming training complete.")


if __name__ == '__main__':
    train_streaming()

