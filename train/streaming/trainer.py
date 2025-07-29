# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/29 15:57
@Author  : Kend
@FileName: trainer
@Software: PyCharm
@modifier: https://github.com/Atze00/MoViNet-pytorch
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .dataset import load_video_clip, StreamingVideoDataset
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg





def train_iter_streaming(
        model,
        optimizer,
        data_loader,
        n_clips=8,
        n_clip_frames=16,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        transform = None
):
    """
    使用流式处理方式训练MoViNet模型
    8 * 16 = 128 帧
    :param model: MoViNet模型
    :param optimizer: 优化器
    :param data_loader: 数据加载器
    :param n_clips: 视频分割的片段数
    :param n_clip_frames: 每个片段的帧数
    :param device: 训练设备
    :param transform: 数据增强
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

        # 对每个clip进行处理
        clip_losses = []
        clip_outputs = []

        """ 
        流式处理 
        对于MoViNet，顺序分段方式更好，原因如下：
        模型设计匹配：MoViNet的流式处理就是为顺序处理设计的
        时间建模优势：顺序处理能更好地利用模型的时序建模能力
        缓冲区效率：连续帧处理能最大化缓冲区机制的效果
        """

        # NOTE: 把当前分为n_clips, 段, 也就是指定的切片数, clip
        for clip_idx in range(n_clips):
            # 加载当前clip的帧
            clip_frames = []
            for i, video_path in enumerate(video_paths):
                """ 对具体的视频进行操作 """
                frames = load_video_clip(
                    video_path,
                    start_frame=clip_idx * n_clip_frames,  # 0, 16, 32
                    num_frames=n_clip_frames,
                    transform=transform
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
                  f'Loss: {avg_loss:.4f}, '
                  f'Acc: {100. * correct / total_samples:.2f}%')

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = 100. * correct / total_samples

    return epoch_loss, epoch_acc


def evaluate_streaming(model, data_loader, n_clips=8, n_clip_frames=16,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    使用流式处理方式评估MoViNet模型
    :param model: MoViNet模型
    :param data_loader: 数据加载器
    :param n_clips: 视频分割的片段数
    :param n_clip_frames: 每个片段的帧数
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




""" 流式的整体训练 """
def train_streaming(
        data_root='dataset/train',
        val_root='dataset/val',
        batch_size=2,
        num_epochs=100,
        learning_rate=3e-4,
        num_classes=2,
        n_clips=8,
        n_clip_frames=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # log_dir='runs/movinet_a0_streaming',
        save_dir='checkpoints'
):
    # # 参数配置
    # data_root = 'dataset/train'
    # val_root = 'dataset/val'
    # batch_size = 2  # 建议使用较小的batch size
    # num_epochs = 100
    # learning_rate = 3e-4
    # num_classes = 2  # 拆家/正常视频
    # n_clips = 8  # 视频分割的片段数
    # n_clip_frames = 16  # 每个片段的帧数
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志和模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/movinet_a0_streaming_{timestamp}'
    # save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集 (使用StreamingVideoDataset)
    train_dataset = StreamingVideoDataset(root_dir=data_root, transform=None, clip_frames=n_clip_frames)
    val_dataset = StreamingVideoDataset(root_dir=val_root, transform=None, clip_frames=n_clip_frames)

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
            n_clips=n_clips, n_clip_frames=n_clip_frames, device=device
        )

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 验证
        val_loss, val_acc = evaluate_streaming(
            model, val_loader,
            n_clips=n_clips, n_clip_frames=n_clip_frames, device=device
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


"""
总结
核心观点确认
✅ 学习完整性：理论上，一次性加载整个视频和分段处理，模型最终学到的信息是一样的。
✅ 内存差异：这是最主要的区别，特别是对于长视频。
但还有一些重要区别
    # MoViNet的流式处理优势：
    model.clean_activation_buffers()  # 缓冲区管理
    # 这是MoViNet的核心特性，不是所有模型都有
时序建模方式不同
    整段处理：模型一次性看到完整的时序信息
    分段处理：模型通过缓冲区机制"记住"之前的时序信息
分段处理：模型通过缓冲区机制"记住"之前的时序信息
3. 实际训练效果差异
方式      内存使用    时序建模     适用场景
整段处理    高       完整上下文   短视频(<50帧)
分段处理    低       通过缓冲区   长视频(>50帧)
4. MoViNet的设计初衷
MoViNet分段处理的真正优势在于：
    实时推理：可以在视频播放时实时处理，不需要等待完整视频
    工业应用：更适合监控、直播等场景
    内存效率：可以处理任意长度的视频

分段处理主要优势是内存效率和实时处理能力，而不是学习效果本身。对于较短视频，整段处理可能更简单直接；对于长视频，分段处理是必要的。
这就是为什么我们要根据视频长度和应用场景来选择合适的处理方式。
"""