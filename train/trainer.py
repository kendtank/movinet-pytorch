# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/29 15:57
@Author  : Kend
@FileName: trainer
@Software: PyCharm
@modifier: https://github.com/Atze00/MoViNet-pytorch
"""
import os
import sys
# 添加上次目录到项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train.logger import setup_logger
from train.utils import check_model_learning_capability

"""
官方建议的方法：
    分段处理（Streaming）：
    将视频分为 n_clips 个片段
    每个片段包含 n_clip_frames 帧
    逐段输入模型处理
    流式缓冲：使用 model.clean_activation_buffers() 清理激活缓冲区
    梯度累积：对每个片段分别计算损失并反向传播，最后统一优化
为MoViNet优化的数据集加载器，支持混合长度视频数据集训练
"""


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


import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from train.dataset import (
    StreamingVideoDataset,
    get_batch_adaptive_params,
    adaptive_clip_strategy,
    load_video_clip_adaptive_strategy
)
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg
from train.transforms import VideoTransform



""" 训练函数 训练一个epoch中一个batch的函数, """
def train_iter_streaming_adaptive(
        model,
        optimizer,
        data_loader,
        base_n_clips=8,
        base_n_clip_frames=16,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        transform=None,
        logger = None
):
    """
    自适应流式训练处理不同长度视频（训练一个epoch）
    """
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (video_paths, targets) in enumerate(data_loader):
        targets = targets.to(device)

        # 在每个视频批次开始前清理模型的激活缓冲区
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        optimizer.zero_grad()

        # 获取batch级别自适应参数，确保一致性
        batch_n_clips, batch_n_clip_frames = get_batch_adaptive_params(
            video_paths, base_n_clips, base_n_clip_frames
        )
        # TODO: # 问题：每个clip独立计算损失，然后简单平均, 应该：收集所有clip的输出，平均后再计算损失
        # 收集所有clip的输出
        clip_outputs = []

        # # 对每个clip进行处理
        # clip_losses = []

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
            # 集成所有clip的输出
            clip_outputs.append(output)
            # # 添加调试信息
            # if batch_idx == 0 and logger:
            #     logger.info(f"Train batch output range: [{output.min():.4f}, {output.max():.4f}]")

        # 对所有clip的输出求平均
        if clip_outputs:
            final_output = torch.stack(clip_outputs).mean(dim=0)
            # 计算最终损失
            loss = F.cross_entropy(final_output, targets)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计信息
            total_loss += loss.item()
            pred = torch.argmax(final_output, dim=1)
            correct += pred.eq(targets).sum().item()
            total_samples += targets.size(0)


    # TODO:
        """
                主要改进点
        统一损失计算：收集所有clip的输出，平均后再计算损失，这样更符合MoViNet的设计理念
        准确的准确率计算：基于所有clip的平均输出计算准确率
        正确的梯度更新：在处理完所有clip后进行一次参数更新
        这种修改方式更符合MoViNet流式处理的设计思想，能够更好地利用模型的时序建模能力。
        """

            # # 计算loss并累积梯度
            # loss = F.cross_entropy(final_output, targets) / batch_n_clips
            # loss.backward()
            # clip_losses.append(loss.item())
        #
        # # 更新参数
        # optimizer.step()

        # # 统计信息
        # avg_loss = sum(clip_losses)
        # total_loss += avg_loss

        # with torch.no_grad():
        #     pred = torch.argmax(output, dim=1)
        #     correct += pred.eq(targets).sum().item()
        #
        # total_samples += targets.size(0)

        if batch_idx % 10 == 0:
            logger.info(f'Batch {batch_idx}, '
                        f'Clips: {batch_n_clips}x{batch_n_clip_frames}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Acc: {100. * correct / total_samples:.2f}%')

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = 100. * correct / total_samples

    return epoch_loss, epoch_acc




"""  002  验证有效果"""
def evaluate_streaming_adaptive(
        model,
        data_loader,
        base_n_clips=8,
        base_n_clip_frames=16,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        logger=None
):
    """
    自适应流式评估处理不同长度视频
    修改为与训练函数一致的处理方式
    """
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (video_paths, targets) in enumerate(data_loader):
            targets = targets.to(device)
            all_targets.extend(targets.cpu().numpy())

            # 在处理每个视频批次前清理缓冲区
            if hasattr(model, 'clean_activation_buffers'):
                model.clean_activation_buffers()

            # 获取batch级别自适应参数
            batch_n_clips, batch_n_clip_frames = get_batch_adaptive_params(
                video_paths, base_n_clips, base_n_clip_frames
            )

            # 收集所有clip的输出（与训练函数保持一致）
            clip_outputs = []

            # 处理每个clip
            for clip_idx in range(batch_n_clips):
                clip_frames = []
                for i, video_path in enumerate(video_paths):
                    # 获取该视频的具体策略
                    _, _, strategy, total_frames = adaptive_clip_strategy(
                        video_path, 128, batch_n_clip_frames
                    )

                    # 加载帧
                    frames = load_video_clip_adaptive_strategy(
                        video_path, clip_idx, batch_n_clip_frames, strategy, total_frames
                    )

                    # 应用变换（评估时不使用数据增强）
                    if isinstance(frames, list):
                        frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]

                    if isinstance(frames, list):
                        frames = torch.stack(frames).permute(1, 0, 2, 3)

                    clip_frames.append(frames)

                clip_frames = torch.stack(clip_frames).to(device)

                # 前向传播
                output = model(clip_frames)
                clip_outputs.append(output)

                # 注意：不要在这里清理缓冲区，保持时序状态

            # 对所有clip的输出求平均（与训练函数保持一致）
            if clip_outputs:
                final_output = torch.stack(clip_outputs).mean(dim=0)

                loss = F.cross_entropy(final_output, targets)
                total_loss += loss.item()

                pred = torch.argmax(final_output, dim=1)
                all_predictions.extend(pred.cpu().numpy())

                correct += pred.eq(targets).sum().item()
                total_samples += targets.size(0)

                if logger and batch_idx == 0:  # 只在第一个batch记录详细信息
                    logger.info(f"Val batch targets: {targets.cpu().numpy()}, predictions: {pred.cpu().numpy()}")
                    logger.info(f"Val batch output logits: {final_output.cpu().numpy()}")

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total_samples

    if logger:
        import numpy as np
        unique_preds, counts = np.unique(all_predictions, return_counts=True)
        unique_targets, target_counts = np.unique(all_targets, return_counts=True)
        logger.info(f'Prediction distribution: {dict(zip(unique_preds, counts))}')
        logger.info(f'Target distribution: {dict(zip(unique_targets, target_counts))}')
        logger.info(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%')

    return avg_loss, accuracy



def train_streaming_adaptive(
        data_root='dataset/train',
        val_root='dataset/val',
        batch_size=1,
        num_epochs=100,
        learning_rate=3e-4,
        num_classes=2,
        base_n_clips=8,
        base_n_clip_frames=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir='checkpoints'
):
    """
    完整的自适应流式训练流程
    """
    # 日志和模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/movinet_a0_streaming_adaptive_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # 设置日志记录器
    my_logger = setup_logger(log_dir)


    # 加载数据集
    train_dataset = StreamingVideoDataset(root_dir=data_root)
    val_dataset = StreamingVideoDataset(root_dir=val_root)


    my_logger.info(f"Validation dataset size: {len(val_dataset)}")

    # # 检查验证集标签分布
    val_loader_temp = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_val_targets = []
    for _, targets in val_loader_temp:
        all_val_targets.extend(targets.numpy())

    import numpy as np
    unique_targets, counts = np.unique(all_val_targets, return_counts=True)
    my_logger.info(f"Validation target distribution: {dict(zip(unique_targets, counts))}")

    # 检查训练集标签分布
    my_logger.info(f"Training dataset size: {len(train_dataset)}")
    train_loader_temp = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_train_targets = []
    for _, targets in train_loader_temp:
        all_train_targets.extend(targets.numpy())

    unique_train_targets, train_counts = np.unique(all_train_targets, return_counts=True)
    my_logger.info(f"Training target distribution: {dict(zip(unique_train_targets, train_counts))}")




    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 模型初始化
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=True, num_classes=num_classes, conv_type="2plus1d", tf_like=True)
    model = model.to(device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # SGD不同的优化器
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):

        my_logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        # 在训练循环中添加检查
        if epoch == 0:
            check_model_learning_capability(model, val_loader, device, my_logger)
        # 训练
        train_loss, train_acc = train_iter_streaming_adaptive(
            model, optimizer, train_loader,
            base_n_clips=base_n_clips,
            base_n_clip_frames=base_n_clip_frames,
            device=device,
            transform=VideoTransform(is_train=True).transform,
            logger=my_logger
        )

        my_logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 验证
        val_loss, val_acc = evaluate_streaming_adaptive(
            model, val_loader,
            base_n_clips=base_n_clips,
            base_n_clip_frames=base_n_clip_frames,
            device=device,
            logger=my_logger
        )
        my_logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

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
            my_logger.info(f"✅ Best model saved with accuracy: {best_acc:.4f}")

    writer.close()
    my_logger.info("Adaptive streaming training complete.")
    return model




# 使用
if __name__ == "__main__":
    # 参数配置
    config = {
        'data_root': '/home/kend/Guanxin/work/workspace/movinet-pytorch/dataset/train',
        'val_root': '/home/kend/Guanxin/work/workspace/movinet-pytorch/dataset/val',
        # 'data_root': '/home/kend/Guanxin/Datasets/dataset/classes/movinet-pet-destruction-video/hmdb_test/train',
        # 'val_root': '/home/kend/Guanxin/Datasets/dataset/classes/movinet-pet-destruction-video/hmdb_test/val',

        'batch_size': 2,
        'num_epochs': 100,
        'learning_rate': 3e-4,
        'num_classes': 2,
        'base_n_clips': 8,
        'base_n_clip_frames': 16,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'save_dir': 'checkpoints'
    }

    # 开始训练
    trained_model = train_streaming_adaptive(**config)
