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
from idlelib.debugobj import myrepr

# 添加上次目录到项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

from train.tf_dataset_loder import VideoDataset, collate_fn  # 这是我写的和tf官方的训练一致的数据集加载
from train.transforms import VideoTransformPad
# from net.movinet_lite import MoViNet
# from net.movinet_4d import MoViNet4D
# from net.cfg import build_movinet_a0_cfg
from net.movinet_lite import MoViNet2D1D
from train.logger import setup_logger

"""
官方建议的方法：
    分段处理（Streaming）：
    将视频分为 n_clips 个片段
    每个片段包含 n_clip_frames 帧
    逐段输入模型处理
    流式缓冲：使用 model.clean_activation_buffers() 清理单个视频的激活缓冲区
    梯度累积：对每个片段分别计算损失并反向传播，最后统一优化
为MoViNet优化的数据集加载器，支持混合长度视频数据集训练
注意： 这里对于端侧， 流式处理的动态缓存都是不支持的， 我们直接舍弃

总结
核心观点确认
    学习完整性：理论上，一次性加载整个视频和分段处理，模型最终学到的信息是一样的。
    内存差异：这是最主要的区别，特别是对于长视频。
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
这就是为什么要根据视频长度和应用场景来选择合适的处理方式。
"""

# def get_logger(log_dir="train_logger"):
#     os.makedirs(log_dir, exist_ok=True)
#     logger = setup_logger(log_dir)
#     return logger



# ---- 训练 / 验证循环（固定 clip） ----
def train_iter_fixed_clip(model, optimizer, data_loader, device, logger):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (clips, targets) in enumerate(data_loader):
        # 过滤空 batch（collate_fn 已过滤，大概率不触发）
        if clips is None or clips.numel() == 0:
            logger.warning(f"第 {batch_idx} 个 batch 为空，跳过。")
            continue

        clips = clips.to(device)      # [B, C, T, H, W]
        targets = targets.to(device)

        # 对非流式模型，通常不需要清缓存，但调用也安全
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        optimizer.zero_grad()
        outputs = model(clips)        # [B, num_classes]
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total_samples += targets.size(0)

        if batch_idx % 10 == 0:
            acc = 100.0 * correct / total_samples if total_samples > 0 else 0.0
            logger.info(f"[训练] Batch {batch_idx}  Loss={loss.item():.4f}  累计Acc={acc:.2f}%")

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = 100.0 * correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc



def evaluate_fixed_clip(model, data_loader, device, logger):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (clips, targets) in enumerate(data_loader):
            if clips is None or clips.numel() == 0:
                continue
            clips = clips.to(device)
            targets = targets.to(device)

            if hasattr(model, 'clean_activation_buffers'):
                model.clean_activation_buffers()

            outputs = model(clips)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()

            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0

    # 打印分布（中文）
    try:
        unique_preds, counts = np.unique(all_preds, return_counts=True)
        unique_targets, tcounts = np.unique(all_targets, return_counts=True)
        logger.info(f"验证预测分布: {dict(zip(unique_preds, counts))}")
        logger.info(f"验证目标分布: {dict(zip(unique_targets, tcounts))}")
    except Exception:
        pass

    logger.info(f"[验证] Loss={avg_loss:.4f}  Acc={accuracy:.2f}%")
    return avg_loss, accuracy



# ---- 主训练函数 ----
def train_fixed_clip(
        data_root,
        batch_size=2,
        clip_len=16,
        stride=None,
        num_epochs=100,
        lr=3e-4,
        num_classes=2,
        num_workers=2,
        device=None,
        save_dir='checkpoints',
        log_dir=None
):
    """
    data_root: 数据集根目录，要求结构为:
        data_root/train/<class>/*.mp4
        data_root/val/<class>/*.mp4
    clip_len: clip 帧长度 T
    stride: 滑动步长，默认 T//2（可设为 T//4 增加样本密度）
    num_workers: DataLoader 的 worker 数量（IO/内存权衡）
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    stride = stride or max(1, clip_len // 4)

    # 日志 & 保存路径
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = f"runs/movinet_2d_lite_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    logger.info("训练启动")
    logger.info(f"Device: {device}, clip_len={clip_len}, stride={stride}, batch_size={batch_size}, workers={num_workers}")

    # 数据集和 DataLoader

    train_dataset = VideoDataset(data_root, mode='train', clip_len=clip_len, transform=VideoTransformPad(is_train=True, resize=224))
    val_dataset = VideoDataset(data_root, mode='val', clip_len=clip_len, transform=VideoTransformPad(is_train=False, resize=224))


    # 注意: 我们内部使用了滑动 step=clip_len//4（如需修改，修改 VideoDataset 构造）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=max(1, num_workers//2), collate_fn=collate_fn, pin_memory=True)

    logger.info(f"训练集大小: {len(train_dataset)} clips  验证集大小: {len(val_dataset)} clips")

    # 模型
    # cfg = build_movinet_a0_cfg()

    # 这里是修改动态shape后的模型定义
    # model = MoViNet4D(num_classes=num_classes)
    # model = MoViNet4D(num_classes=num_classes)
    model = MoViNet2D1D(num_classes=num_classes)
    model = model.to(device)

    # 优化器、调度
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0
    for epoch in range(num_epochs):
        t0 = time.time()
        logger.info(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        train_loss, train_acc = train_iter_fixed_clip(model, optimizer, train_loader, device, logger)
        logger.info(f"[Epoch {epoch+1}] 训练 Loss={train_loss:.4f}, Acc={train_acc:.4f}%")

        val_loss, val_acc = evaluate_fixed_clip(model, val_loader, device, logger)
        logger.info(f"[Epoch {epoch+1}] 验证 Loss={val_loss:.4f}, Acc={val_acc:.4f}%")

        # LR 调度
        scheduler.step(val_loss)

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f"movinet_2d_lite_{timestamp}.pth")
            torch.save(model.state_dict(), save_path)
            # torch.save(model, save_path)  # 保存完整模型
            logger.info(f"✅ 保存最佳模型: {save_path}  (Acc={best_acc:.4f}%)")

        t1 = time.time()
        logger.info(f"Epoch 耗时: {(t1-t0):.1f}s")

    writer.close()
    logger.info("训练结束")
    return model



if __name__ == "__main__":

    config = {
        'data_root': '/home/kend/Guanxin/work/workspace/movinet-pytorch/dataset',
        'batch_size': 8,          # 推荐 16（16GB GPU 下视模型/其它占用微调）
        'clip_len': 16,
        'stride': 4,              # T//4，若想更密集需要再源码中修改， 这里我没有做接口的暴露
        'num_epochs': 1,
        'lr': 3e-4,
        'num_classes': 2,
        'num_workers': 8,         # IO/内存平衡: 2或4，若OOM降为1
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'save_dir': 'checkpoints',
        'log_dir': None,
    }

    trained_model = train_fixed_clip(**config)


"""


"""




# ##### 旧的训练函数
# def train_iter_fixed_clip(
#         model,
#         optimizer,
#         data_loader,
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         logger=None,
# ):
#     """
#     固定每个clip为16帧训练，不再拼接长时间维。
#     batch 后形状 [B,3,16,H,W]
#     """
#     model.train()
#     total_loss = 0
#     correct = 0
#     total_samples = 0
#
#     for batch_idx, (clips, targets) in enumerate(data_loader):
#         # if not clips.any():
#         if clips is None or len(clips) == 0:
#             if logger:
#                 logger.warning(f"Empty batch at index {batch_idx}, skipping...")
#             continue
#
#         clips = clips.to(device)  # [B,3,16,H,W]
#         targets = targets.to(device)
#
#         # 清理模型缓冲区
#         if hasattr(model, 'clean_activation_buffers'):
#             model.clean_activation_buffers()
#
#         optimizer.zero_grad()
#
#         # 前向
#         outputs = model(clips)  # [B,num_classes]
#         loss = F.cross_entropy(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         # 统计信息
#         total_loss += loss.item()
#         pred = torch.argmax(outputs, dim=1)
#         correct += pred.eq(targets).sum().item()
#         total_samples += targets.size(0)
#
#         if logger and batch_idx % 10 == 0:
#             logger.info(f"[Batch {batch_idx}] Loss: {loss.item():.4f}, "
#                         f"Acc: {100.*correct/total_samples:.2f}%")
#
#     epoch_loss = total_loss / len(data_loader)
#     epoch_acc = 100. * correct / total_samples if total_samples > 0 else 0.0
#     return epoch_loss, epoch_acc




# """ 训练函数 训练一个epoch中一个batch的函数, """
# def train_iter_streaming_adaptive(
#         model,
#         optimizer,
#         data_loader,
#         base_n_clips=8,
#         base_n_clip_frames=16,
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         transform=None,
#         logger = None,
#         dsize=(224, 224)
# ):
#     """
#     自适应流式训练处理不同长度视频（训练一个epoch）
#     """
#     model.train()
#     total_loss = 0
#     correct = 0
#     total_samples = 0
#
#     for batch_idx, (video_paths, targets) in enumerate(data_loader):
#
#         if not video_paths:
#             if logger:
#                 logger.warning(f"Empty video path list at batch {batch_idx}, skipping...")
#             continue
#         targets = targets.to(device)
#
#         # 在每个视频开始前清理模型的激活缓冲区, 单独的一个视频的激活缓冲区应该被清理
#         if hasattr(model, 'clean_activation_buffers'):
#             model.clean_activation_buffers()
#
#         optimizer.zero_grad()
#
#         # 获取batch参数，确保一致性
#         batch_n_clips, batch_n_clip_frames, length_category = get_batch_adaptive_params(
#             video_paths, base_n_clips, base_n_clip_frames
#         )
#         if logger and batch_idx % 10 == 0:
#             logger.info(
#                 f"[Batch {batch_idx}] Length category: {length_category}, Clips: {batch_n_clips}, Clip frames: {batch_n_clip_frames}")
#
#
#         # TODO: # 问题：每个clip独立计算损失，然后简单平均, 应该：收集所有clip的输出，平均后再计算损失
#         # 收集所有clip的输出
#         clip_outputs = []
#
#
#         # 对每个clip进行处理
#         for clip_idx in range(batch_n_clips):
#             clip_frames = []
#             valid_target_indices = []  # 记录有效视频的索引和目标值
#
#             for i, video_path in enumerate(video_paths):
#                 try:
#                     # 为每个视频获取具体策略（但使用统一的clip参数）
#                     _, _, strategy, total_frames = adaptive_clip_strategy(
#                         video_path, 128, batch_n_clip_frames
#                     )
#
#                     # 加载帧
#                     frames = load_video_clip_adaptive_strategy(
#                         video_path, clip_idx, batch_n_clip_frames, strategy, total_frames, dsize, logger
#                     )
#
#                     # 应用变换
#                     if isinstance(frames, list):
#                         if transform:
#                             frames = [transform(frame) for frame in frames]
#                         else:
#                             frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]
#
#                         frames = torch.stack(frames).permute(1, 0, 2, 3)
#                         clip_frames.append(frames)
#                         valid_target_indices.append(i)  # 记录有效视频
#                     # else:  # skip broken video，不添加到clip_frames中
#
#
#                 except Exception as e:
#                     if logger:
#                         logger.error(f"Failed to load video {video_path}: {e}")
#                     continue
#
#             # 检查是否有有效帧
#             if not clip_frames:
#                 if logger:
#                     logger.warning(f"No valid frames loaded for clip {clip_idx}")
#                 continue
#
#             # 将列表转换为Tensor并移到设备上
#             clip_frames_tensor = torch.stack(clip_frames).to(device)
#
#             # 前向传播
#             output = model(clip_frames_tensor)
#             # 集成所有clip的输出
#             clip_outputs.append(output)   # BUG: 将Tensor当作list处理时
#             # # 添加调试信息
#             # if batch_idx == 0 and logger:
#             #     logger.info(f"Train batch output range: [{output.min():.4f}, {output.max():.4f}]")
#
#         # 对所有clip的输出求平均
#         if clip_outputs:
#
#             final_output = torch.stack(clip_outputs).mean(dim=0)
#
#             # 使用有效目标值
#             valid_targets = targets[valid_target_indices] if valid_target_indices else targets
#
#             # 计算最终损失, 确保所有clip的输出都是有效的, 不会影响训练
#             try:
#                 loss = F.cross_entropy(final_output, valid_targets)
#             except Exception as e:
#                 if logger:
#                     logger.error(f"Loss calculation error at batch {batch_idx}: {e}")
#                 continue
#
#             loss.backward()
#             # 更新参数
#             optimizer.step()
#
#             # 统计信息
#             total_loss += loss.item()
#             pred = torch.argmax(final_output, dim=1)
#             correct += pred.eq(valid_targets).sum().item()
#             total_samples += valid_targets.size(0)
#             if logger and batch_idx % 10 == 0:
#                 logger.info(
#                     f'[Train] Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100. * correct / total_samples:.2f}%')
#
#         # TODO:
#         """
#         主要改进点
#             统一损失计算：收集所有clip的输出，平均后再计算损失，这样更符合MoViNet的设计理念
#             准确的准确率计算：基于所有clip的平均输出计算准确率
#             正确的梯度更新：在处理完所有clip后进行一次参数更新
#             这种修改方式更符合MoViNet流式处理的设计思想，能够更好地利用模型的时序建模能力。
#         """
#
#             # # 计算loss并累积梯度
#             # loss = F.cross_entropy(final_output, targets) / batch_n_clips
#             # loss.backward()
#             # clip_losses.append(loss.item())
#         #
#         # # 更新参数
#         # optimizer.step()
#
#         # # 统计信息
#         # avg_loss = sum(clip_losses)
#         # total_loss += avg_loss
#
#         # with torch.no_grad():
#         #     pred = torch.argmax(output, dim=1)
#         #     correct += pred.eq(targets).sum().item()
#         # total_samples += targets.size(0)
#
#     epoch_loss = total_loss / len(data_loader)
#     epoch_acc = 100. * correct / total_samples if total_samples > 0 else 0.0
#
#     return epoch_loss, epoch_acc
#
#
#
#
# """ 验证函数 验证一个epoch中一个batch的函数, 保持和训练一致, 取消图像增强"""
# def evaluate_streaming_adaptive(
#         model,
#         data_loader,
#         base_n_clips=8,
#         base_n_clip_frames=16,
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         logger=None,
#         dsize=(224, 224)
# ):
#     """
#     自适应流式评估处理不同长度视频
#     保持训练函数一致的处理方式
#     """
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total_samples = 0
#
#     all_predictions = []
#     all_targets = []
#
#     with torch.no_grad():
#         for batch_idx, (video_paths, targets) in enumerate(data_loader):
#             if not video_paths:
#                 if logger:
#                     logger.warning(f"Empty video path list at val batch {batch_idx}, skipping...")
#                 continue
#
#             targets = targets.to(device)
#             all_targets.extend(targets.cpu().numpy())
#
#             # 在处理每个视频批次前清理缓冲区
#             if hasattr(model, 'clean_activation_buffers'):
#                 model.clean_activation_buffers()
#
#             # 获取batch级别自适应参数
#             batch_n_clips, batch_n_clip_frames, length_category = get_batch_adaptive_params(
#                 video_paths, base_n_clips, base_n_clip_frames
#             )
#
#             # 收集所有clip的输出（与训练函数保持一致）
#             clip_outputs = []
#
#             # 处理每个clip
#             for clip_idx in range(batch_n_clips):
#                 clip_frames = []
#                 valid_target_indices = []  # 记录有效视频的索引
#
#                 # 为当前clip收集所有视频的帧
#                 for i, video_path in enumerate(video_paths):
#
#                     try:
#                         # 获取该视频的具体策略
#                         _, _, strategy, total_frames = adaptive_clip_strategy(
#                             video_path, 128, batch_n_clip_frames
#                         )
#
#                         # 加载帧
#                         frames = load_video_clip_adaptive_strategy(
#                             video_path, clip_idx, batch_n_clip_frames, strategy, total_frames, dsize, logger
#                         )
#                         if isinstance(frames, list):
#                             frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]
#                             frames = torch.stack(frames).permute(1, 0, 2, 3)
#                             clip_frames.append(frames)
#                             valid_target_indices.append(i)  # 记录有效视频索引
#                         # else:
#                         #     continue
#
#                     except Exception as e:
#                         if logger:
#                             logger.error(f"[Eval] Failed to load video {video_path}: {e}")
#                         continue
#
#                 # 检查是否有有效帧
#                 if not clip_frames:
#                     continue
#
#                 clip_frames = torch.stack(clip_frames).to(device)
#
#                 # 前向传播
#                 output = model(clip_frames)
#                 clip_outputs.append(output)
#
#             # 注意：不要在这里清理缓冲区，保持时序状态
#
#             # 对所有clip的输出求平均（与训练函数保持一致）
#             if clip_outputs:
#                 final_output = torch.stack(clip_outputs).mean(dim=0)
#                 # 使用有效目标值
#                 valid_targets = targets[valid_target_indices] if valid_target_indices else targets
#
#                 try:
#                     loss = F.cross_entropy(final_output, valid_targets)
#                 except Exception as e:
#                     if logger:
#                         logger.error(f"Loss error at val batch {batch_idx}: {e}")
#                     continue
#
#                 total_loss += loss.item()
#
#                 pred = torch.argmax(final_output, dim=1)
#                 all_predictions.extend(pred.cpu().numpy())
#
#                 correct += pred.eq(targets).sum().item()
#                 total_samples += valid_targets.size(0)
#
#                 if logger and batch_idx == 0:  # 只在第一个batch记录详细信息
#                     logger.info(f"Val batch targets: {valid_targets.cpu().numpy()}, predictions: {pred.cpu().numpy()}")
#                     logger.info(f"Val batch output logits: {final_output.cpu().numpy()}")
#
#     avg_loss = total_loss / len(data_loader)
#     accuracy = 100. * correct / total_samples if total_samples > 0 else 0.0
#
#     if logger:
#         import numpy as np
#         unique_preds, counts = np.unique(all_predictions, return_counts=True)
#         unique_targets, target_counts = np.unique(all_targets, return_counts=True)
#         logger.info(f'Prediction distribution: {dict(zip(unique_preds, counts))}')
#         logger.info(f'Target distribution: {dict(zip(unique_targets, target_counts))}')
#         logger.info(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%')
#
#     return avg_loss, accuracy
#
#
#
# """ 训练函数, 训练整个数据集 """
# def train_streaming_adaptive(
#         data_root='dataset/train',
#         val_root='dataset/val',
#         batch_size=1,
#         num_epochs=100,
#         learning_rate=3e-4,
#         num_classes=2,
#         base_n_clips=8,
#         base_n_clip_frames=16,
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#         save_dir='checkpoints',
#         dsize=(224, 224)
# ):
#     """
#     完整的自适应流式训练流程
#     """
#     # 日志和模型保存路径
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_dir = f'runs/movinet_a0_no_streaming_adaptive_{timestamp}'
#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
#     # 设置日志记录器
#     my_logger = setup_logger(log_dir)
#
#
#     # 加载数据集
#     train_dataset = StreamingVideoDataset(root_dir=data_root)
#     val_dataset = StreamingVideoDataset(root_dir=val_root)
#
#     # 记录数据量信息
#     my_logger.info(f"Training dataset size: {len(train_dataset)}")
#     my_logger.info(f"Validation dataset size: {len(val_dataset)}")
#
#     # # 检查验证集标签分布
#     val_loader_temp = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     all_val_targets = []
#     for _, targets in val_loader_temp:
#         all_val_targets.extend(targets.numpy())
#
#
#     # unique_targets, counts = np.unique(all_val_targets, return_counts=True)
#     # my_logger.info(f"Validation target distribution: {dict(zip(unique_targets, counts))}")
#
#
#     # 检查训练集标签分布
#     def analyze_distribution(dataloader, label_name):
#         targets = []
#         for _, y in dataloader:
#             targets.extend(y.numpy())
#         unique, counts = np.unique(targets, return_counts=True)
#         my_logger.info(f"{label_name} target distribution: {dict(zip(unique, counts))}")
#
#     analyze_distribution(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4), "Training")
#     analyze_distribution(DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4), "Validation")
#
#
#
#     # 构建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     # 模型初始化
#     cfg = build_movinet_a0_cfg()
#     # my_logger.info(f"model classes={num_classes}")
#     # 注意：当 pretrained=True 时，模型会强制使用600个类别（K600数据集）
#     # model = MoViNet(cfg, causal=True, pretrained=False, num_classes=num_classes, conv_type="2plus1d", tf_like=True)# 先加载预训练模型
#     # model = MoViNet(cfg, causal=True, pretrained=True, conv_type="2plus1d", tf_like=True)
#     model = MoViNet(
#         cfg,
#         # causal=True,  # 因果模式
#         causal=False,  # 因果模式
#         pretrained=False,  # 使用预训练权重
#         num_classes=num_classes,  # 明确类别数
#         conv_type="2plus1d",  # 2+1D卷积
#         tf_like=False  # 避免pad错误
#     )
#
#     # 然后替换分类器以适应你的任务
# #     model.classifier = nn.Sequential(
# #         # 新的分类器层，适配你的类别数
# #         ConvBlock3D(cfg.conv7.out_channels, cfg.dense9.hidden_dim,
# #                     kernel_size=(1, 1, 1), tf_like=True, causal=True, conv_type="2plus1d", bias=True),
# #         Swish(),
# #         nn.Dropout(p=0.2, inplace=True),
# #         ConvBlock3D(cfg.dense9.hidden_dim, num_classes,  # 使用实际的类别数
# #                     kernel_size=(1, 1, 1), tf_like=True, causal=True, conv_type="2plus1d", bias=True),
# # )
#     model = model.to(device)
#     # my_logger.info(f"Model : {model}")
#
#     # 优化器和学习率调度器
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     # SGD不同的优化器
#     # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
#
#     # TensorBoard
#     writer = SummaryWriter(log_dir=log_dir)
#
#     # 训练循环
#     best_acc = 0.0
#     for epoch in range(num_epochs):
#
#         my_logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
#         # 在训练循环中添加检查
#         if epoch == 0:
#             check_model_learning_capability(model, val_loader, device, my_logger, dsize)
#         # 训练  这里替换为T固定16帧：
#         train_loss, train_acc = train_iter_fixed_clip(
#             model, optimizer, train_loader,
#             device=device,
#             logger=my_logger
#         )
#
#         # train_loss, train_acc = train_iter_streaming_adaptive(
#         #     model, optimizer, train_loader,
#         #     base_n_clips=base_n_clips,
#         #     base_n_clip_frames=base_n_clip_frames,
#         #     device=device,
#         #     transform=VideoTransform(is_train=True).transform,
#         #     logger=my_logger,
#         #     dsize=dsize
#         # )
#
#         my_logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
#
#         # 验证
#         val_loss, val_acc = evaluate_streaming_adaptive(
#             model, val_loader,
#             base_n_clips=base_n_clips,
#             base_n_clip_frames=base_n_clip_frames,
#             device=device,
#             logger=my_logger,
#             dsize=dsize
#         )
#         my_logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
#
#         # 学习率调度
#         scheduler.step(val_loss)
#
#         # TensorBoard 日志
#         writer.add_scalar('Loss/train', train_loss, epoch)
#         writer.add_scalar('Accuracy/train', train_acc, epoch)
#         writer.add_scalar('Loss/val', val_loss, epoch)
#         writer.add_scalar('Accuracy/val', val_acc, epoch)
#
#         # 保存最佳模型
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), os.path.join(save_dir, f'movinet_lite_best.pth'))
#             my_logger.info(f"✅ Best model saved with accuracy: {best_acc:.4f}")
#
#     writer.close()
#     my_logger.info("Adaptive streaming training complete.")
#     return model




# 使用
# if __name__ == "__main__":
#     # 参数配置
#     config = {
#         'data_root': '/home/kend/Guanxin/work/workspace/movinet-pytorch/dataset/train',
#         'val_root': '/home/kend/Guanxin/work/workspace/movinet-pytorch/dataset/val',
#         # 'data_root': '/home/kend/Guanxin/Datasets/dataset/classes/movinet-pet-destruction-video/hmdb_test/train',
#         # 'val_root': '/home/kend/Guanxin/Datasets/dataset/classes/movinet-pet-destruction-video/hmdb_test/val',
#         'batch_size': 2,
#         'num_epochs': 5,
#         'learning_rate': 3e-4,
#         'num_classes': 2,
#         'base_n_clips': 8,
#         'base_n_clip_frames': 16,
#         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#         'save_dir': 'movinet-pro',
#         'dsize': (224, 224)
#     }
#
#     # 开始训练
#     trained_model = train_streaming_adaptive(**config)

"""
测试：
    1：使用流模式， conv_type = "2plus1d"， 测试导出onnx ops = 11
    问题：
        MoViNet 在剪枝后导出 ONNX 出现的主要问题是 自适应池化输出大小非常量 + einops 的 rearrange / reduce + SE 模块里的动态计算 导致 TorchScript / trace 都不稳定。
    解决办法：
        1, 添加参数来解决einops和adaptive pooling问题
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        但是解决了导出了的问题， 但是以来pytorch的算子，对于端侧部署是不适用的。
        2, ONNX 导出要求 AdaptiveAvgPool / AdaptiveMaxPool 的 output_size 必须是常量（tuple）
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) 改为：self.avg_pool = nn.AdaptiveAvgPool3d((T, 1, 1))  # T是你推理时固定的时间步数，比如8， 但是还需要确保其他维度保持对齐


    2：取消流模式， conv_type = "2plus1d"， 测试导出onnx ops = 11， 导出不需要ATEN操作
    但是转RKNN时候
    模型中名为/conv1/Pad的Pad操作节点存在以下问题：
    Pads参数数量不正确
    形状推断失败，(5,2)形状无法广播到(6,2)
    导出的 movinet_optimized.onnx 里面，Pad 节点的 pads 参数数量不符合 Rockchip RKNN 的要求，或者在 shape 推理时多了一个维度/少了一个维度
    RKNN Toolkit 1.7.5 基于 ONNX Runtime 做 shape 推理，但它对 Pad 操作的支持比较有限，要求：

    Pads 数组长度 = 2 × 输入张量维度
    不支持动态 pads（即 pads 必须是常量）
    不支持空 shape 输入
    MoviNet 的模型里 /conv1/Pad 可能是 动态计算 padding（比如通过 ConstantOfShape 或 Shape → Slice → Concat），RKNN 在 shape 推理时就直接挂了。


解决方案：
    1) 开启“流模式”，conv_type="2plus1d"（端侧能不能行？）
        现实限制
        
        你代码里的 CausalModule/TemporalCGAvgPool3D 通过 self.activation 维护跨 chunk 的时序缓存；
        ONNX/RKNN 不支持这种内部可变状态，导图时它会被“折叠”为无状态计算图（导出那一刻 activation is None）。
        
        结论：模型图本身不会记住上个 chunk。要在端侧“流式”推，必须由应用层维护 FIFO，把“历史帧+当前帧”拼在一起送进 RKNN。
        
        可行做法（工程实现）
        
        训练/离线验证时仍用 causal=True（对精度有帮助）。
        
        导出 ONNX 时用固定 T（比如 8/16），模型输入固定为 [1,3,T,224,224]。
        
        端侧“流式”时，在 应用层维护一个长度为 T 的环形缓冲：
        
        每来 1 帧就把它 append；
        
        取最近 T 帧组成一个 [1,3,T,H,W] 窗口喂 RKNN；
        
        取输出作为当前时刻的结果。
        
        这就是“滑窗等效流式”。相比真正因果的内部缓存，精度基本一致（甚至因为窗口更完整，有时更稳）。
        
        要点：不要指望 RKNN 内部保存 state；把 state 放在应用层，你就能“流”起来。
        
        2) 关闭“流模式”，conv_type="2plus1d"（整段 or 滑窗）
        
        这条路径最稳：
        
        导出稳定：完全不需要 ATEN fallback。
        
        RKNN 转换稳定：避免了动态 pads / 动态 shape 带来的坑。
        
        你提到的 /conv1/Pad 报错怎么规避？
        
        报错本质：Pad 的 pads 长度/维度与输入维度推断不一致（RKNN 的 shape 推断更苛刻）。
        最简单可靠的规避策略：
        
        禁用 tf_like：tf_like=False
        
        这样卷积直接用 padding=(..., ...) 的对称填充，不会生成额外的 Pad 节点（ONNX 里是 Conv 的 attr，不是 Pad op）。
        
        你担心 TF SAME 的非对称性？对 3×3, stride=2 的场景，用对称 padding=1 与 TF SAME 差距非常小，几乎不伤精度，但极大提升端侧稳定性。
        
        保持 3D 池化只在空间维（我已改为 AvgPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1), count_include_pad=False) 的 TFCompatAvgPool3D）。
        
        这是 ONNX 11 + RKNN 友好的固定常量 padding 路径，不会再触发 index_put。
        
        如果你一定要用 TF SAME 的非对称 Pad：也能行，我的 same_padding 会给出常量 pads；但为了 1.7.5 的 RKNN 兼容性，建议优先采用上面的“对称 padding 优先策略”，基本无精度损失。

"""
