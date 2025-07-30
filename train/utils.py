# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/30 18:20
@Author  : Kend
@FileName: utils
@Software: PyCharm
@modifier:
"""


import torch
from train.dataset import get_batch_adaptive_params, load_video_clip_adaptive_strategy, adaptive_clip_strategy


def check_model_learning_capability(model, data_loader, device, logger):
    """
    检查模型是否具备基本的学习能力, 为了验证模型的拟合能力
    """
    model.eval()
    with torch.no_grad():
        # 获取一个batch的数据
        for video_paths, targets in data_loader:
            targets = targets.to(device)
            break

        # 在处理前清理缓冲区
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        # 使用固定参数处理一个clip
        _, batch_n_clip_frames = get_batch_adaptive_params(video_paths, 8, 16)

        clip_frames = []
        for i, video_path in enumerate(video_paths):
            _, _, strategy, total_frames = adaptive_clip_strategy(video_path, 128, batch_n_clip_frames)
            frames = load_video_clip_adaptive_strategy(video_path, 0, batch_n_clip_frames, strategy, total_frames)

            if isinstance(frames, list):
                frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]
            if isinstance(frames, list):
                frames = torch.stack(frames).permute(1, 0, 2, 3)

            clip_frames.append(frames)

        clip_frames = torch.stack(clip_frames).to(device)

        # 前向传播
        output = model(clip_frames)

        if logger:
            logger.info(f"Model check - Targets: {targets.cpu().numpy()}")
            logger.info(f"Model check - Output: {output.cpu().numpy()}")
            logger.info(f"Model check - Output std: {output.std().item():.4f}")


# def evaluate_streaming_adaptive(
#         model,
#         data_loader,
#         base_n_clips=8,
#         base_n_clip_frames=16,
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         logger = None
# ):
#     """
#     自适应流式评估处理不同长度视频
#     """
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total_samples = 0
#
#     # TODO add: 添加调试信息
#     all_predictions = []
#     all_targets = []
#
#     with torch.no_grad():
#         for video_paths, targets in data_loader:
#             targets = targets.to(device)
#
#             # 记录真实标签
#             all_targets.extend(targets.cpu().numpy())
#
#
#             # 清理模型的激活缓冲区
#             if hasattr(model, 'clean_activation_buffers'):
#                 model.clean_activation_buffers()
#
#             # 获取batch级别自适应参数
#             batch_n_clips, batch_n_clip_frames = get_batch_adaptive_params(
#                 video_paths, base_n_clips, base_n_clip_frames
#             )
#
#             # 存储所有clip的输出用于集成
#             clip_outputs = []
#
#             for clip_idx in range(batch_n_clips):
#
#                 # 注意：不要在clip之间清理缓冲区，让模型维持时序状态
#                 # 只有在处理完一个完整视频后才清理缓冲区
#
#                 clip_frames = []
#                 for i, video_path in enumerate(video_paths):
#                     # 获取该视频的具体策略
#                     _, _, strategy, total_frames = adaptive_clip_strategy(
#                         video_path, 128, batch_n_clip_frames
#                     )
#
#                     # 加载帧
#                     frames = load_video_clip_adaptive_strategy(
#                         video_path, clip_idx, batch_n_clip_frames, strategy, total_frames
#                     )
#
#                     # 应用变换（评估时不使用数据增强）
#                     if isinstance(frames, list):
#                         frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]
#
#                     if isinstance(frames, list):
#                         frames = torch.stack(frames).permute(1, 0, 2, 3)
#
#                     clip_frames.append(frames)
#
#                 clip_frames = torch.stack(clip_frames).to(device)
#
#                 # 前向传播
#                 output = model(clip_frames)
#                 clip_outputs.append(output)
#
#                 # # # 不要在clip之间清理缓冲区  TODO:
#                 # if hasattr(model, 'clean_activation_buffers'):
#                 #     model.clean_activation_buffers()
#
#             # 集成所有clip的输出（平均）
#             if clip_outputs:
#                 final_output = torch.stack(clip_outputs).mean(dim=0)
#
#                 # # 检查输出是否正常 test- 正常
#                 # if logger:
#                 #     logger.info(f"Output range: [{final_output.min():.4f}, {final_output.max():.4f}]")
#                 #     logger.info(f"Output std: {final_output.std():.4f}")
#
#                 loss = F.cross_entropy(final_output, targets)
#                 total_loss += loss.item()
#
#                 pred = torch.argmax(final_output, dim=1)
#
#                 # 记录预测结果
#                 all_predictions.extend(pred.cpu().numpy())
#
#                 correct += pred.eq(targets).sum().item()
#                 total_samples += targets.size(0)
#
#                 # # TODO: 添加调试信息
#                 # if logger:
#                 #     logger.info(f"Batch targets: {targets.cpu().numpy()}, predictions: {pred.cpu().numpy()}")
#                 #     logger.info(f"Batch output logits: {final_output.cpu().numpy()}")
#
#
#     avg_loss = total_loss / len(data_loader)
#     accuracy = 100. * correct / total_samples
#
#     # 添加整体预测分布信息
#     if logger:
#         import numpy as np
#         unique_preds, counts = np.unique(all_predictions, return_counts=True)
#         unique_targets, target_counts = np.unique(all_targets, return_counts=True)
#         logger.info(f'Prediction distribution: {dict(zip(unique_preds, counts))}')
#         logger.info(f'Target distribution: {dict(zip(unique_targets, target_counts))}')
#         logger.info(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%')
#
#
#     return avg_loss, accuracy