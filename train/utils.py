# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/30 18:20
@Author  : Kend
@FileName: utils
@Software: PyCharm
@modifier:
"""

import os
import torch
from train.dataset import get_batch_adaptive_params, load_video_clip_adaptive_strategy, adaptive_clip_strategy





def check_model_learning_capability(model, data_loader, device, logger, dsize):
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
        _, batch_n_clip_frames, length_category = get_batch_adaptive_params(video_paths, 8, 16)

        clip_frames = []
        for i, video_path in enumerate(video_paths):
            _, _, strategy, total_frames = adaptive_clip_strategy(video_path, 128, batch_n_clip_frames)
            frames = load_video_clip_adaptive_strategy(video_path, 0, batch_n_clip_frames, strategy, total_frames, dsize, logger)

            if isinstance(frames, list):
                frames = [torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 for frame in frames]
            if isinstance(frames, list):
                frames = torch.stack(frames).permute(1, 0, 2, 3)

            clip_frames.append(frames)

        clip_frames = torch.stack(clip_frames).to(device)

        # 前向传播
        output = model(clip_frames)

        if logger:
            logger.info(f"Model check - Input shape: {clip_frames.shape}")
            logger.info(f"Model check - Targets: {targets.cpu().numpy()}")
            logger.info(f"Model check - Output shape: {output.shape}")
            # logger.info(f"Model check - Output: {output.cpu().numpy()}")
            # logger.info(f"Model check - Output std: {output.std().item():.4f}")

            # # 添加softmax后的概率分布
            # if output.shape[-1] > 1:  # 确保是分类输出
            #     probs = torch.softmax(output, dim=-1)
            #     logger.info(f"Model check - Probabilities: {probs.cpu().numpy()}")