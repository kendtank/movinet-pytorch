# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/29 15:34
@Author  : Kend
@FileName: train
@Software: PyCharm
@modifier:
"""


"""
训练自定义数据集的MoViNet
支持:
    流式训练模式 (Streaming Mode)
        适用于长视频（>50帧）
        使用因果模式 (causal=True)
        分段处理长视频，节省内存
        适合实时推理场景
    批量训练模式 (Batch Mode) -- 使用场景用不到,不使用
        适用于较短视频（<50帧）
        可以使用非因果模式 (causal=False)
        一次性处理整个视频
"""


import argparse
import os
import torch
from datetime import datetime
from net.cfg import build_movinet_a0_cfg



def parse_args():
    parser = argparse.ArgumentParser(description='MoViNet Training')

    # 基础参数
    parser.add_argument('--data_root', type=str, default='dataset/train', help='训练数据路径')
    parser.add_argument('--val_root', type=str, default='dataset/val', help='验证数据路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')

    # 模型参数
    parser.add_argument('--model_version', type=str, default='A0', choices=['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
                        help='模型版本')
    parser.add_argument('--causal', action='store_true', default=True, help='是否使用因果模式(流式处理)')  # 默认为True
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--conv_type', type=str, default='2plus1d', help='卷积类型')

    # 训练模式参数
    parser.add_argument('--training_mode', type=str, default='streaming', choices=['streaming', 'batch'],
                        help='训练模式')
    parser.add_argument('--max_frames', type=int, default=256, help='最大帧数(batch模式)')
    parser.add_argument('--n_clips', type=int, default=8, help='clip数量(streaming模式)')
    parser.add_argument('--n_clip_frames', type=int, default=16, help='每个clip的帧数(streaming模式)')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default='runs', help='日志保存路径')

    return parser.parse_args()


def get_model_config(args):
    """根据参数获取模型配置, 暂时只提供A0"""
    if args.model_version == 'A0':
        cfg = build_movinet_a0_cfg()
    else:
        raise NotImplementedError(f"Model version {args.model_version} is not implemented.")

    return cfg


def main():
    args = parse_args()

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = args.save_dir
    log_dir = f'{args.log_dir}/movinet_{args.model_version}_{args.training_mode}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # 确保使用流式训练模式和因果模式
    if args.training_mode != 'streaming':
        print("Warning: For your use case, streaming mode is recommended. Switching to streaming mode.")
        args.training_mode = 'streaming'

    if not args.causal:
        print("Warning: For streaming mode, causal mode is required. Enabling causal mode.")
        args.causal = True

    print(f"Training started with {args.training_mode} mode")
    print(f"Model: MoViNet-{args.model_version}")
    print(f"Causal mode: {args.causal}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Number of clips: {args.n_clips}")
    print(f"Frames per clip: {args.n_clip_frames}")

    # 导入流式训练函数
    try:
        from train.trainer import train_streaming_adaptive
    except ImportError:
        # 如果从train模块导入失败，尝试直接导入
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from train.trainer import train_streaming_adaptive

    # 准备训练参数
    config = {
        'data_root': args.data_root,
        'val_root': args.val_root,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'num_classes': args.num_classes,
        'base_n_clips': args.n_clips,
        'base_n_clip_frames': args.n_clip_frames,
        'device': torch.device(args.device),
        'save_dir': args.save_dir,
        'dsize': (224, 224)
    }

    print(f"Starting streaming training with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 开始训练
    train_streaming_adaptive(**config)
    print("Training completed successfully!")


if __name__ == '__main__':
    main()



# # 自定义参数训练
# python train.py --data_root dataset/train --val_root dataset/val --batch_size 1 --num_epochs 50 --n_clips 8 --n_clip_frames 16