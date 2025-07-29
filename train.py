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
    批量训练模式 (Batch Mode)
        适用于较短视频（<50帧）
        可以使用非因果模式 (causal=False)
        一次性处理整个视频
"""


import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.optim as optim
from load_dataset_with_video import VideoDataset, StreamingVideoDataset
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg, build_movinet_a1_cfg, build_movinet_a2_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='MoViNet Training')

    # 基础参数
    parser.add_argument('--data_root', type=str, default='dataset/train', help='训练数据路径')
    parser.add_argument('--val_root', type=str, default='dataset/val', help='验证数据路径')
    parser.add_argument('--batch_size', type=int, default=2, help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')

    # 模型参数
    parser.add_argument('--model_version', type=str, default='A0', choices=['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
                        help='模型版本')
    parser.add_argument('--causal', action='store_true', help='是否使用因果模式(流式处理)')
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
    elif args.model_version == 'A1':
        cfg = build_movinet_a1_cfg()
    elif args.model_version == 'A2':
        cfg = build_movinet_a2_cfg()
    else:
        cfg = build_movinet_a0_cfg()

    return cfg



def main():
    args = parse_args()

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = args.save_dir
    log_dir = f'{args.log_dir}/movinet_{args.model_version}_{args.training_mode}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # 获取模型配置
    cfg = get_model_config(args)

    # 初始化模型
    model = MoViNet(
        cfg,
        causal=args.causal,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        conv_type=args.conv_type
    )
    model = model.to(args.device)

    # 根据训练模式选择数据集和训练函数
    if args.training_mode == 'streaming':
        # 流式训练模式
        train_dataset = StreamingVideoDataset(
            root_dir=args.data_root,
            transform=None,
            clip_frames=args.n_clip_frames
        )
        val_dataset = StreamingVideoDataset(
            root_dir=args.val_root,
            transform=None,
            clip_frames=args.n_clip_frames
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # 使用流式训练函数
        # train_streaming(model, train_loader, val_loader, args)

    else:
        # 批量训练模式
        train_dataset = VideoDataset(
            root_dir=args.data_root,
            transform=None,
            max_frames=args.max_frames
        )
        val_dataset = VideoDataset(
            root_dir=args.val_root,
            transform=None,
            max_frames=args.max_frames
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # 使用批量训练函数
        # train_batch(model, train_loader, val_loader, args)

    print(f"Training started with {args.training_mode} mode")
    print(f"Model: MoViNet-{args.model_version}")
    print(f"Causal mode: {args.causal}")
    print(f"Pretrained: {args.pretrained}")



if __name__ == '__main__':
    main()
