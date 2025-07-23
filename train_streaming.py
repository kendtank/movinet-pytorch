# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:28
@Author  : Kend
@FileName: train_streaming
@Software: PyCharm
@modifier:
"""


"""
流式训练
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import transforms
from load_dataset_with_video import VideoDataset
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg


def train_streaming():
    # 参数配置
    data_root = 'dataset/train'
    val_root = 'dataset/val'
    batch_size = 1
    num_epochs = 100
    learning_rate = 3e-4
    num_classes = 2   # 猫拆家, 狗拆家
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 日志和模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/movinet_a0_streaming_{timestamp}'
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 数据增强 适用于单张图像
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = VideoDataset(root_dir=data_root, transform=transform, max_frames=256)
    val_dataset = VideoDataset(root_dir=val_root, transform=transform, max_frames=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=num_classes, conv_type="2plus1d", tf_like=True)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 多尺度 clip 设置
            n_clips = torch.randint(2, 5, (1,)).item()
            clip_frames = torch.randint(4, 16, (1,)).item()
            loss_total = 0

            model.clean_activation_buffers()
            optimizer.zero_grad()

            for j in range(n_clips):
                start = j * clip_frames
                end = start + clip_frames
                if end > inputs.shape[2]:
                    break
                clip = inputs[:, :, start:end]
                outputs = model(clip)
                loss = criterion(outputs, labels) / n_clips
                loss.backward()
                loss_total += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            model.clean_activation_buffers()

            running_loss += loss_total
            total += labels.size(0)

            # 推理最后一次 clip 的结果作为准确率
            with torch.no_grad():
                outputs = model(clip)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证
        model.eval()
        val_loss, val_acc = validate_streaming(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

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


def validate_streaming(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.clean_activation_buffers()

            n_clips = 5
            clip_frames = 8
            for j in range(n_clips):
                start = j * clip_frames
                end = start + clip_frames
                if end > inputs.shape[2]:
                    break
                clip = inputs[:, :, start:end]
                outputs = model(clip)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), correct / total


if __name__ == '__main__':
    train_streaming()
