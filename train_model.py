# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 10:48
@Author  : Kend
@FileName: train_model
@Software: PyCharm
@modifier:
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms
from load_dataset_with_video import VideoDataset
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg


def train_model():
    # 参数配置
    data_root = 'dataset/train'
    val_root = 'dataset/val'
    batch_size = 4
    num_epochs = 50
    learning_rate = 3e-4
    num_classes = 2
    num_frames = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志和模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/movinet_a0_pet_destruction_{timestamp}'
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 数据增强
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集
    train_dataset = VideoDataset(root_dir=data_root, transform=transform, num_frames=num_frames)
    val_dataset = VideoDataset(root_dir=val_root, transform=transform, num_frames=num_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=False, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # TensorBoard  # TODO 环境出现了问题, 需要额外新的py环境做隔离查看-0723
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

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证
        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion, device)
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
    print("Training complete.")


def validate(model, val_loader, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), correct / total


if __name__ == '__main__':
    train_model()

