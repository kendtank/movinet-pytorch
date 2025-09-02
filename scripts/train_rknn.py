"""
MoViNet-RKNN 训练/评估脚本
用于训练和评估为瑞芯微 RK-NPU 优化的 MoViNet-A0 模型
包含数据加载、训练循环、验证、模型保存和 ONNX 导出功能
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的MoViNet-RKNN模型
from net.movinet_rknn import MoViNetRKNNA0, export_onnx

# ======== 数据加载 ========
class VideoDataset(torch.utils.data.Dataset):
    """视频数据集示例类
    实际使用时需要根据您的数据格式进行修改
    """
    def __init__(self, data_root, split='train', clip_len=16, frame_size=224, transform=None):
        self.data_root = data_root
        self.split = split
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.transform = transform
        
        # 示例：假设文件夹按类别组织
        self.classes = sorted(os.listdir(data_root))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有视频路径和标签
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for video_file in os.listdir(cls_dir):
                # 这里假设视频已经被预处理为帧或可以直接读取
                self.samples.append((os.path.join(cls_dir, video_file), self.class_to_idx[cls_name]))
        
        # 简单的数据集分割
        if split == 'train':
            self.samples = self.samples[:int(0.8 * len(self.samples))]
        else:
            self.samples = self.samples[int(0.8 * len(self.samples)):]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        video_path, label = self.samples[index]
        
        # 示例：读取视频帧并预处理
        # 实际应用中需要根据您的数据格式修改
        # 这里我们创建一个随机张量作为示例
        # 真实场景下应该读取实际的视频帧
        frames = torch.randn(3, self.clip_len, self.frame_size, self.frame_size)
        
        # 应用数据增强
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label

# ======== 数据增强 ========
def get_transforms(frame_size=224):
    """获取数据增强变换
    针对视频数据的简单增强策略，专为端侧部署优化
    """
    train_transform = transforms.Compose([
        # 随机裁剪（空间维度）
        lambda x: F.interpolate(x, size=(None, frame_size, frame_size), mode='bilinear', align_corners=False),
        # 随机水平翻转（空间维度）
        transforms.RandomHorizontalFlip(p=0.5),
        # 色彩抖动
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 归一化 - 使用 ImageNet 的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        # 中心裁剪（空间维度）
        lambda x: F.interpolate(x, size=(None, frame_size, frame_size), mode='bilinear', align_corners=False),
        # 归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

# ======== 训练器 ========
class Trainer:
    def __init__(self, args):
        self.args = args
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 1. 创建模型
        self.model = MoViNetRKNNA0(
            num_classes=args.num_classes,
            export_T=args.clip_len
        ).to(self.device)
        
        # 2. 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 3. 设置优化器 - 使用 AdamW 更好地适应端侧量化
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 4. 设置学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        # 5. 准备数据
        self._prepare_data()
        
        # 6. 创建日志和检查点目录
        self.log_dir = os.path.join(args.output_dir, 'logs')
        self.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 7. 初始化训练指标
        self.best_val_acc = 0.0

    def _prepare_data(self):
        """准备数据集和数据加载器"""
        train_transform, val_transform = get_transforms(self.args.frame_size)
        
        # 创建数据集
        if self.args.data_root:
            train_dataset = VideoDataset(
                self.args.data_root,
                split='train',
                clip_len=self.args.clip_len,
                frame_size=self.args.frame_size,
                transform=train_transform
            )
            val_dataset = VideoDataset(
                self.args.data_root,
                split='val',
                clip_len=self.args.clip_len,
                frame_size=self.args.frame_size,
                transform=val_transform
            )
        else:
            # 使用随机数据进行测试
            print("Warning: No data root specified. Using synthetic data for testing.")
            train_dataset = torch.utils.data.TensorDataset(
                torch.randn(100, 3, self.args.clip_len, self.args.frame_size, self.args.frame_size),
                torch.randint(0, self.args.num_classes, (100,))
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.randn(20, 3, self.args.clip_len, self.args.frame_size, self.args.frame_size),
                torch.randint(0, self.args.num_classes, (20,))
            )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失和准确率
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印训练进度
            if batch_idx % self.args.log_interval == 0:
                acc = 100. * correct / total
                elapsed_time = time.time() - start_time
                print(f'Epoch: {epoch}/{self.args.epochs} | '\
                      f'Batch: {batch_idx}/{len(self.train_loader)} | '\
                      f'Loss: {train_loss/(batch_idx+1):.3f} | '\
                      f'Acc: {acc:.2f}% | '\
                      f'Time: {elapsed_time:.2f}s')
        
        return train_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 记录损失和准确率
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(self.val_loader)
        
        print(f'Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.2f}%')
        
        return val_loss, val_acc

    def train(self):
        """完整的训练循环"""
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch} / {self.args.epochs}")
            print('-' * 50)
            
            # 训练一个epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            
            # 验证模型
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
            
            # 定期保存检查点
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 导出最佳模型为ONNX
        self.export_model()

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存模型检查点"""
        checkpoint_name = f"model_epoch_{epoch}_val_acc_{val_acc:.2f}.pth"
        if is_best:
            checkpoint_name = "model_best.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'args': self.args
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")

    def export_model(self):
        """导出模型为ONNX格式"""
        # 加载最佳模型
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pth")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # 导出ONNX
        onnx_path = os.path.join(self.args.output_dir, f"movinet_rknn_a0_{self.args.num_classes}cls.onnx")
        export_onnx(
            self.model,
            onnx_path,
            T=self.args.clip_len,
            H=self.args.frame_size,
            W=self.args.frame_size,
            opset=11
        )

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Current best validation accuracy: {self.best_val_acc:.2f}%")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

# ======== 主函数 ========
def main():
    # 设置随机种子以保证结果可复现
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train MoViNet-RKNN model for RK-NPU deployment')
    parser.add_argument('--data-root', type=str, default='', help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./output_rknn', help='Directory to save outputs')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--clip-len', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--frame-size', type=int, default=224, help='Frame height and width')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=10, help='Log training status every N batches')
    parser.add_argument('--save-interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--export-only', action='store_true', help='Only export the model to ONNX')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 加载检查点（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 执行训练/评估/导出
    if args.export_only:
        trainer.export_model()
    elif args.eval_only:
        trainer.validate()
    else:
        trainer.train()

if __name__ == '__main__':
    main()

# ======== 使用方法示例 ========
"""
# 1. 训练模型
python scripts/train_rknn.py --data-root /path/to/your/dataset --num-classes 2 --batch-size 8 --epochs 30

# 2. 评估模型
python scripts/train_rknn.py --eval-only --resume ./output_rknn/checkpoints/model_best.pth

# 3. 仅导出模型
python scripts/train_rknn.py --export-only --resume ./output_rknn/checkpoints/model_best.pth

# 4. 恢复训练
python scripts/train_rknn.py --resume ./output_rknn/checkpoints/model_epoch_10_val_acc_85.50.pth
"""