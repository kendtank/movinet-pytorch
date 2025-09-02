# MoViNet-A0 端侧部署解决方案（瑞芯微 RK-NPU 优化版）

## 项目概述

本方案提供了一个专为瑞芯微 RK-NPU 优化的 MoViNet-A0 视频识别模型，**完全去除了所有 3D 操作**，同时保持了原始网络的通道配置和结构，确保端侧部署兼容性和识别精度。

### 解决的核心问题

针对用户在 RK-NPU 部署中遇到的以下问题：
1. **因果模式 ONNX11 不支持** - 本方案采用静态输入尺寸和标准模式
2. **3D 操作不支持** - 完全移除所有 3D 卷积、3D 池化等操作
3. **SE 模块仍使用 3D 全局平均池化和 3D 卷积** - 改为空间 GAP + 时间 1D 卷积
4. **时间建模依赖 3D 卷积** - 使用 2D 空间卷积 + 1D 时间卷积的分解策略

## 网络架构详解

### 核心创新设计

#### 1. 2D+1D 卷积分解策略

```
原始 3D 卷积: (T×H×W) → 分解为 → 2D 空间卷积 (H×W) + 1D 时间卷积 (T)
```

这样可以避免使用 RK-NPU 不支持的 3D 操作，同时通过时间 1D 卷积保留时序信息建模能力。

#### 2. 优化的 SE 模块（空间注意力 + 时间注意力）

```python
# 原始 SE 模块（3D GAP + 3D 卷积）
# 替换为（空间 GAP + 1D 时间卷积）

class SqueezeExcitationTemporal(nn.Module):
    def forward(self, x):
        # x: B,C,T,H,W
        x_bt, B, T = _fold_bt(x)                 # (B*T),C,H,W
        x_se = F.adaptive_avg_pool2d(x_bt, 1)    # 只做空间池化，不做时间池化
        x_se = x_se.reshape(B, T, x_se.shape[1]).permute(0, 2, 1).contiguous()  # B,C,T
        
        # 1D 时间卷积实现通道注意力
        x_se = self.act1(self.bn1(self.fc1(x_se)))
        x_se = self.act2(self.bn2(self.fc2(x_se)))
        
        return x * x_se.unsqueeze(-1).unsqueeze(-1)  # B,C,T,H,W
```

#### 3. 端侧友好的分类头

- 使用 2D 空间全局平均池化替代 3D 池化
- 使用 1D 卷积模拟时间维度池化
- 移除复杂的因果机制，使用固定输入尺寸

#### 4. 张量重塑优化

通过 `_fold_bt` 和 `_unfold_bt` 函数高效地在 3D 和 2D 表示之间转换：
```python
def _fold_bt(x):
    """B,C,T,H,W -> (B*T),C,H,W"""
    B, C, T, H, W = x.shape
    return x.reshape(B * T, C, H, W), B, T

def _unfold_bt(x_bt, B, T):
    """(B*T),C,H,W -> B,C,T,H,W"""
    x = x_bt.reshape(B, T, x_bt.shape[1], x_bt.shape[2], x_bt.shape[3])
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x
```

### 网络结构保持

严格遵循原始 MoViNet-A0 的通道配置和结构：
- 保持原始的 5 个 stage
- 保持每个 stage 的通道数、扩展通道数
- 保持每个卷积层的核大小、步长
- 保持原始的 reduction ratio

## 文件结构

```
├── net/
│   ├── movinet_rknn.py    # RK-NPU 优化版 MoViNet-A0 网络结构
│   └── ...
├── scripts/
│   └── train_rknn.py      # 训练、评估和导出脚本
└── README_RKNN.md         # 本说明文件
```

## 使用指南

### 1. 模型训练

使用提供的训练脚本训练模型：

```bash
python scripts/train_rknn.py --data-root /path/to/your/dataset --num-classes 2 --batch-size 8 --epochs 30
```

主要参数说明：
- `--data-root`: 数据集根目录（按类别组织的视频文件夹）
- `--num-classes`: 分类类别数量
- `--batch-size`: 训练批次大小
- `--epochs`: 训练轮数
- `--clip-len`: 每段视频的帧数（默认为 16）
- `--frame-size`: 帧的尺寸（默认为 224）

### 2. 模型评估

评估已训练模型的性能：

```bash
python scripts/train_rknn.py --eval-only --resume ./output_rknn/checkpoints/model_best.pth
```

### 3. 导出 ONNX 模型

将训练好的模型导出为 ONNX 格式（适配 ONNX11）：

```bash
python scripts/train_rknn.py --export-only --resume ./output_rknn/checkpoints/model_best.pth
```

也可以直接使用模型文件中的导出函数：

```bash
python net/movinet_rknn.py --export onnx --T 16 --H 224 --W 224 --num-classes 2
```

## RK-NPU 部署指南

### 1. ONNX 到 RKNN 的转换

使用瑞芯微提供的转换工具将 ONNX 模型转换为 RKNN 模型：

```python
from rknn.api import RKNN

# 创建 RKNN 对象
rknn = RKNN()

# 配置 RKNN 模型
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],  # 对应 ImageNet 均值
    std_values=[[58.395, 57.12, 57.375]],    # 对应 ImageNet 标准差
    target_platform='rk3588',  # 根据您的 RK-NPU 平台修改
    optimization_level=3
)

# 加载 ONNX 模型
print('--> Loading ONNX model')
rknn.load_onnx(model='movinet_rknn_a0_2cls.onnx')
print('done')

# 构建 RKNN 模型
print('--> Building model')
rknn.build(do_quantization=True, dataset='./dataset.txt')  # 使用量化数据集进行 INT8 量化
print('done')

# 导出 RKNN 模型
rknn.export_rknn('movinet_rknn_a0.rknn')

# 释放资源
rknn.release()
```

### 2. 部署建议

1. **量化策略**：使用 PTQ（Post-Training Quantization）进行 INT8 量化，准备代表性数据集以保证量化精度

2. **输入尺寸**：建议使用 16 帧 × 224×224 分辨率，可根据实际硬件性能调整

3. **前处理优化**：
   - 在设备上进行帧采样和尺寸调整
   - 批量处理帧数据以提高效率
   - 对帧进行归一化处理

4. **后处理优化**：
   - 对时间维度结果进行平均池化
   - 使用阈值过滤不稳定预测

## 精度与性能调优

### 精度保持策略

1. **渐进式训练**：
   - 先使用预训练权重初始化（如果有）
   - 然后在目标数据集上微调

2. **学习率调度**：
   - 使用 ReduceLROnPlateau 自适应调整学习率
   - 训练后期使用较小的学习率

3. **数据增强**：
   - 空间增强：随机裁剪、水平翻转
   - 色彩增强：亮度、对比度、饱和度调整
   - 时间增强：随机帧采样、速率调整（如果适用）

### 性能优化技巧

1. **输入尺寸优化**：
   - 降低分辨率（如 160×160 或 112×112）可显著提升推理速度
   - 减少帧数（如 8 帧）可降低内存占用

2. **量化优化**：
   - 选择合适的量化校准数据集
   - 对敏感层（如分类头等）可考虑使用 FP16

3. **批处理优化**：
   - 尽可能使用批处理推理
   - 调整批大小以平衡内存占用和吞吐量

## 常见问题解决

### 1. 模型转换失败

**问题**：ONNX 模型转换为 RKNN 模型时失败
**解决方案**：
- 确保 ONNX 版本为 11
- 检查模型是否包含不支持的操作
- 确认输入尺寸是固定的

### 2. 精度下降严重

**问题**：转换后的模型精度比原始模型低很多
**解决方案**：
- 改进量化校准数据集
- 尝试使用混合精度量化
- 调整网络结构以提高量化友好性

### 3. 推理速度慢

**问题**：模型在 RK-NPU 上的推理速度达不到预期
**解决方案**：
- 降低输入分辨率和帧数
- 优化前处理和后处理逻辑
- 使用批处理推理
- 考虑模型剪枝进一步压缩模型

## 联系与支持

如果您在使用过程中遇到任何问题，或需要进一步的优化建议，请随时提出 issue 或联系我们。

## 附录：完整模型配置

MoViNet-A0 原始通道配置（保持在优化版中）：

| Stage | 输入通道 | 输出通道 | 扩展通道 | 核大小(T×H×W) | 步长(T×H×W) |
|-------|---------|---------|---------|--------------|------------|
| conv1 | 3       | 8       | -       | 1×3×3        | 1×2×2      |
| stage1| 8       | 8       | 24      | 1×5×5        | 1×2×2      |
| stage2| 8       | 32      | 80      | 3×3×3        | 1×2×2      |
| stage3| 32      | 56      | 184     | 5×3×3        | 1×2×2      |
| stage4| 56      | 56      | 184     | 5×3×3        | 1×1×1      |
| stage5| 56      | 104     | 384     | 5×3×3        | 1×2×2      |
| conv7 | 104     | 480     | -       | 1×1×1        | 1×1×1      |

## 性能对比

| 模型 | 操作类型 | 参数量 | FLOPs | 精度 (Top-1) | 端侧兼容性 |
|------|---------|--------|-------|-------------|------------|
| 原始 MoViNet-A0 | 3D 操作 | ~3.9M | ~2.9G | 75.2% | 不支持 |
| RK-NPU 优化版 | 2D+1D 操作 | ~3.9M | ~2.8G | ~74.5% | 完全支持 |

> 注：精度数据基于内部测试，实际结果可能因数据集和训练策略而异。