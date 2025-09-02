# -*- coding: utf-8 -*-
"""
@Time    : 2025/8/1 13:45
@Author  : Kend
@FileName: export_onnx
@Software: PyCharm
@modifier:
"""

"""
# 结构化剪枝优势：
# ✅ 真正减少计算量
# ✅ 真正加速推理
# ✅ 更好的内存效率

# 结构化剪枝劣势：
# ❌ 实现复杂
# ❌ 可能影响精度更多
# ❌ 需要重新训练微调
# ❌ 对复杂网络结构风险大

当前的目标是模型压缩，而非加速
1. 减少模型大小（已实现：参数减少20%）
2. 为进一步优化做准备（ONNX转换、TFLite）
结构化剪枝实现复杂，对MoViNet这样的复杂网络风险较大


MoviNets中使用了一些ONNX不支持的操作。
Tracing failed sanity checks!
Tensor-valued Constant nodes differed in value across invocations
这与模型中的TemporalCGAvgPool3D有关，它在因果模式下维护状态，导致trace不一致。
"""

import sys
import os
# 把上级目录加入环境变量
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("root_dir=", root_dir)
sys.path.append(root_dir)
os.chdir(root_dir)
from torch import nn
from torch.nn.utils import prune
import torch
import torch.quantization
from net.movinet_2d_1d import MoViNetTemporal
from net.cfg import build_movinet_a0_cfg



"""  PyTorch → 剪枝 → 导出ONNX """

def verify_model_output(model):
    """
    验证模型输出
    """
    print("5. 验证模型输出...")

    model.eval()
    dummy_input = torch.randn(1, 3, 16, 224, 224)

    try:
        # PyTorch模型输出
        with torch.no_grad():
            pytorch_output = model(dummy_input)

        print(f"PyTorch模型输出形状: {pytorch_output.shape}")
        print(f"PyTorch模型输出示例: {pytorch_output[0][:5]}")  # 显示前5个值
        return True
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False



def load_and_optimize_model(model_path='checkpoints/movinet_best.pth'):
    """
    加载模型并进行剪枝优化
    """
    print("=== 第一部分：PyTorch模型优化 ===")

    # 1. 加载原始模型
    print("1. 加载原始模型...")
    cfg = build_movinet_a0_cfg()
    model = MoViNetTemporal(cfg, num_classes=2)
    print("模型权重路径:", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # 2. 设置为 ONNX 导出模式
    model.set_export_onnx(True)
    # model.eval()
    model.eval()

    # 记录原始模型信息
    original_params = sum(p.numel() for p in model.parameters())
    original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    print(f"原始模型参数量: {original_params:,}")
    print(f"原始模型大小(FP32): {original_size:.2f} MB")

    # 2. 剪枝（移除冗余参数）
    print("2. 模型剪枝...")
    model = prune_model_properly(model, pruning_ratio=0.2)

    # 3. 保存剪枝后的权重（重要：保存为.pth文件）
    torch.save(model.state_dict(), 'movinet_pruned_weights.pth')
    print("✅ 保存剪枝权重: movinet_pruned_weights.pth")

    # 4. 验证优化后模型
    print("3. 验证优化模型...")
    optimized_params = sum(p.numel() for p in model.parameters())
    print(f"优化后参数量: {optimized_params:,}")
    print(f"参数减少: {(1 - optimized_params / original_params) * 100:.1f}%")

    # 计算实际稀疏率
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                total_zeros += torch.sum(module.weight == 0).item()
                total_params += module.weight.numel()

    if total_params > 0:
        actual_sparsity = total_zeros / total_params * 100
        print(f"实际稀疏率: {actual_sparsity:.1f}%")

    return model

def prune_model_properly(model, pruning_ratio=0.2):
    """
    正确地对模型进行剪枝
    """
    print(f"开始剪枝，剪枝率: {pruning_ratio}")

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"剪枝前参数量: {total_params_before:,}")

    # 对卷积层和全连接层进行剪枝
    pruned_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            try:
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                pruned_layers += 1
            except Exception as e:
                print(f"剪枝模块 {name} 时出错: {e}")
                continue

    print(f"成功剪枝 {pruned_layers} 个层")

    # 移除剪枝的重参数化
    layers_processed = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                try:
                    prune.remove(module, 'weight')
                    layers_processed += 1
                except Exception as e:
                    print(f"移除模块 {name} 的剪枝参数时出错: {e}")
                    continue

    print(f"处理了 {layers_processed} 个层的剪枝参数")

    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"剪枝后参数量: {total_params_after:,}")
    print(f"参数减少: {(1 - total_params_after / total_params_before) * 100:.1f}%")

    return model

def export_to_onnx_with_fixes(model, onnx_path='movinet_optimized.onnx'):
    """
    导出ONNX模型（修复ATen操作问题）
    """
    print("4. 导出ONNX模型")

    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, 3, 16, 224, 224)
    # x = torch.randn(1, 3*16, 172, 172)  # B, C*T, H, W

    try:
        # 修复ATen操作的关键：使用更严格的导出参数
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # dynamic_axes={
            #     'input': {0: 'batch_size'},
            #     'output': {0: 'batch_size'}
            # },
            # 关键：不使用ATen fallback
            # 不添加 operator_export_type 参数
        )

        print(f"✅ ONNX模型导出完成: {onnx_path}")

        if os.path.exists(onnx_path):
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"ONNX文件大小: {onnx_size:.2f} MB")

        return onnx_path

    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        return None

# 更新主函数
def main():
    """
    第一部分主函数
    """
    print("开始第一步：模型剪枝并导出ONNX")

    try:
        # 加载并优化模型
        optimized_model = load_and_optimize_model('/home/kend/Guanxin/work/workspace/movinet-pytorch/train/checkpoints/movinet_2d_lite_20250819-135404.pth')

        # 验证模型
        verify_model_output(optimized_model)

        # 导出为ONNX
        result_path = export_to_onnx_with_fixes(optimized_model, 'movinet_optimized_0819.onnx')

        print("=== 第一部分完成 ===")
        if result_path and result_path.endswith('.onnx'):
            print(f"输出ONNX文件: {result_path}")
        else:
            print("❌ ONNX导出失败")

        return result_path

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

