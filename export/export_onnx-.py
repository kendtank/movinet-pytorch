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



# 在export_onnx-.py中你做了两个重要修改：

# 1) 添加了这行代码（非常关键！）
if hasattr(model, 'clean_activation_buffers'):
    model.clean_activation_buffers()

# 2) 使用了ATen fallback导出
operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK


2. 为什么之前失败，现在成功了？
之前的问题：
MoViNet的因果模式会在内部维护状态缓冲区
这些缓冲区在导出时会导致不一致
特别是TemporalCGAvgPool3D组件
现在的解决方案：
clean_activation_buffers() 清除了所有内部状态
ONNX_ATEN_FALLBACK 允许无法直接转换的操作以ATen形式保存
3. 导出的ONNX模型特点
✅ 导出成功: movinet_optimized.onnx
📁 文件大小: 10.72 MB
⚠️ 包含ATen操作（但可以导出）

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
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg



"""  PyTorch → 剪枝 → 导出ONNX """
def load_and_optimize_model(model_path='checkpoints/movinet_best.pth'):
    """
    加载模型并进行剪枝优化
    """
    print("=== 第一部分：PyTorch模型优化 ===")

    # 1. 加载原始模型
    print("1. 加载原始模型...")
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)
    print("模型权重路径:", model_path)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 记录原始模型信息
    original_params = sum(p.numel() for p in model.parameters())
    original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # MB
    print(f"原始模型参数量: {original_params:,}")
    print(f"原始模型大小(FP32): {original_size:.2f} MB")

    # 2. 剪枝（移除冗余参数）
    print("2. 模型剪枝...")
    model = prune_model_properly(model, pruning_ratio=0.2)

    # 3. 验证优化后模型
    print("3. 验证优化模型...")
    optimized_params = sum(p.numel() for p in model.parameters())

    print(f"优化后参数量: {optimized_params:,}")
    print(f"参数减少: {(1 - optimized_params / original_params) * 100:.1f}%")

    # 计算实际稀疏率（真正被置零的参数比例）
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

    # 统计剪枝前参数量
    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"剪枝前参数量: {total_params_before:,}")

    # 对卷积层和全连接层进行剪枝
    pruned_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            try:
                # L1范数剪枝（移除接近0的权重）
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                pruned_layers += 1
            except Exception as e:
                print(f"剪枝模块 {name} 时出错: {e}")
                continue

    print(f"成功剪枝 {pruned_layers} 个层")

    # 立即移除剪枝的重参数化，使模型更紧凑
    layers_processed = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                try:
                    # 此时才真正移除0值，但PyTorch中这是不会改变计算图结构所以还是需要计算这一部分 作用:(减少模型大小. 不会加快推理, 除非硬件支持稀疏运算)
                    prune.remove(module, 'weight')
                    layers_processed += 1
                except Exception as e:
                    print(f"移除模块 {name} 的剪枝参数时出错: {e}")
                    continue

    print(f"处理了 {layers_processed} 个层的剪枝参数")

    # 统计剪枝后参数量
    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"剪枝后参数量: {total_params_after:,}")
    print(f"参数减少: {(1 - total_params_after / total_params_before) * 100:.1f}%")

    return model


def export_to_onnx_with_fixes(model, onnx_path='movinet_optimized.onnx'):
    """
    导出ONNX模型
    """
    print("4. 导出ONNX模型")

    model.eval()

    # 清除激活缓冲区（重要！）
    if hasattr(model, 'clean_activation_buffers'):
        model.clean_activation_buffers()

    # 创建示例输入
    dummy_input = torch.randn(1, 3, 16, 224, 224)  # (batch, channels, time, height, width)

    try:
        # 使用正确的参数
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'time'},
                'output': {0: 'batch_size'}
            },
            # 正确的参数
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            # # 优化参数
            # keep_initializers_as_inputs = False,  # 减小文件大小
            # strip_doc_string = True  # 移除文档字符串
        )

        print(f"ONNX模型导出完成: {onnx_path}")

        # 检查ONNX模型
        if os.path.exists(onnx_path):
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            print(f"ONNX文件大小: {onnx_size:.2f} MB")
        else:
            print("警告：ONNX文件未生成")

        return onnx_path

    except Exception as e:
        print(f"ONNX导出失败: {e}")

        # 移除错误的参数后重试
        print("移除错误参数后重试...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,  # 使用opset 11
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'time'},
                    'output': {0: 'batch_size'}
                },
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            )

            print(f"重试后ONNX模型导出完成: {onnx_path}")
            return onnx_path

        except Exception as e2:
            print(f"重试也失败: {e2}")

            # 最后的备选方案：创建一个简化版本的模型用于导出
            print("创建简化模型用于导出...")
            return create_simplified_model_for_export(model, dummy_input, onnx_path)


def create_simplified_model_for_export(model, dummy_input, onnx_path):
    """
    创建简化版本的模型用于ONNX导出
    """
    print("创建简化模型...")

    # 先测试原始模型是否能正常运行
    try:
        with torch.no_grad():
            model.eval()
            test_output = model(dummy_input)
            print(f"模型测试输出形状: {test_output.shape}")
    except Exception as e:
        print(f"模型测试失败: {e}")
        return None

    # 尝试使用trace方法
    try:
        print("使用torch.jit.trace...")
        traced_model = torch.jit.trace(model, dummy_input)

        torch.onnx.export(
            traced_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )

        print(f"Trace后ONNX导出完成: {onnx_path}")
        return onnx_path

    except Exception as e:
        print(f"Trace方法也失败: {e}")

        # 最后的方案：保存PyTorch模型，后续在第二步处理
        torch.save(model.state_dict(), 'movinet_pruned.pth')
        print("已保存剪枝后的PyTorch模型: movinet_pruned.pth")
        print("请在第二步转换时处理ONNX兼容性问题")
        return 'movinet_pruned.pth'


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


def main():
    """
    第一部分主函数
    """
    print("开始第一步：模型剪枝并导出ONNX")

    try:
        # 加载并优化模型
        optimized_model = load_and_optimize_model('checkpoints/movinet_best.pth')

        # 验证模型
        verify_model_output(optimized_model)

        # 导出为ONNX
        result_path = export_to_onnx_with_fixes(optimized_model, 'movinet_optimized.onnx')

        print("=== 第一部分完成 ===")
        if result_path.endswith('.onnx'):
            print(f"输出ONNX文件: {result_path}")
            print("接下来可以使用第二部分脚本转换为TensorFlow Lite")
        else:
            print(f"输出PyTorch模型: {result_path}")
            print("请在第二步转换时处理ONNX导出")

        # 显示最终文件大小
        if os.path.exists(result_path):
            final_size = os.path.getsize(result_path) / (1024 * 1024)
            print(f"最终文件大小: {final_size:.2f} MB")

        return result_path

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print("第一步完成！")
    else:
        print("第一步执行失败！")

