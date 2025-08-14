# -*- coding: utf-8 -*-
"""
@Time    : 2025/8/1 16:01
@Author  : Kend
@FileName: analyze_onnx_model
@Software: PyCharm
@modifier:
"""
# analyze_onnx_model.py
import onnx
import torch
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg
import os


def analyze_onnx_model(model_path):
    """详细分析ONNX模型"""
    print(f"=== 分析ONNX模型: {model_path} ===")

    try:
        # 加载模型
        model = onnx.load(model_path)
        print("✅ 模型加载成功")

        # 基本信息
        print(f"IR版本: {model.ir_version}")
        print(f"Opset版本: {model.opset_import[0].version}")
        print(f"Producer: {model.producer_name if model.producer_name else 'Unknown'}")

        # 分析节点
        print(f"\n=== 节点分析 ===")
        print(f"总节点数: {len(model.graph.node)}")

        # 统计操作类型
        op_types = {}
        aten_ops = []

        for i, node in enumerate(model.graph.node):
            # 统计操作类型
            op_type = node.op_type
            op_types[op_type] = op_types.get(op_type, 0) + 1

            # 检查ATen操作
            if 'ATen' in node.op_type or 'org.pytorch' in node.domain:
                aten_ops.append({
                    'index': i,
                    'name': node.name,
                    'op_type': node.op_type,
                    'domain': node.domain,
                    'inputs': list(node.input),
                    'outputs': list(node.output)
                })

        # 显示操作类型统计
        print("\n操作类型统计:")
        for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_type}: {count}")

        # 显示ATen操作详情
        if aten_ops:
            print(f"\n⚠️  发现 {len(aten_ops)} 个ATen操作:")
            for op in aten_ops[:10]:  # 只显示前10个
                print(f"  [{op['index']}] {op['op_type']} ({op['domain']})")
                print(f"      名称: {op['name']}")
                print(f"      输入: {op['inputs']}")
                print(f"      输出: {op['outputs']}")
            if len(aten_ops) > 10:
                print(f"      ... 还有 {len(aten_ops) - 10} 个ATen操作")
        else:
            print("✅ 未发现ATen操作")

        # 分析输入输出
        print(f"\n=== 输入输出信息 ===")
        print("输入:")
        for inp in model.graph.input:
            print(f"  {inp.name}")

        print("输出:")
        for out in model.graph.output:
            print(f"  {out.name}")

        # 检查初始化器
        print(f"\n=== 初始化器信息 ===")
        print(f"初始化器数量: {len(model.graph.initializer)}")

        return model, aten_ops

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_model_compatibility(model_path):
    """检查模型兼容性"""
    print(f"=== 检查模型兼容性 ===")

    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("✅ ONNX模型结构检查通过")
        return True
    except Exception as e:
        print(f"❌ ONNX模型检查失败: {e}")
        return False


def export_model_without_aten_fallback():
    """重新导出模型，不使用ATen fallback"""
    print("=== 重新导出模型 ===")

    try:
        # 加载模型
        cfg = build_movinet_a0_cfg()
        model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)

        if os.path.exists('movinet_pruned.pth'):
            model.load_state_dict(torch.load('movinet_pruned.pth', map_location='cpu'))
            print("✅ 加载剪枝权重")
        else:
            print("⚠️  未找到剪枝权重文件")
            return False

        model.eval()

        # 清除激活缓冲区
        if hasattr(model, 'clean_activation_buffers'):
            model.clean_activation_buffers()

        # 创建输入
        dummy_input = torch.randn(1, 3, 16, 224, 224)

        # 测试模型
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ 模型测试输出: {output.shape}")

        # 导出（不使用ATen fallback）
        print("开始导出...")
        torch.onnx.export(
            model,
            dummy_input,
            'movinet_optimized_fixed.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'time'},
                'output': {0: 'batch_size'}
            }
            # 不使用 operator_export_type 参数！
        )

        print("✅ 模型导出完成: movinet_optimized_fixed.onnx")
        return True

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    model_path = 'movinet_optimized.onnx'

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    # 1. 分析当前模型
    model, aten_ops = analyze_onnx_model(model_path)

    # 2. 检查兼容性
    is_compatible = check_model_compatibility(model_path)

    # 3. 如果有ATen操作，尝试重新导出
    if aten_ops:
        print(f"\n=== 建议 ===")
        print("检测到ATen操作，建议重新导出模型...")

        success = export_model_without_aten_fallback()
        if success:
            print("✅ 请使用新导出的模型: movinet_optimized_fixed.onnx")
        else:
            print("❌ 重新导出失败")
    else:
        print("✅ 模型应该可以正常使用")


if __name__ == "__main__":
    main()
