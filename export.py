# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:26
@Author  : Kend
@FileName: export
@Software: PyCharm
@modifier:
"""


"""
将生成的ONNX文件转换为MCU支持的格式（如TensorFlow Lite Micro）

PyTorch → ONNX → TensorFlow → TensorFlow Lite 转换流程 

# MCU部署前的检查清单
MCU_DEPLOYMENT_CHECKLIST = {
    "模型大小": "确保模型适合MCU内存（通常<10MB）",
    "计算复杂度": "检查FLOPs是否适合MCU计算能力",
    "输入尺寸": "确认输入尺寸（如224x224）是否适合MCU",
    "量化精度": "验证INT8量化后的精度损失",
    "推理时间": "测试单次推理时间是否满足实时要求",
    "内存使用": "监控推理过程中的内存峰值"
}

"""


def optimize_movinet_for_mcu(model_path='checkpoints/movinet_best.pth'):
    """
    完整的MCU优化流程
    """
    print("=== MoViNet MCU优化流程 ===")

    # 1. 加载模型
    print("1. 加载原始模型...")
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. 量化为INT8
    print("2. 模型量化为INT8...")
    quantized_model = quantize_movinet_for_mcu(model)

    # 3. 剪枝冗余参数
    print("3. 剪枝冗余参数...")
    pruned_model = prune_movinet_redundant_weights(quantized_model, pruning_ratio=0.2)
    pruned_model = remove_pruning_reparametrization(pruned_model)

    # 4. 验证模型输出
    print("4. 验证优化后模型...")
    dummy_input = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        original_output = model(dummy_input)
        optimized_output = pruned_model(dummy_input)

        # 检查输出差异
        diff = torch.abs(original_output - optimized_output).mean()
        print(f"优化前后输出差异: {diff.item():.6f}")

    # 5. 保存优化后的模型
    print("5. 保存优化模型...")
    save_quantized_model(pruned_model, 'checkpoints/movinet_mcu_optimized.pth')

    # 6. 导出ONNX
    print("6. 导出ONNX格式...")
    export_quantized_model_to_onnx(pruned_model, 'checkpoints/movinet_mcu_optimized.onnx')

    print("=== MCU优化完成 ===")
    return pruned_model


# 使用示例
if __name__ == "__main__":
    optimized_model = optimize_movinet_for_mcu()



