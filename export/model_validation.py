# verify_exported_model.py
import onnx
import onnxruntime as ort
import torch
import numpy as np

"""
ℹ️  由于包含ATen操作，跳过ONNX Runtime测试(测试不了, 因为onnx不支持这个算子)
   但这不影响转换到TensorFlow(需要测试)
"""


def analyze_exported_model():
    """分析导出的ONNX模型"""
    print("=== 分析导出的ONNX模型 ===")

    # 1. 加载并检查模型
    model = onnx.load("movinet_optimized.onnx")
    onnx.checker.check_model(model)
    print("✅ ONNX模型结构检查通过")

    # 2. 统计ATen操作
    aten_ops = []
    for node in model.graph.node:
        if 'ATen' in node.op_type or 'org.pytorch' in str(node.domain):
            aten_ops.append(node)

    print(f"🔍 ATen操作数量: {len(aten_ops)}")
    if aten_ops:
        print("⚠️  模型包含ATen操作，可能无法直接用于ONNX Runtime")
        print("   但可以用于onnx-tf转换到TensorFlow")
    else:
        print("✅ 模型无ATen操作，可直接使用")

    # 3. 检查输入输出
    print(f"📥 输入: {[inp.name for inp in model.graph.input]}")
    print(f"📤 输出: {[out.name for out in model.graph.output]}")

    # 4. 测试是否能加载到ONNX Runtime（如果无ATen操作）
    if len(aten_ops) == 0:
        try:
            sess = ort.InferenceSession("movinet_optimized.onnx")
            dummy_input = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: dummy_input})
            print(f"✅ ONNX Runtime推理成功: {output[0].shape}")
        except Exception as e:
            print(f"❌ ONNX Runtime推理失败: {e}")
    else:
        print("ℹ️  由于包含ATen操作，跳过ONNX Runtime测试")
        print("   但这不影响转换到TensorFlow")


if __name__ == "__main__":
    analyze_exported_model()
