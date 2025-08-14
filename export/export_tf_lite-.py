# analyze_and_fix_onnx.py
import onnx
import os


def analyze_onnx_issues():
    """分析ONNX模型中的问题"""
    print("=== 分析ONNX模型问题 ===")

    onnx_path = "movinet_optimized.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX文件不存在: {onnx_path}")
        return

    try:
        # 加载模型
        model = onnx.load(onnx_path)

        print("模型基本信息:")
        print(f"  IR版本: {model.ir_version}")
        print(f"  Opset版本: {model.opset_import[0].version}")
        print(f"  节点总数: {len(model.graph.node)}")

        # 分析操作类型
        op_types = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_types[op_type] = op_types.get(op_type, 0) + 1

        print("\n操作类型统计 (前10):")
        sorted_ops = sorted(op_types.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_ops[:10]:
            print(f"  {op_type}: {count}")

        # 特别关注ATen操作
        aten_ops = [node for node in model.graph.node if 'ATen' in node.op_type]
        print(f"\nATen操作: {len(aten_ops)} 个")
        if aten_ops:
            for i, op in enumerate(aten_ops):
                print(f"  {i + 1}. {op.name} ({op.op_type})")
                print(f"     输入: {list(op.input)}")
                print(f"     输出: {list(op.output)}")

        # 检查输入输出
        print(f"\n模型输入:")
        for inp in model.graph.input:
            print(f"  {inp.name}")

        print(f"\n模型输出:")
        for out in model.graph.output:
            print(f"  {out.name}")

        return len(aten_ops)

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return -1


if __name__ == "__main__":
    aten_count = analyze_onnx_issues()
    if aten_count > 0:
        print(f"\n⚠️  模型包含 {aten_count} 个ATen操作")
        print("建议:")
        print("1. 尝试使用onnx-tf命令行工具")
        print("2. 如果失败，考虑手动重构模型")
        print("3. 或者接受精度上的轻微损失")
