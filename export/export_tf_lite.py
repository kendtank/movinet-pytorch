import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import os


def simple_onnx_to_tf():
    """ONNX到TF转换"""
    print("=== 简化版ONNX到TF转换 ===")

    onnx_path = "movinet_optimized.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX文件不存在: {onnx_path}")
        return False

    try:
        # 加载ONNX模型
        print("加载ONNX模型...")
        onnx_model = onnx.load(onnx_path)

        # 检查模型
        print("检查ONNX模型...")
        onnx.checker.check_model(onnx_model)

        # 显示ATen操作信息
        aten_ops = [node for node in onnx_model.graph.node if 'ATen' in node.op_type]
        print(f"发现 {len(aten_ops)} 个ATen操作")
        if aten_ops:
            print("ATen操作详情:")
            for i, op in enumerate(aten_ops):
                print(f"  {i + 1}. {op.op_type} - {op.name}")

        # 尝试转换
        print("准备TensorFlow表示...")
        tf_rep = prepare(onnx_model)

        # 导出模型
        print("导出TensorFlow模型...")
        output_dir = "movinet_tf_simple"
        tf_rep.export_graph(output_dir)

        print(f"✅ 转换成功: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        print("这可能是因为ATen操作不被支持")
        return False


def check_existing_conversion():
    """检查已有的转换结果"""
    print("=== 检查已有的转换结果 ===")

    possible_dirs = ["movinet_tf_model", "movinet_tf_simple"]

    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            print(f"📁 找到目录: {dir_name}")

            # 检查关键文件
            key_files = ["saved_model.pb", "variables"]
            for key_file in key_files:
                path = os.path.join(dir_name, key_file)
                if os.path.exists(path):
                    if os.path.isfile(path):
                        size = os.path.getsize(path) / (1024 * 1024)  # MB
                        print(f"  ✅ {key_file}: {size:.2f} MB")
                    else:
                        print(f"  ✅ {key_file}: 目录存在")
                else:
                    print(f"  ⚠️  {key_file}: 不存在")

            # 列出所有文件
            print("  文件列表:")
            for root, dirs, files in os.walk(dir_name):
                level = root.replace(dir_name, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"  {indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  {subindent}{file} ({size:.2f} KB)")

            return dir_name

    print("❌ 未找到任何转换结果")
    return None


def attempt_tflite_conversion(tf_model_path):
    """尝试转换为TFLite"""
    print("=== 尝试TFLite转换 ===")

    if not tf_model_path or not os.path.exists(tf_model_path):
        print(f"❌ TensorFlow模型路径无效: {tf_model_path}")
        return False

    try:
        # 创建TFLite转换器
        print("创建TFLite转换器...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

        # 设置优化选项
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 尝试转换
        print("执行转换...")
        tflite_model = converter.convert()

        # 保存模型
        output_path = "movinet_model.tflite"
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # 显示结果
        size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ TFLite转换成功: {output_path} ({size:.2f} MB)")

        # 显示模型信息
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("模型信息:")
        print(f"  输入: {input_details[0]['shape']} (dtype: {input_details[0]['dtype']})")
        print(f"  输出: {output_details[0]['shape']} (dtype: {output_details[0]['dtype']})")

        return True

    except Exception as e:
        print(f"❌ TFLite转换失败: {e}")
        return False


if __name__ == "__main__":
    print("开始处理ONNX到TFLite转换...")

    # 1. 首先检查是否已有转换结果
    existing_model = check_existing_conversion()

    if not existing_model:
        # 2. 如果没有，尝试转换
        print("\n尝试ONNX到TF转换...")
        success = simple_onnx_to_tf()
        if success:
            existing_model = "movinet_tf_simple"

    # 3. 尝试转换为TFLite
    if existing_model:
        print(f"\n使用模型: {existing_model}")
        attempt_tflite_conversion(existing_model)
    else:
        print("\n❌ 没有可用的TensorFlow模型进行TFLite转换")
