"""
MoViNet-RKNN 模型测试脚本
用于快速验证模型的基本功能，包括前向传播、ONNX导出和性能评估
"""

import os
import sys
import time
import argparse
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的MoViNet-RKNN模型
from net.movinet_rknn import MoViNetRKNNA0, export_onnx

# ======== 模型测试 ========
def test_model(args):
    """测试模型的基本功能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = MoViNetRKNNA0(
        num_classes=args.num_classes,
        export_T=args.clip_len
    ).to(device)
    model.eval()
    
    print("Model created successfully!")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Input clip length: {args.clip_len}")
    print(f"Input frame size: {args.frame_size}x{args.frame_size}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    
    # 创建随机输入张量
    input_tensor = torch.randn(1, 3, args.clip_len, args.frame_size, args.frame_size).to(device)
    print(f"Input shape: {input_tensor.shape}")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()
    
    print(f"Forward pass completed successfully!")
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
    
    # 检查输出是否合理
    output_softmax = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output_softmax, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Class probabilities: {output_softmax.cpu().numpy()[0]}")
    
    # 评估多次推理的平均时间
    if args.benchmark:
        print("\nBenchmarking inference speed...")
        num_runs = 50
        total_time = 0.0
        
        with torch.no_grad():
            for i in range(num_runs):
                # 创建新的随机输入以避免缓存影响
                input_tensor = torch.randn(1, 3, args.clip_len, args.frame_size, args.frame_size).to(device)
                
                start_time = time.time()
                model(input_tensor)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                # 打印进度
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{num_runs} runs completed")
        
        avg_time = (total_time / num_runs) * 1000  # 转换为毫秒
        fps = 1000 / avg_time  # 每秒帧数
        
        print(f"Average inference time over {num_runs} runs: {avg_time:.2f} ms")
        print(f"Estimated FPS: {fps:.2f}")
    
    # 导出ONNX模型
    if args.export_onnx:
        print("\nExporting model to ONNX...")
        onnx_path = os.path.join(args.output_dir, f"movinet_rknn_a0_{args.num_classes}cls.onnx")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 导出ONNX
        export_onnx(
            model,
            onnx_path,
            T=args.clip_len,
            H=args.frame_size,
            W=args.frame_size,
            opset=11
        )
        
        # 检查ONNX文件是否创建成功
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # 转换为MB
            print(f"ONNX model exported successfully!")
            print(f"ONNX file path: {onnx_path}")
            print(f"ONNX file size: {file_size:.2f} MB")
        else:
            print("Failed to export ONNX model!")

# ======== 主函数 ========
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Test MoViNet-RKNN model functionality')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--clip-len', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--frame-size', type=int, default=224, help='Frame height and width')
    parser.add_argument('--output-dir', type=str, default='./output_onnx', help='Directory to save ONNX model')
    parser.add_argument('--export-onnx', action='store_true', help='Export model to ONNX')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark to measure inference speed')
    
    args = parser.parse_args()
    
    # 测试模型
    test_model(args)
    
    print("\nModel test completed!")
    print("\nNext steps:")
    print("1. Use scripts/train_rknn.py to train the model on your dataset")
    print("2. Export the trained model to ONNX")
    print("3. Convert ONNX to RKNN format using Rockchip's conversion tools")
    print("4. Deploy on RK-NPU device")

if __name__ == '__main__':
    main()

# ======== 使用方法示例 ========
"""
# 1. 基本测试 - 验证模型功能
python scripts/test_rknn_model.py

# 2. 自定义参数测试
python scripts/test_rknn_model.py --num-classes 5 --clip-len 8 --frame-size 160

# 3. 导出ONNX模型
python scripts/test_rknn_model.py --export-onnx --output-dir ./my_onnx_models

# 4. 运行性能基准测试
python scripts/test_rknn_model.py --benchmark

# 5. 完整测试（导出+性能测试）
python scripts/test_rknn_model.py --export-onnx --benchmark --num-classes 2 --clip-len 16 --frame-size 224
"""