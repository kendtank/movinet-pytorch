# export_wrapper.py
import torch
import os
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg


class ExportableMoViNet(torch.nn.Module):
    """用于导出的MoViNet包装类"""

    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        # 直接调用原始模型的前向传播
        return self.original_model(x)

    @property
    def causal(self):
        return self.original_model.causal


def create_exportable_model():
    """创建可用于导出的模型"""
    print("=== 创建可用于导出的模型 ===")

    # 加载原始模型
    cfg = build_movinet_a0_cfg()
    original_model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)

    if os.path.exists('movinet_pruned.pth'):
        original_model.load_state_dict(torch.load('movinet_pruned.pth', map_location='cpu'))
        print("✅ 加载剪枝权重")
    else:
        print("❌ 未找到剪枝权重文件")
        return None

    original_model.eval()

    # 清除激活缓冲区
    if hasattr(original_model, 'clean_activation_buffers'):
        original_model.clean_activation_buffers()

    # 创建包装模型
    export_model = ExportableMoViNet(original_model)
    export_model.eval()

    return export_model


def export_with_wrapper():
    """使用包装模型导出"""
    print("=== 使用包装模型导出 ===")

    model = create_exportable_model()
    if model is None:
        return False

    dummy_input = torch.randn(1, 3, 16, 224, 224)

    # 测试包装模型
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ 包装模型测试成功: {output.shape}")
    except Exception as e:
        print(f"❌ 包装模型测试失败: {e}")
        return False

    # 尝试导出
    try:
        torch.onnx.export(
            model,
            dummy_input,
            'movinet_wrapper.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

        print("✅ 包装模型导出成功: movinet_wrapper.onnx")
        return True

    except Exception as e:
        print(f"❌ 包装模型导出失败: {e}")
        return False


def try_different_export_strategies():
    """尝试不同的导出策略"""
    print("=== 尝试不同的导出策略 ===")

    model = create_exportable_model()
    if model is None:
        return False

    dummy_input = torch.randn(1, 3, 16, 224, 224)

    strategies = [
        ("基本导出", lambda: basic_export(model, dummy_input)),
        ("无动态轴导出", lambda: export_without_dynamic_axes(model, dummy_input)),
        ("固定输入导出", lambda: export_with_fixed_input(model, dummy_input)),
    ]

    for name, strategy in strategies:
        print(f"\n尝试 {name}...")
        if strategy():
            print(f"✅ {name} 成功")
            return True
        else:
            print(f"❌ {name} 失败")

    return False


def basic_export(model, dummy_input):
    """基本导出"""
    try:
        torch.onnx.export(
            model,
            dummy_input,
            'movinet_basic.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        return True
    except:
        return False


def export_without_dynamic_axes(model, dummy_input):
    """无动态轴导出"""
    try:
        torch.onnx.export(
            model,
            dummy_input,
            'movinet_no_dynamic.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        return True
    except:
        return False


def export_with_fixed_input(model, dummy_input):
    """固定输入导出"""
    try:
        torch.onnx.export(
            model,
            dummy_input,
            'movinet_fixed_input.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        return True
    except:
        return False


if __name__ == "__main__":
    success = try_different_export_strategies()
    if not success:
        print("\n❌ 所有导出策略都失败了")
    else:
        print("\n✅ 至少有一种策略成功了")
