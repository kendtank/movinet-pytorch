# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:26
@Author  : Kend
@FileName: export
@Software: PyCharm
@modifier:
"""


"""
ONNX 格式
TensorFlow Lite 格式（用于部署在边缘设备）
"""

import torch
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg



def export_to_onnx(model, save_path='movinet.onnx'):
    dummy_input = torch.randn(1, 3, 8, 224, 224)  # B, C, T, H, W
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'T'},  # 可变帧数
            'output': {}
        }
    )
    print(f"ONNX 模型已保存到 {save_path}")


def export_to_tflite(model, save_path='movinet.tflite'):
    model.eval()
    dummy_input = torch.randn(1, 3, 8, 224, 224)

    # 导出 TorchScript
    script_model = torch.jit.script(model)
    torch.jit.save(script_model, 'movinet_script.pt')

    # 使用 TorchScript 转换为 TFLite（需安装 torchscript2tflite）

    # 常规做法是使用转换工具或导出为 ONNX 再转 TFLite # TODO
    print(f"TFLite 模型已保存到 {save_path}")




if __name__ == '__main__':
    # 加载模型
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)
    model.load_state_dict(torch.load('checkpoints/movinet_best.pth'))
    model.eval()

    # 导出 ONNX
    export_to_onnx(model)

    # 导出 TFLite（需 ONNX 转换）
    export_to_tflite(model)
