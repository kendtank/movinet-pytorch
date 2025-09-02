import torch
import os
import sys
# 添加上次目录到项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.movinet_4d import MoViNet4D

# ======== 配置 ========
pth_path = "/home/kend/Guanxin/work/workspace/movinet-pytorch/train/checkpoints/movinet_2d_lite_20250819-151700.pth"
onnx_path = "movinet4d_rknn.onnx"
num_classes = 2
dummy_input_shape = (1, 3, 16, 224, 224)  # B,C,T,H,W

# ======== 加载训练好的模型 ========
model = MoViNet4D(num_classes=num_classes)
checkpoint = torch.load(pth_path,map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()


# ======== 导出 ONNX ========
dummy_input = torch.randn(*dummy_input_shape)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    # dynamic_axes={
    #     'input': {0:'batch', 2:'frames'},
    #     'output': {0:'batch'}
    # },
    opset_version=11
)
print("ONNX exported:", onnx_path)
