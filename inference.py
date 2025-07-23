# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:24
@Author  : Kend
@FileName: inference
@Software: PyCharm
@modifier:
"""


"""
流式推理
"""


import torch
import cv2
import numpy as np
from torchvision import transforms
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg


def stream_inference(model, video_path, clip_frames=8):
    cap = cv2.VideoCapture(video_path)
    model.eval()
    model.clean_activation_buffers()

    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        buffer.append(frame)

        if len(buffer) == clip_frames:
            clip = torch.stack(buffer).unsqueeze(0).to(device)  # (1, C, T, H, W)
            with torch.no_grad():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1).item()
                print(f"预测类别: {'猫' if predicted == 0 else '狗'}, 置信度: {probs[0][predicted].item():.4f}")
            buffer = []

    cap.release()
    print("推理完成")



if __name__ == '__main__':
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)
    model.load_state_dict(torch.load('checkpoints/movinet_best.pth'))
    model.eval()
    stream_inference(model, 'test_video.mp4')

