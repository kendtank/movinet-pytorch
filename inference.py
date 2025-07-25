# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:24
@Author  : Kend
@FileName: inference
@Software: PyCharm
@modifier:
"""
import time

"""
流式推理
"""


# inference.py

import torch
import cv2
import numpy as np
from torchvision import transforms
from net.movinet import MoViNet
from net.cfg import build_movinet_a0_cfg


# def stream_inference(model, video_path, clip_frames=8):
#     cap = cv2.VideoCapture(video_path)
#     model.eval()
#     model.clean_activation_buffers()
#
#     device = next(model.parameters()).device
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     buffer = []
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = transform(frame)  # (C, H, W)
#         buffer.append(frame)
#
#         if len(buffer) == clip_frames:
#             # TODO ✅ 正确的维度调整
#             clip = torch.stack(buffer)         # (T, C, H, W)
#             clip = clip.permute(1, 0, 2, 3)    # (C, T, H, W)
#             clip = clip.unsqueeze(0).to(device)  # (1, C, T, H, W)
#
#             with torch.no_grad():
#                 print("clip.shape:", clip.shape)  # torch.Size([1, 3, 8, 224, 224])
#                 outputs = model(clip)
#                 probs = torch.softmax(outputs, dim=1)
#                 predicted = torch.argmax(outputs, dim=1).item()
#                 print(f"预测类别: {'猫' if predicted == 0 else '狗'}, 置信度: {probs[0][predicted].item():.4f}")
#             buffer = []
#
#     cap.release()
#     print("推理完成")


def stream_inference(model, video_path, clip_frames=8):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    model.eval()
    model.clean_activation_buffers()  # 初始化状态

    device = next(model.parameters()).device
    print("device::", device)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    buffer = []
    clip_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)  # (C, H, W)
        buffer.append(frame)

        if len(buffer) == clip_frames:
            clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1).item()
                print(f"[Clip {clip_count}] 预测类别: {'猫' if predicted == 0 else '狗'}, 置信度: {probs[0][predicted].item():.4f}")
                clip_count += 1
            buffer = []
            # 如果每个 clip 独立推理，取消下面注释
            # model.clean_activation_buffers()

    # 处理剩余帧
    if len(buffer) > 0:
        clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(clip)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1).item()
            print(f"[Clip {clip_count}] 最后一段预测类别: {'猫' if predicted == 0 else '狗'}, 置信度: {probs[0][predicted].item():.4f}")

    cap.release()
    print(f"总用时: {time.time() - start_time:.2f}秒")
    print("推理完成")


if __name__ == '__main__':
    cfg = build_movinet_a0_cfg()
    model = MoViNet(cfg, causal=True, pretrained=False, num_classes=2, conv_type="2plus1d", tf_like=True)
    model.load_state_dict(torch.load('checkpoints/movinet_best.pth'))
    model.eval()
    stream_inference(model, 'tt2.mp4')


