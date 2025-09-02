# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/15
@Author  : Kend
@FileName: dataset_loader
@Software: PyCharm
@Description: tensorflow官方的MoViNet视频数据集加载方式，支持滑动窗口、等比例缩放、数据增强、中文日志
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from train.transforms import VideoTransformPad


def logger(info):
    print(f"2025-08-15 INFO - {info}")


# ---------------------------
# 视频数据集
# ---------------------------
class VideoDataset(Dataset):
    def __init__(self, data_root, mode='train', clip_len=16, transform=None, logger_fn=logger):
        """
        data_root: dataset根目录
        mode: 'train' 或 'val'
        """
        self.root_dir = os.path.join(data_root, mode)
        self.clip_len = clip_len
        self.transform = transform
        self.is_train = True if mode == 'train' else False
        self.logger = logger_fn

        # 存储数据
        self.clips = []
        self.labels = []
        self.class_to_idx = {}

        self._build_dataset()

    def _build_dataset(self):
        self.logger(f"开始构建数据集 [{ 'train' if self.is_train else 'val' }] ...")
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir,d))])
        total_clips = 0
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            class_path = os.path.join(self.root_dir, cls)
            video_files = glob(os.path.join(class_path, '*.mp4')) + glob(os.path.join(class_path, '*.avi'))
            cls_clip_count = 0
            for vf in video_files:
                cap = cv2.VideoCapture(vf)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # 滑动窗口 T/2
                # for start in range(0, total_frames - self.clip_len + 1, self.clip_len // 2):
                # 滑动窗口 T/4
                for start in range(0, total_frames - self.clip_len + 1, self.clip_len // 4):
                    self.clips.append((vf, start))
                    self.labels.append(idx)
                    cls_clip_count += 1

            self.logger(f"类别 [{cls}] -> {cls_clip_count} 个 clip")
            total_clips += cls_clip_count

        self.logger(f"数据集 [{'train' if self.is_train else 'val'}] 总 clip 数量: {total_clips}")

    def debug_clips(self, n=10):
        """打印前 n 个 clip 信息，并显示第一个视频的完整切片帧序列"""
        self.logger(f"调试模式：前 {n} 个 clip")
        for i in range(min(n, len(self.clips))):
            video_path, start_idx = self.clips[i]
            label = self.labels[i]
            self.logger(f"[{i + 1}] 视频={os.path.basename(video_path)}, 起始帧={start_idx}, 标签={label}")

        # 找第一个视频的所有 clip
        if len(self.clips) > 0:
            first_video_path = self.clips[0][0]
            self.logger(f"第一个视频: {os.path.basename(first_video_path)} 所有 clip 帧序列:")
            frame_sequences = []

            for vf, start_idx in self.clips:
                if vf != first_video_path:
                    break
                frame_sequences.append(list(range(start_idx, start_idx + self.clip_len)))

            for i, seq in enumerate(frame_sequences):
                self.logger(f"clip {i + 1}: {seq}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        video_path, start_idx = self.clips[index]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []

        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.tensor(frame).permute(2,0,1).float() / 255.0
            frames.append(frame)
        cap.release()

        if len(frames) < self.clip_len:
            return None  # DataLoader collate_fn 会过滤

        clip_tensor = torch.stack(frames)          # [T, C, H, W]
        clip_tensor = clip_tensor.permute(1,0,2,3) # -> [C, T, H, W]
        label = self.labels[index]
        return clip_tensor, label


# ---------------------------
# collate_fn
# ---------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    clips, labels = zip(*batch)
    return torch.stack(clips), torch.tensor(labels)


# ---------------------------
# 构建 DataLoader 示例
# ---------------------------
if __name__ == "__main__":
    data_root = '/home/kend/Guanxin/work/workspace/movinet-pytorch/dataset'
    clip_len = 16
    batch_size = 2
    num_workers = 4

    train_dataset = VideoDataset(data_root, mode='train', clip_len=clip_len, transform=VideoTransformPad(is_train=True))
    val_dataset = VideoDataset(data_root, mode='val', clip_len=clip_len, transform=VideoTransformPad(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)

    train_dataset.debug_clips(n=10)
    val_dataset.debug_clips(n=10)

    logger("DataLoader 构建完成")



"""
2025-08-15 INFO - 开始构建数据集 [train] ...
2025-08-15 INFO - 类别 [pet_destruction] -> 1389 个 clip
2025-08-15 INFO - 类别 [pet_normal] -> 1505 个 clip
2025-08-15 INFO - 数据集 [train] 总 clip 数量: 2894
2025-08-15 INFO - 开始构建数据集 [val] ...
2025-08-15 INFO - 类别 [pet_destruction] -> 615 个 clip
2025-08-15 INFO - 类别 [pet_normal] -> 629 个 clip
2025-08-15 INFO - 数据集 [val] 总 clip 数量: 1244
2025-08-15 INFO - 调试模式：前 10 个 clip
2025-08-15 INFO - [1] 视频=dog_action_026.mp4, 起始帧=0, 标签=0
2025-08-15 INFO - [2] 视频=dog_action_026.mp4, 起始帧=8, 标签=0
2025-08-15 INFO - [3] 视频=dog_action_026.mp4, 起始帧=16, 标签=0
2025-08-15 INFO - [4] 视频=dog_action_026.mp4, 起始帧=24, 标签=0
2025-08-15 INFO - [5] 视频=dog_action_026.mp4, 起始帧=32, 标签=0
2025-08-15 INFO - [6] 视频=dog_action_026.mp4, 起始帧=40, 标签=0
2025-08-15 INFO - [7] 视频=dog_action_026.mp4, 起始帧=48, 标签=0
2025-08-15 INFO - [8] 视频=cat_action_train_027.mp4, 起始帧=0, 标签=0
2025-08-15 INFO - [9] 视频=cat_action_train_027.mp4, 起始帧=8, 标签=0
2025-08-15 INFO - [10] 视频=cat_action_train_027.mp4, 起始帧=16, 标签=0
2025-08-15 INFO - 第一个视频: dog_action_026.mp4 所有 clip 帧序列:
2025-08-15 INFO - clip 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
2025-08-15 INFO - clip 2: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
2025-08-15 INFO - clip 3: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
2025-08-15 INFO - clip 4: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
2025-08-15 INFO - clip 5: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
2025-08-15 INFO - clip 6: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
2025-08-15 INFO - clip 7: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
2025-08-15 INFO - 调试模式：前 10 个 clip
2025-08-15 INFO - [1] 视频=dog_action_val_011.mp4, 起始帧=0, 标签=0
2025-08-15 INFO - [2] 视频=dog_action_val_011.mp4, 起始帧=8, 标签=0
2025-08-15 INFO - [3] 视频=dog_action_val_011.mp4, 起始帧=16, 标签=0
2025-08-15 INFO - [4] 视频=dog_action_val_011.mp4, 起始帧=24, 标签=0
2025-08-15 INFO - [5] 视频=dog_action_val_011.mp4, 起始帧=32, 标签=0
2025-08-15 INFO - [6] 视频=dog_action_val_011.mp4, 起始帧=40, 标签=0
2025-08-15 INFO - [7] 视频=dog_action_val_011.mp4, 起始帧=48, 标签=0
2025-08-15 INFO - [8] 视频=cat_action_val_003.mp4, 起始帧=0, 标签=0
2025-08-15 INFO - [9] 视频=cat_action_val_003.mp4, 起始帧=8, 标签=0
2025-08-15 INFO - [10] 视频=cat_action_val_003.mp4, 起始帧=16, 标签=0
2025-08-15 INFO - 第一个视频: dog_action_val_011.mp4 所有 clip 帧序列:
2025-08-15 INFO - clip 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
2025-08-15 INFO - clip 2: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
2025-08-15 INFO - clip 3: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
2025-08-15 INFO - clip 4: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
2025-08-15 INFO - clip 5: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
2025-08-15 INFO - clip 6: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
2025-08-15 INFO - clip 7: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
2025-08-15 INFO - DataLoader 构建完成
"""