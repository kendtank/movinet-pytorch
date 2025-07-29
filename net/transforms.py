# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/23 14:29
@Author  : Kend
@FileName: transforms
@Software: PyCharm
@modifier:
"""


"""
数据增强
1. 提升模型泛化能力
    增加数据多样性，防止过拟合
    提高模型对不同光照、色彩条件的鲁棒性
2. MoViNet训练的最佳实践
    官方Kinetics训练中也使用了数据增强
    有助于提升最终的准确率
"""


import torchvision.transforms as transforms

class VideoTransform:
    def __init__(self, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, frame):
        return self.transform(frame)


"""
总结
建议进行适度的数据增强，重点是：
    保持时间一致性（对所有帧使用相同的增强参数）
    专注于颜色和光照变化的增强
    避免破坏动作语义的几何变换
    根据具体任务调整增强强度
    这样既能提升模型性能，又不会破坏视频数据的时间连续性特征。
"""