"""
你修改的 movinet_lite.py 和原生 movinet.py 的结构一致性。
    结构对比分析
    相同点：
    整体架构：
        都有 stem layer (conv1)
        都有多个 block 层
        都有 final layer (conv7)
        都有分类头
        模块设计思路：
        都使用了倒残差结构 (Inverted Residual)
        都包含 expand -> deep -> se -> project 的流程
        都有 shortcut 连接

不同点（关键差异）：
    模块
    原生 MoViNet
    修改版 MoViNet Lite
        卷积操作
            3D卷积或2+1D分解卷积
            空间2D卷积 + 时间1D卷积
    SE模块
        3D全局平均池化 + 3D卷积
        空间全局平均池化 + 1D时间卷积
    时间建模
        3D卷积直接建模时空特征
        分解为空间和时间两个独立操作
    激活函数
        Swish/Hardsigmoid
        ReLU（默认）
    归一化
        BatchNorm3d
        BatchNorm2d


原始配置 (cfg.py)：
# conv1: 3 -> 8
# blocks:
#   stage1: 8 -> 8 (expand: 24)
#   stage2: 8 -> 32 (expand: 80) -> 32 -> 32
#   stage3: 32 -> 56 (expand: 184) -> 56 -> 56
#   stage4: 56 -> 56 (expand: 184) -> 56 -> 56 -> 56
#   stage5: 56 -> 104 (expand: 384) -> 104 -> 104 -> 104
# conv7: 104 -> 480

修改版 (movinet_lite.py) 当前实现：
# conv1: 3 -> 32 (与原版不同)
# blocks:
#   stage1: 32 -> 64 (expand: 96)
#   stage2: 64 -> 128 (expand: 192)
#   stage3: 128 -> 256 (expand: 384)
# conv7: 256 -> 256



结论
你的修改版没有完全保持原始 MoViNet 的结构，主要差异包括：
通道数不一致：修改版使用了不同的通道配置，从原始的 A0 架构改为自定义的更大通道数
层数不一致：原始有5个block stage，修改版只有3个
卷积分解方式不同：虽然都是2+1D思想，但实现细节不同
建议的改进方案
如果你希望更准确地评估精度损失，建议：
保持通道数一致：按照原始 cfg.py 的配置修改 movinet_lite.py
"""


# 修改 MoViNet2D1DBackbone 以匹配原始配置
class MoViNet2D1DBackbone(nn.Module):
    def __init__(self, cfg, in_ch=3):
        super().__init__()
        # conv1
        conv1_cfg = cfg.conv1
        self.conv1 = ConvBlock2DTemporal(
            in_ch, conv1_cfg.out_channels,
            spatial_kernel=conv1_cfg.kernel_size[1:],
            temporal_kernel=conv1_cfg.kernel_size[0],
            stride=conv1_cfg.stride
        )

        # blocks - 按照原始配置构建
        blocks_list = []
        for stage_blocks in cfg.blocks:
            for block_cfg in stage_blocks:
                blocks_list.append(BasicBneckTemporal(
                    block_cfg.input_channels,
                    block_cfg.out_channels,
                    block_cfg.expanded_channels,
                    kernel_size=block_cfg.kernel_size,
                    stride=block_cfg.stride
                ))
        self.blocks = nn.Sequential(*blocks_list)

        # conv7
        conv7_cfg = cfg.conv7
        self.conv7 = ConvBlock2DTemporal(
            conv7_cfg.input_channels,
            conv7_cfg.out_channels,
            spatial_kernel=conv7_cfg.kernel_size[1:],
            temporal_kernel=conv7_cfg.kernel_size[0],
            stride=conv7_cfg.stride
        )
        self.out_channels = conv7_cfg.out_channels


"""
保持激活函数一致：使用 Swish/Hardswish 替代 ReLU
保持 SE 模块结构一致：确保 reduction ratio 等参数一致
这样修改后，才能更准确地评估由于 3D->2D+1D 转换带来的精度损失。
"""