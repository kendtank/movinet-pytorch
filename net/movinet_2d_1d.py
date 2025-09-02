import torch
import torch.nn as nn
from einops import rearrange

# 2D Conv + BN + Activation
class Conv2dBNActivation(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn = norm_layer(out_ch)
        self.act = activation_layer() if activation_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Temporal 1D Conv 聚合
class TemporalConv1D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x_t = x.mean(dim=[3,4])  # [B, C, T]
        x_t = self.conv(x_t)
        x = x * x_t.unsqueeze(-1).unsqueeze(-1)  # 广播回 H, W
        return x

# 2D + Temporal Conv Block
class ConvBlock2DTemporal(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_kernel=(3,3), temporal_kernel=3, stride=(1,1,1), activation_layer=nn.ReLU):
        super().__init__()
        self.spatial_conv = Conv2dBNActivation(in_ch, out_ch, kernel_size=spatial_kernel, stride=stride[1:], padding=(spatial_kernel[0]//2, spatial_kernel[1]//2), activation_layer=activation_layer)
        self.temporal_conv = TemporalConv1D(out_ch, kernel_size=temporal_kernel, padding=temporal_kernel//2)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial_conv(x)
        _, C2, H2, W2 = x.shape
        x = rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T)
        x = self.temporal_conv(x)
        return x

# SE模块
class SqueezeExcitationTemporal(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(8, channels // reduction)
        self.fc1 = nn.Conv1d(channels, squeezed, 1)
        self.fc2 = nn.Conv1d(squeezed, channels, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_se = x.mean(dim=[3,4])  # [B, C, T]
        x_se = self.act1(self.fc1(x_se))
        x_se = self.act2(self.fc2(x_se))
        x = x * x_se.unsqueeze(-1).unsqueeze(-1)
        return x

# BasicBneck Temporal
class BasicBneckTemporal(nn.Module):
    def __init__(self, in_ch, out_ch, expanded_ch, kernel_size=(3,3,3), stride=(1,1,1)):
        super().__init__()
        self.expand = ConvBlock2DTemporal(in_ch, expanded_ch, spatial_kernel=kernel_size[1:], temporal_kernel=kernel_size[0], stride=stride)
        self.deep = ConvBlock2DTemporal(expanded_ch, expanded_ch, spatial_kernel=kernel_size[1:], temporal_kernel=kernel_size[0], stride=(1,1,1))
        self.se = SqueezeExcitationTemporal(expanded_ch)
        self.project = ConvBlock2DTemporal(expanded_ch, out_ch, spatial_kernel=(1,1), temporal_kernel=1, stride=(1,1,1))
        self.shortcut = (in_ch == out_ch and stride==(1,1,1))

    def forward(self, x):
        out = self.expand(x)
        out = self.deep(out)
        out = self.se(out)
        out = self.project(out)
        if self.shortcut:
            out = out + x
        return out

# MoViNet Temporal 主干
class MoViNetTemporal(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super().__init__()
        self.conv1 = ConvBlock2DTemporal(cfg.conv1.input_channels, cfg.conv1.out_channels, spatial_kernel=cfg.conv1.kernel_size[1:], temporal_kernel=cfg.conv1.kernel_size[0], stride=cfg.conv1.stride)
        blocks = []
        for block_list in cfg.blocks:
            for b in block_list:
                blocks.append(BasicBneckTemporal(b.input_channels, b.out_channels, b.expanded_channels, kernel_size=b.kernel_size, stride=b.stride))
        self.blocks = nn.Sequential(*blocks)
        self.conv7 = ConvBlock2DTemporal(cfg.conv7.input_channels, cfg.conv7.out_channels, spatial_kernel=cfg.conv7.kernel_size[1:], temporal_kernel=cfg.conv7.kernel_size[0], stride=cfg.conv7.stride)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Linear(cfg.conv7.out_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


""" 模型的加载方式的测试 """
if __name__ == '__main__':
    from cfg import build_movinet_a0_cfg

    movinet_a0_cfg = build_movinet_a0_cfg()
    print("movinet_a0_cfg==", movinet_a0_cfg)
    model = MoViNetTemporal(movinet_a0_cfg, num_classes=2)
    # NOTE: 模型结构加载成功.
    print("model==", model)
    x = torch.randn(1, 3, 16, 172, 172)

    # 推理
    with torch.no_grad():
        y = model(x)

    print("模型输出形状:", y.shape)  # 模型输出形状: torch.Size([1, 600]) K600数据集的分类数。 TODO: TEST-ok-0718