import torch
import torch.nn as nn
import torch.nn.functional as F


# ======== 2D Conv + BN + Act ========
class Conv2dBNActivation(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.bn = norm_layer(out_ch)
        self.act = activation_layer() if activation_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ======== Squeeze-Excitation ========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, squeezed, 1)
        self.fc2 = nn.Conv2d(squeezed, channels, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,H,W
        x_se = F.adaptive_avg_pool2d(x, 1)  # B,C,1,1
        x_se = self.act1(self.fc1(x_se))
        x_se = self.act2(self.fc2(x_se))
        return x * x_se

# ======== Bottleneck Block ========
class BasicBneck(nn.Module):
    def __init__(self, in_ch, out_ch, expanded_ch, stride=1):
        super().__init__()
        self.use_residual = in_ch == out_ch and stride == 1
        self.expand = Conv2dBNActivation(in_ch, expanded_ch, kernel_size=1)
        self.depthwise = Conv2dBNActivation(expanded_ch, expanded_ch, kernel_size=3, stride=stride, padding=1, activation_layer=nn.ReLU)
        self.se = SEBlock(expanded_ch)
        self.project = Conv2dBNActivation(expanded_ch, out_ch, kernel_size=1, activation_layer=None)

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out

# ======== 端侧友好 MoViNet-like 模型 ========
class MoViNet4D(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, frames=16):
        super().__init__()
        # 将时间帧堆叠到 channel 维
        self.in_ch = in_channels * frames
        self.conv1 = Conv2dBNActivation(self.in_ch, 32, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            BasicBneck(32, 32, 64),
            BasicBneck(32, 64, 128, stride=2),
            BasicBneck(64, 128, 256, stride=2),
        )
        self.conv_last = Conv2dBNActivation(128, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: B,C,T,H,W -> B, C*T, H, W
        B, C, T, H, W = x.shape
        x = x.view(B, C * T, H, W)
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# ======== RKNN 导出 forward 保持一致 ========
def export_rknn_forward(model, x):
    # 输入同样 B,C,T,H,W
    return model(x)




# ======== quick test ========
if __name__ == '__main__':
    model = MoViNet4D(num_classes=2)
    print(model)
    x = torch.randn(1, 3, 16, 224, 224)
    y = model(x)
    print("output:", y.shape)  # torch.Size([1,2])
