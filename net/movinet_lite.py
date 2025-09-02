import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======== helpers: reshape without einops (RKNN-friendly) ========

def _fold_bt(x):
    """B,C,T,H,W -> (B*T),C,H,W  +  (B,T)"""
    B, C, T, H, W = x.shape
    return x.reshape(B * T, C, H, W), B, T


def _unfold_bt(x_bt, B, T):
    """(B*T),C,H,W -> B,C,T,H,W"""
    x = x_bt.reshape(B, T, x_bt.shape[1], x_bt.shape[2], x_bt.shape[3])
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x


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



# ======== Temporal 1D Conv (DW-Conv1d over T) ========
class TemporalConv1D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels)

    def forward(self, x):
        # x: B,C,T,H,W
        x_bt, B, T = _fold_bt(x)                 # (B*T),C,H,W
        x_pool = F.adaptive_avg_pool2d(x_bt, 1)  # (B*T),C,1,1   ——只在空间聚合
        x_pool = x_pool.reshape(B, T, x_pool.shape[1]).permute(0, 2, 1).contiguous()  # B,C,T
        x_w = self.conv(x_pool)                  # B,C,T
        return x * x_w.unsqueeze(-1).unsqueeze(-1)  # B,C,T,H,W（门控）



# ======== Squeeze-Excitation（空间 SE，不跨 T） ========
class SqueezeExcitationTemporal(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(8, channels // reduction)
        self.fc1 = nn.Conv1d(channels, squeezed, 1)
        self.fc2 = nn.Conv1d(squeezed, channels, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        # x: B,C,T,H,W
        x_bt, B, T = _fold_bt(x)                 # (B*T),C,H,W
        x_se = F.adaptive_avg_pool2d(x_bt, 1)    # (B*T),C,1,1
        x_se = x_se.reshape(B, T, x_se.shape[1]).permute(0, 2, 1).contiguous()  # B,C,T
        x_se = self.act1(self.fc1(x_se))
        x_se = self.act2(self.fc2(x_se))
        return x * x_se.unsqueeze(-1).unsqueeze(-1)  # B,C,T,H,W



# ======== 2D + Temporal Block（保持你现有 block 逻辑不动） ========
class ConvBlock2DTemporal(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_kernel=(3, 3), temporal_kernel=3,
                 stride=(1, 1, 1), activation_layer=nn.ReLU):
        super().__init__()
        pad_h, pad_w = spatial_kernel[0] // 2, spatial_kernel[1] // 2
        self.spatial_conv = Conv2dBNActivation(
            in_ch, out_ch,
            kernel_size=spatial_kernel,
            stride=stride[1:],
            padding=(pad_h, pad_w),
            activation_layer=activation_layer,
        )
        self.temporal_conv = TemporalConv1D(out_ch, kernel_size=temporal_kernel, padding=temporal_kernel // 2)

    def forward(self, x):
        # x: B,C,T,H,W
        x_bt, B, T = _fold_bt(x)            # -> (B*T),C,H,W
        x_bt = self.spatial_conv(x_bt)      # 2D Conv (H,W)
        x = _unfold_bt(x_bt, B, T)          # -> B,C,T,H,W
        x = self.temporal_conv(x)           # DW-Conv1d over T
        return x



class BasicBneckTemporal(nn.Module):
    def __init__(self, in_ch, out_ch, expanded_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        self.expand = ConvBlock2DTemporal(in_ch, expanded_ch,
                                          spatial_kernel=kernel_size[1:], temporal_kernel=kernel_size[0],
                                          stride=stride)
        self.deep = ConvBlock2DTemporal(expanded_ch, expanded_ch,
                                        spatial_kernel=kernel_size[1:], temporal_kernel=kernel_size[0],
                                        stride=(1, 1, 1))
        self.se = SqueezeExcitationTemporal(expanded_ch)
        self.project = ConvBlock2DTemporal(expanded_ch, out_ch,
                                           spatial_kernel=(1, 1), temporal_kernel=1,
                                           stride=(1, 1, 1))
        self.shortcut = (in_ch == out_ch and stride == (1, 1, 1))

    def forward(self, x):
        out = self.expand(x)
        out = self.deep(out)
        out = self.se(out)
        out = self.project(out)
        if self.shortcut:
            out = out + x
        return out


# ======== 一个简化的 MoViNet-like 2D+1D 主干（你可以用你自己的 cfg 替换这里的通道数） ========
class MoViNet2D1DBackbone(nn.Module):
    """
    输出张量形状保持 B,C,T,H,W，方便 head 做时间聚合。
    下面通道配置只是示例，你可按自己 cfg 改；最关键是保持 5D 张量流，避免 3D 池化。
    """
    def __init__(self, in_ch=3, stem_out=32):
        super().__init__()
        # conv1
        self.conv1 = ConvBlock2DTemporal(in_ch, stem_out, spatial_kernel=(3, 3), temporal_kernel=3, stride=(1, 2, 2))
        # 三个 stage（示例）
        self.blocks = nn.Sequential(
            BasicBneckTemporal(stem_out,   64,  96, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            BasicBneckTemporal(64,        128, 192, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
            BasicBneckTemporal(128,       256, 384, kernel_size=(3, 3, 3), stride=(1, 2, 2)),
        )
        # conv7（最后一层升维）
        self.conv7 = ConvBlock2DTemporal(256, 256, spatial_kernel=(1, 1), temporal_kernel=3, stride=(1, 1, 1))
        self.out_channels = 256  # 给 head 用

    def forward(self, x):  # x: B,C,T,H,W
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        return x  # B,C,T,H,W


# ======== RKNN 友好的 Temporal Head：空间 GAP + DW-Conv1d(kernel=T) ========
class TemporalHead(nn.Module):
    def __init__(self, in_channels, num_classes, export_T, trainable_temporal=True):
        super().__init__()
        self.export_T = int(export_T)

        # 深度可分离 1D 卷积：kernel = T，等价于全局时间池化（可学习）
        self.temporal_pool = nn.Conv1d(in_channels, in_channels,
                                       kernel_size=self.export_T,
                                       groups=in_channels,
                                       bias=False)
        if not trainable_temporal:
            with torch.no_grad():
                w = torch.ones_like(self.temporal_pool.weight) / float(self.export_T)
                self.temporal_pool.weight.copy_(w)
            for p in self.temporal_pool.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):  # x: B,C,T,H,W
        B, C, T, H, W = x.shape
        # 1) 仅做空间 GAP（用 2D 池化，避免 3D 池化）
        x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # (B*T),C,H,W
        x_bt = F.adaptive_avg_pool2d(x_bt, 1)                     # (B*T),C,1,1
        x_t = x_bt.reshape(B, T, C).permute(0, 2, 1).contiguous() # B,C,T

        # 2) 时间聚合：DW-Conv1d，kernel=T -> 输出 B,C,1
        x_t = self.temporal_pool(x_t)                             # B,C,1

        # 3) 分类
        x_vec = x_t.squeeze(-1)                                   # B,C
        out = self.classifier(x_vec)                              # B,num_classes
        return out


# ======== 整体模型：主干（保持不变）+ 新 head ========
class MoViNet2D1D(nn.Module):
    def __init__(self, num_classes=2, export_T=16, in_ch=3):
        super().__init__()
        self.backbone = MoViNet2D1DBackbone(in_ch=in_ch, stem_out=32)
        self.head = TemporalHead(self.backbone.out_channels, num_classes, export_T=export_T, trainable_temporal=True)

    def forward(self, x):  # x: B,C,T,H,W
        feats = self.backbone(x)   # B,C,T,H,W
        logits = self.head(feats)  # B,num_classes
        return logits


# ======== 导出 ONNX：固定输入尺寸，避免动态维 ========
def export_onnx(model, onnx_path, T=16, H=224, W=224, opset=11):
    model.eval()
    dummy = torch.randn(1, 3, T, H, W)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['logits'],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # 固定形状，便于 RKNN
    )
    print(f"[OK] Exported ONNX to: {onnx_path}")


# ======== quick self-test / CLI ========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', choices=['none', 'onnx'], default='none')
    parser.add_argument('--onnx-path', default='movinet_2d1d_rknn.onnx')
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--H', type=int, default=224)
    parser.add_argument('--W', type=int, default=224)
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()

    model = MoViNet2D1D(num_classes=args.num_classes, export_T=args.T).eval()
    x = torch.randn(1, 3, args.T, args.H, args.W)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)  # (1, num_classes)

    if args.export == 'onnx':
        export_onnx(model, args.onnx_path, T=args.T, H=args.H, W=args.W, opset=11)


if __name__ == '__main__':
    main()

    # model = MoViNet2D1D()
    # print(model)


# export： python movinet_lite.py --export onnx --T 16 --H 224 --W 224 --num-classes 2
# 生成 movinet_2d1d_rknn.onnx