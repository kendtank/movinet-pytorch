import torch
import torch.nn as nn
import torch.nn.functional as F
from net.cfg import build_movinet_a0_cfg

# ======== helpers ========
def _fold_bt(x):
    B, C, T, H, W = x.shape
    return x.reshape(B*T, C, H, W), B, T

def _unfold_bt(x_bt, B, T):
    BT, C, H, W = x_bt.shape
    assert BT == B*T
    x = x_bt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
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

# ======== Temporal 1D Conv ========
class TemporalConv1D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels)
    def forward(self, x, rknn_mode=False):
        B, C, T, H, W = x.shape
        if rknn_mode:
            # RKNN friendly: mean over H,W
            x_pool = x.mean(dim=[3,4])  # B,C,T
            x_w = self.conv(x_pool)     # B,C,T
            x_w = x_w.unsqueeze(-1).unsqueeze(-1)
            return x * x_w
        else:
            x_bt, B, T = _fold_bt(x)
            x_pool = F.adaptive_avg_pool2d(x_bt,1)
            x_pool = x_pool.reshape(B, T, x_pool.shape[1]).permute(0,2,1)
            x_w = self.conv(x_pool)
            return x * x_w.unsqueeze(-1).unsqueeze(-1)

# ======== Squeeze-Excitation ========
class SqueezeExcitationTemporal(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(8, channels//reduction)
        self.fc1 = nn.Conv1d(channels, squeezed, 1)
        self.fc2 = nn.Conv1d(squeezed, channels,1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
    def forward(self,x,rknn_mode=False):
        B,C,T,H,W = x.shape
        if rknn_mode:
            x_se = x.mean(dim=[3,4])
            x_se = self.act1(self.fc1(x_se))
            x_se = self.act2(self.fc2(x_se))
            x_se = x_se.unsqueeze(-1).unsqueeze(-1)
            return x * x_se
        else:
            x_bt, B, T = _fold_bt(x)
            x_se = F.adaptive_avg_pool2d(x_bt,1)
            x_se = x_se.reshape(B,T,x_se.shape[1]).permute(0,2,1)
            x_se = self.act1(self.fc1(x_se))
            x_se = self.act2(self.fc2(x_se))
            return x * x_se.unsqueeze(-1).unsqueeze(-1)

# ======== ConvBlock2DTemporal ========
class ConvBlock2DTemporal(nn.Module):
    def __init__(self,in_ch,out_ch,spatial_kernel=(3,3), temporal_kernel=3, stride=(1,1,1), act_layer=nn.ReLU):
        super().__init__()
        pad_h, pad_w = spatial_kernel[0]//2, spatial_kernel[1]//2
        self.spatial_conv = Conv2dBNActivation(in_ch, out_ch, kernel_size=spatial_kernel,
                                               stride=stride[1:], padding=(pad_h,pad_w), activation_layer=act_layer)
        self.temporal_conv = TemporalConv1D(out_ch, kernel_size=temporal_kernel, padding=temporal_kernel//2)
    def forward(self,x,rknn_mode=False):
        B,C,T,H,W = x.shape
        x_bt = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        x_spatial = self.spatial_conv(x_bt)
        C_new,H_new,W_new = x_spatial.shape[1:]
        x_spatial = x_spatial.reshape(B,T,C_new,H_new,W_new).permute(0,2,1,3,4)
        x_out = self.temporal_conv(x_spatial,rknn_mode)
        return x_out

# ======== BasicBneckTemporal ========
class BasicBneckTemporal(nn.Module):
    def __init__(self, in_ch,out_ch,expanded_ch,kernel_size=(3,3,3),stride=(1,1,1)):
        super().__init__()
        self.expand = ConvBlock2DTemporal(in_ch, expanded_ch, spatial_kernel=kernel_size[1:], temporal_kernel=kernel_size[0], stride=stride)
        self.deep = ConvBlock2DTemporal(expanded_ch, expanded_ch, spatial_kernel=kernel_size[1:], temporal_kernel=kernel_size[0])
        self.se = SqueezeExcitationTemporal(expanded_ch)
        self.project = ConvBlock2DTemporal(expanded_ch, out_ch, spatial_kernel=(1,1), temporal_kernel=1)
        self.shortcut = (in_ch==out_ch and stride==(1,1,1))
    def forward(self,x,rknn_mode=False):
        out = self.expand(x,rknn_mode)
        out = self.deep(out,rknn_mode)
        out = self.se(out,rknn_mode)
        out = self.project(out,rknn_mode)
        if self.shortcut:
            out = out + x
        return out

# ======== MoViNetTemporal ========
class MoViNetTemporal(nn.Module):
    def __init__(self,cfg,num_classes=2):
        super().__init__()
        self.conv1 = ConvBlock2DTemporal(cfg.conv1.input_channels, cfg.conv1.out_channels,
                                         spatial_kernel=cfg.conv1.kernel_size[1:], temporal_kernel=cfg.conv1.kernel_size[0],
                                         stride=cfg.conv1.stride)
        blocks = []
        for block_list in cfg.blocks:
            for b in block_list:
                blocks.append(BasicBneckTemporal(b.input_channels,b.out_channels,b.expanded_channels,
                                                kernel_size=b.kernel_size,stride=b.stride))
        self.blocks = nn.Sequential(*blocks)
        self.conv7 = ConvBlock2DTemporal(cfg.conv7.input_channels,cfg.conv7.out_channels,
                                         spatial_kernel=cfg.conv7.kernel_size[1:], temporal_kernel=cfg.conv7.kernel_size[0],
                                         stride=cfg.conv7.stride)
        self.classifier = nn.Linear(cfg.conv7.out_channels,num_classes)
        self.export_rknn = False
    def forward(self,x):
        x = self.conv1(x,self.export_rknn)
        for b in self.blocks:
            x = b(x,self.export_rknn)
        x = self.conv7(x,self.export_rknn)
        if self.export_rknn:
            x = x.mean(dim=[2,3,4]) # B,C
        else:
            x = F.adaptive_avg_pool3d(x,(1,1,1)).flatten(1)
        x = self.classifier(x)
        return x
    def set_rknn_export(self,mode=True):
        self.export_rknn = mode

# ======== dummy cfg ========
class DummyBlock:
    def __init__(self,in_ch,out_ch,exp_ch,kernel=(3,3,3),stride=(1,1,1)):
        self.input_channels=in_ch
        self.out_channels=out_ch
        self.expanded_channels=exp_ch
        self.kernel_size=kernel
        self.stride=stride
class DummyConv:
    def __init__(self,in_ch,out_ch,kernel=(3,3,3),stride=(1,1,1)):
        self.input_channels=in_ch
        self.out_channels=out_ch
        self.kernel_size=kernel
        self.stride=stride


class DummyCfg:
    def __init__(self):
        self.conv1 = DummyConv(3,16,(3,3,3),(1,2,2))
        self.conv7 = DummyConv(16,32,(3,3,3),(1,1,1))
        self.blocks = [
            [DummyBlock(16,16,32),(DummyBlock(16,16,32))]
        ]

# ======== quick test ========
if __name__ == '__main__':
    cfg = build_movinet_a0_cfg()
    model = MoViNetTemporal(cfg,num_classes=2)
    print(model)
    x = torch.randn(1,3,16,224,224)
    # training forward
    y = model(x)
    print("train output:",y.shape)
    # export forward
    model.set_rknn_export(True)
    y2 = model(x)
    print("rknn export output:",y2.shape)
