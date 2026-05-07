import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaleAwareChannelAttention(nn.Module):
    def __init__(self, c, c_event=None, c_out=None, reduction=4, DW_Expand=1, FFN_Expand=2):
        super(ScaleAwareChannelAttention, self).__init__()

        if c_event is None:
            c_event = c
        if c_out is None:
            c_out = c

        self.c = c
        self.c_event = c_event
        self.c_out = c_out

        # Scale routing network
        self.scale_encoder = nn.Sequential(
            nn.Linear(2, c // 2),
            nn.ReLU(True),
            nn.Linear(c // 2, c),
            nn.Sigmoid()
        )

        # Depthwise convolution
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)

        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Event feature processing
        self.conv_event = nn.Conv2d(c_event, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # FFN
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c_out, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)

    def forward(self, x, y, scale):
        B, C, H, W = x.shape

        if isinstance(scale, (int, float)):
            scale_tensor = torch.tensor([scale, scale], dtype=x.dtype, device=x.device)
        elif isinstance(scale, np.ndarray) and scale.ndim == 0:
            scale_val = float(scale)
            scale_tensor = torch.tensor([scale_val, scale_val], dtype=x.dtype, device=x.device)
        else:
            scale_tensor = torch.tensor([scale[0], scale[1]], dtype=x.dtype, device=x.device)

        scale_tensor = scale_tensor.unsqueeze(0).repeat(B, 1)  # [B, 2]
        scale_weights = self.scale_encoder(scale_tensor)
        scale_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        x_scaled = x * (self.gamma * scale_weights + self.beta)
        inp = x_scaled
        x_norm = self.norm1(x_scaled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_dw = self.conv1(x_norm) 
        x_dw = self.conv2(x_dw) 
        x1, x2 = x_dw.chunk(2, dim=1) 

        y_feat = self.conv_event(y)    # C_event → dw_channel//2

        x_fused = x1 * y_feat * torch.sigmoid(x2)  # Event-Gated GLU

        x_out = self.conv3(x_fused)
        x_out = inp + x_out 

        ca = self.channel_attention(x_out)
        x_out = x_out * ca

        inp_ffn = x_out
        x_ffn = self.norm2(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_ffn = self.conv4(x_ffn)
        x1_ffn, x2_ffn = x_ffn.chunk(2, dim=1)
        x_ffn = F.gelu(x1_ffn) * x2_ffn 
        x_out = self.conv5(x_ffn) + inp_ffn

        return x_out


class LightweightScaleAwareAttention(nn.Module):
    def __init__(self, c, c_event=None, c_out=None, reduction=4):
        super(LightweightScaleAwareAttention, self).__init__()

        if c_event is None:
            c_event = c
        if c_out is None:
            c_out = c

        self.c = c
        self.c_out = c_out

        # Scale encoder
        self.scale_encoder = nn.Sequential(
            nn.Linear(2, c // reduction),
            nn.ReLU(True),
            nn.Linear(c // reduction, c),
            nn.Sigmoid()
        )

        # Fusion conv
        self.fusion_conv = nn.Conv2d(c + c_event, c_out, kernel_size=1, bias=True)

        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out // reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(c_out // reduction, c_out, 1, bias=True),
            nn.Sigmoid()
        )

        # Learnable scale sensitivity
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, scale):
        B = x.size(0)

        # Encode scale
        if isinstance(scale, (int, float)):
            scale_vec = torch.tensor([scale, scale], dtype=x.dtype, device=x.device)
        elif isinstance(scale, np.ndarray) and scale.ndim == 0:
            scale_val = float(scale)
            scale_vec = torch.tensor([scale_val, scale_val], dtype=x.dtype, device=x.device)
        else:
            scale_vec = torch.tensor([scale[0], scale[1]], dtype=x.dtype, device=x.device)

        scale_vec = scale_vec.unsqueeze(0).repeat(B, 1)
        scale_weights = self.scale_encoder(scale_vec).unsqueeze(-1).unsqueeze(-1)

        x_mod = x * (1 + self.alpha * scale_weights)

        # Fusion
        fused = torch.cat([x_mod, y], dim=1)
        out = self.fusion_conv(fused)

        # Channel attention
        out = out * self.ca(out)

        return out