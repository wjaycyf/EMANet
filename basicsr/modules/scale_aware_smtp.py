import torch
import torch.nn as nn

from basicsr.archs.module_util import ConvLayer, SimpleRecurrentConv, ChannelAtention
from basicsr.modules.scale_aware_modules import (
    ScaleAwareChannelAttention,
    LightweightScaleAwareAttention,
)

class ScaleAwareSMTPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, num_block=3, fuse_two_direction=False,
                 use_scale_aware=True, scale_aware_mode='lightweight'):
        super(ScaleAwareSMTPLayer, self).__init__()

        self.relu_slope = relu_slope
        self.use_scale_aware = use_scale_aware
        self.scale_aware_mode = scale_aware_mode

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)

        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_scale_aware:
            if scale_aware_mode == 'full':
                self.scale_atten_fuse = ScaleAwareChannelAttention(
                    c=in_channels,
                    c_event=in_channels,
                    c_out=out_channels,
                    DW_Expand=1,
                    FFN_Expand=2
                )
            else:  # lightweight
                self.scale_atten_fuse = LightweightScaleAwareAttention(
                    c=in_channels,
                    c_event=in_channels,
                    c_out=out_channels,
                    reduction=4
                )
        else:
            self.atten_fuse = ChannelAtention(c=in_channels, c_out=out_channels, DW_Expand=1, FFN_Expand=2)

        self.recurrent_block = SimpleRecurrentConv(out_channels, out_channels, num_block=num_block)

        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(2 * out_channels, out_channels, 1, 1, 0, relu_slope, norm)

    def forward(self, x, y=None, prev_state=None, bi_direction_state=None, scale=None):
        if y is not None:
            if self.use_scale_aware:
                if scale is None:
                    scale = (4, 4)
                    print("Warning: scale not provided, using default (4, 4)")
                x = self.scale_atten_fuse(x, y, scale)
            else:
                x = self.atten_fuse(x, y)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)

        x, state = self.recurrent_block(x, prev_state)

        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)

        return x, state
