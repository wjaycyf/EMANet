import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.DCNv2.dcn_v2 import DCN_sep
from basicsr.archs.arch_util import make_layer, ResidualBlockNoBN


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class EMB_DAEA(nn.Module):
    def __init__(self, nf):
        super(EMB_DAEA, self).__init__()
        self.event_process = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.img_process = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, img, event):
        img = self.img_process(img)
        event = self.event_process(event)
        return img * event


class EventSubHead_DAEA(nn.Module):
    def __init__(self, front_RBs, inf, nf):
        super(EventSubHead_DAEA, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.feature_extraction = make_layer(ResidualBlockNoBN, front_RBs, num_feat=nf)
        self.fea_E1_conv1 = nn.Conv2d(inf, nf, 3, padding=1)

    def forward(self, event):
        event_b, _, event_h, event_w = event.shape
        E1_fea = self.lrelu(self.fea_E1_conv1(event))
        E1_fea = self.feature_extraction(E1_fea)
        E1_fea = E1_fea.view(event_b, -1, event_h, event_w)
        return E1_fea.clone()


class DAEA_Align(nn.Module):
    """Difficulty-aware forward alignment module (forward branch only)."""

    def __init__(self, nf=64, groups=8, dilation=1):
        super(DAEA_Align, self).__init__()

        # forward branch (fea1 -> fea2)
        self.offset_conv1_forward = nn.Conv2d(2 * nf, nf, 3, 1, 1)
        self.mul_scale1_forward = nn.Conv2d(nf, nf, 3, 1, 1)
        self.mul_scale2_forward = nn.Conv2d(nf, nf, 5, 1, 2)
        self.mul_scale3_forward = nn.Conv2d(nf, nf, 7, 1, 3)
        self.offset_conv2_forward = nn.Conv2d(3 * nf, nf, 3, 1, padding=dilation, dilation=dilation)
        self.dcnpack_forward = DCN_sep(nf, nf, 3, stride=1, padding=1, deformable_groups=groups)

        self.emb_forward = EMB_DAEA(nf)

        self.diff_estimator_f = nn.Sequential(
            nn.Conv2d(nf + nf, nf, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, 1, 3, 1, 1), nn.Sigmoid()
        )

        self.mul_scale1_forward_light = nn.Conv2d(nf, nf, 3, 1, 1)
        self.offset_conv2_forward_light = nn.Conv2d(nf, nf, 3, 1, 1)

        self.gate_f = nn.Sequential(
            nn.Conv2d(1 + nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, 1),
            nn.Sigmoid()
        )

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, fea1, fea2, event1_t):
        offset_input_f = torch.cat([fea1, fea2], dim=1)
        offset1_f = self.lrelu(self.offset_conv1_forward(offset_input_f))

        ms1_f_heavy = self.lrelu(self.mul_scale1_forward(offset1_f))
        ms2_f_heavy = self.lrelu(self.mul_scale2_forward(offset1_f))
        ms3_f_heavy = self.lrelu(self.mul_scale3_forward(offset1_f))
        offset2_f_heavy = self.lrelu(self.offset_conv2_forward(
            torch.cat([ms1_f_heavy, ms2_f_heavy, ms3_f_heavy], dim=1)
        ))

        ms1_f_light = self.lrelu(self.mul_scale1_forward_light(offset1_f))
        offset2_f_light = self.lrelu(self.offset_conv2_forward_light(ms1_f_light))

        offset_seed_f = offset1_f + 0.5 * (offset2_f_heavy + offset2_f_light)
        Rf = self.diff_estimator_f(torch.cat([offset_seed_f, event1_t], dim=1))
        Rf = F.avg_pool2d(Rf, 3, 1, 1)

        offset2_f = (1 - Rf) * offset2_f_light + Rf * offset2_f_heavy
        offset_final_f = offset1_f + (1 + 0.5 * Rf) * offset2_f

        emb_delta_f = self.emb_forward(offset_final_f, event1_t)
        gate_map_f = self.gate_f(torch.cat([Rf, offset_final_f], dim=1))
        offset_final_f = offset_final_f + gate_map_f * emb_delta_f

        aligned_fwd = self.lrelu(self.dcnpack_forward(fea1, offset_final_f))

        return aligned_fwd, Rf


class DAEA(nn.Module):
    """Difficulty-Aware Event-driven Alignment encoder (forward only)."""

    def __init__(self, front_RBs, inf, nf):
        super(DAEA, self).__init__()
        self.daea_align = DAEA_Align(nf=nf)
        self.event_sub_head = EventSubHead_DAEA(front_RBs=front_RBs, inf=inf, nf=nf)

    def forward(self, event, fea1, fea2):
        num_inter = event.shape[1]
        image_feature = [fea1]

        for idx in range(num_inter):
            event_t = event[:, idx, ...]
            event_feat = self.event_sub_head(event_t)

            aligned_feat, _ = self.daea_align(fea1, fea2, event_feat)
            image_feature.append(aligned_feat)

        image_feature.append(fea2)
        return torch.stack(image_feature, dim=1)
