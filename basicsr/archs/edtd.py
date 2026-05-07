import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .function import make_coord, get_coords, get_cells, get_idxlist
from .module_util import PositionEncoder3d
from basicsr.modules.motion_adaptive_temporal_sampling import build_temporal_sampler


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act='gelu'):
        super().__init__()

        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU(True)
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f'activation {act} is not supported'

        layers = []
        lastv = in_dim

        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden

        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class EDTD(nn.Module):
    def __init__(
        self,
        in_dim=64,
        base_dim=16,
        head=8,
        r=3,
        r_t=2,
        # Motion-adaptive temporal sampling parameters
        use_adaptive_temporal=False,
        adaptive_sampler_type='density',  # 'learnable' or 'density'
        r_t_min=1,
        r_t_max=3,
        event_channels=64
    ):
        super().__init__()
        self.in_dim = in_dim
        self.dim = base_dim
        self.head = head
        self.r = r
        self.r_t = r_t  # Default r_t for fixed mode

        # Motion-adaptive temporal sampling
        self.use_adaptive_temporal = use_adaptive_temporal
        self.r_t_min = r_t_min
        self.r_t_max = r_t_max

        if use_adaptive_temporal:
            # Build sampler with appropriate parameters
            if adaptive_sampler_type == 'learnable':
                self.temporal_sampler = build_temporal_sampler(
                    sampler_type=adaptive_sampler_type,
                    event_channels=event_channels,
                    r_t_min=r_t_min,
                    r_t_max=r_t_max,
                )
            elif adaptive_sampler_type == 'density':
                self.temporal_sampler = build_temporal_sampler(
                    sampler_type=adaptive_sampler_type,
                    r_t_min=r_t_min,
                    r_t_max=r_t_max,
                )
            else:
                raise ValueError(f"Unknown adaptive_sampler_type: {adaptive_sampler_type}")
        else:
            self.temporal_sampler = None

        self.conv_ch = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=3, padding=1)

        self.conv_vs = nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1)
        
        self.conv_qs = nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1)
        
        self.conv_ks = nn.Conv3d(self.dim, self.dim, kernel_size=3, padding=1)

        self.pb_encoder = PositionEncoder3d(posenc_scale=10, enc_dims=64, gamma=1)

        self.r_area = (2 * self.r + 1)**2

        # Create separate MLPs for each possible r_t value
        if use_adaptive_temporal:
            # Multi-MLP branches for different temporal window sizes
            self.mlps = nn.ModuleDict()
            for rt in range(r_t_min, r_t_max + 1):
                r_volume_rt = self.r_area * (2 * rt + 1)
                imnet_in_dim = self.dim * r_volume_rt + self.dim + 2
                self.mlps[str(rt)] = MLP(
                    in_dim=imnet_in_dim,
                    out_dim=3,
                    hidden_list=[256, 256, 256, 256],
                    act='gelu'
                )
        else:
            # Original single MLP for fixed r_t
            self.r_volume = self.r_area * (2 * self.r_t + 1)
            imnet_in_dim = self.dim * self.r_volume + self.dim + 2
            self.mlp = MLP(in_dim=imnet_in_dim, out_dim=3, hidden_list=[256, 256, 256, 256], act='gelu')

    def forward(self, feat_img, scale, times, event_features=None):  # feat_img.shape:[8, 64, 9, 32, 32] scale=4.0 times=[0.0, 0.125, 0.25, ..., 1.0]
        feat_img_shape = feat_img.shape
        coords = get_coords(feat_img_shape, scale)  # target HR 坐标 (像素中心点) [8, 128, 128, 2]

        sr_image_list = []

        # Feature generation
        feat = self.conv_ch(feat_img)  # b, c, t, h, w
        del feat_img
        torch.cuda.empty_cache()
        bs, fc, ft, fh, fw = feat.shape  # LR 8, 64, 9, 32, 32
        times = times * (ft - 1)  # [0,1] → [0, ft-1] = [0, 1, 2,.., 8]

        # Query RGB
        coord_lr = make_coord((fh, fw), flatten=False).cuda()
        coord_lr = coord_lr.permute(2, 0, 1).unsqueeze(
            0).repeat(bs, 1, 1, 1)  # b, 2, h, w [8, 2, 32, 32]

        hr_coord = coords.clone()
        hr_coord = hr_coord.reshape(bs, -1, 2)  # (bs, h_hr * w_hr, 2) [8, 128*128, 2]
        hr_coord = hr_coord.unsqueeze(2)  # (b, q, 1, 2)  [8, 128*128, 1, 2]
        q_sample = hr_coord.shape[1] # 采样点数量 q = h_hr * w_hr 

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
            coord_lr, hr_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)   # HR 坐标点对应的 LR 坐标 [8, 128*128, 1, 2] (对于每一个 HR 像素位置，问：在 LR 坐标图里，这个位置对应的坐标是多少)

        del coord_lr
        torch.cuda.empty_cache()

        # field radius (global: [-1, 1])
        rh = 2 / fh   # 每个像素的坐标间距
        rw = 2 / fw
        r = self.r  # 1
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh  # [-1, 0, 1] 即采样一个 3 × 3 的邻域区域
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw
        # 1, 1, r_area, 2
        delta = torch.stack(torch.meshgrid(
            dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)  # [1, 1, 9, 2]  生成一个(2r+1)×(2r+1) = r_area 二维偏移坐标网格 

        # b, q, 1, 2 -> b, q, r_area, 2
        sample_coord_k = sample_coord_k + delta  #  局部空间采样域
        del delta
        torch.cuda.empty_cache()


        feat_q = self.conv_qs(feat)  # b, c, t, h, w [8, 64, 9, 32, 32]
        feat_k = self.conv_ks(feat)
        feat_v = self.conv_vs(feat)

        # b, 2 -> b, q, 2
        rel_cell = get_cells(feat_img_shape, scale).cuda()  # HR 像素单元大小 cell
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= fh
        rel_cell[..., 1] *= fw

        for i in range(len(times)):
            center_time_float = times[i].item() if torch.is_tensor(times[i]) else times[i]
            center_time = round(center_time_float)  # 最近的中心参考帧

            # Motion-adaptive temporal sampling
            # 首尾帧使用固定 r_t，中间帧使用自适应采样
            is_middle_frame = 0 < center_time < feat_img_shape[2] - 1
            if self.use_adaptive_temporal and event_features is not None and is_middle_frame:
                event_idx = center_time - 1  # 当前帧对应的事件对索引

                # 提取当前帧对应的事件对
                current_event = event_features[:, event_idx:event_idx+1, :, :, :]  # [B, 1, C, H, W]

                # 使用当前事件对估计运动强度和动态 r_t
                sample_idx, r_t_dynamic, motion_info = self.temporal_sampler(
                    current_event,  # 传入单个事件对
                    center_time,
                    feat_img_shape[2]
                )
                rel_time = (center_time_float - center_time) * 2 / (2 * r_t_dynamic + 1)
                # Optional: print motion info for debugging
                # print(f"Frame {i}: motion_score={motion_info['motion_score']:.3f}, r_t={r_t_dynamic}, window_size={len(sample_idx)}")
            else:
                # Fixed temporal sampling (original behavior)
                r_t_dynamic = self.r_t
                rel_time = (center_time_float - center_time) * 2 / (2 * self.r_t + 1)   # 目标帧相对于中心帧的归一化偏移
                sample_idx = get_idxlist(center_time, self.r_t, 0, feat_img_shape[2]-1)  # 局部时间域采样 center_time=0 sample_idx=[0,0,1]
            hr_coord3d = torch.cat((torch.ones_like(
                hr_coord[..., :1])*rel_time, hr_coord), dim=-1).float().unsqueeze(3)  # [8, q, 1, 1, 3]
            # Q - b, c, t, h, w -> b, c, q, 1, 1 -> b, q, 1, 1, c -> b, q, 1, h, c/h -> b, q, h, 1, c/h
            sample_feat_q = F.grid_sample(
                feat_q[:, :, sample_idx, ...], hr_coord3d.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 4, 1)   # b, q, 1, 1, c  Query 只有一个：目标 HR 像素在当前时间窗的 Query 向量
            sample_feat_q = sample_feat_q.reshape(
                bs, q_sample, 1, self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)   # b, q, h, 1, c/h

            # b, q, r_area, 2
            rel_coord_0 = hr_coord - sample_coord_k  # 空间偏移 [-1, 1]
           

            feat_in = []
            # Unfold along the temporal dimension (use dynamic r_t)
            for dt in range(2 * r_t_dynamic + 1):  # dt = 0, 1, 2, ..., 2*r_t_dynamic
                time_coord = rel_time - (dt - r_t_dynamic) * 2 / (2 * r_t_dynamic + 1)  # 时间偏移
                rel_coord = torch.cat((torch.ones_like(
                rel_coord_0[..., :1])*time_coord, rel_coord_0), dim=-1).float()   # b, q, r_area, 3
                rel_coord[..., 0] *= 2 * r_t_dynamic + 1
                rel_coord[..., 1] *= fh
                rel_coord[..., 2] *= fw

                # b, q, r_area, h
                _, pb = self.pb_encoder(rel_coord)
                del rel_coord
                torch.cuda.empty_cache()
                # K - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c -> b, q, r_area, h, c/h -> b, q, h, c/h, r_area
                sample_feat_k = F.grid_sample(
                    feat_k[:, :, sample_idx, ...][:, :, dt, ...], sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)       # b, q, r_area, c  K ：HR 像素在 LR 上的 3×3 空间邻域内的特征
                sample_feat_k = sample_feat_k.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                ).permute(0, 1, 3, 4, 2)    # b, q, h, c/h, r_area

                # b, q, h, 1, r_area -> b, q, r_area, h
                attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
                    bs, q_sample, self.head, self.r_area
                ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)
                del sample_feat_k
                torch.cuda.empty_cache()
                attn = F.softmax(torch.add(attn, pb), dim=-2)
                attn = attn.reshape(
                    bs, q_sample, self.r_area, self.head, 1)

                # V - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c
                sample_feat_v = F.grid_sample(
                    feat_v[:, :, sample_idx, ...][:, :, dt, ...], sample_coord_k.flip(-1), mode='nearest', align_corners=False
                ).permute(0, 2, 3, 1)    # b, q, r_area, c

                sample_feat_v = sample_feat_v.reshape(
                    bs, q_sample, self.r_area, self.head, self.dim // self.head
                )       # b, q, r_area, h, c/h
                attn = torch.mul(sample_feat_v, attn).reshape(
                    bs, q_sample, -1)
                del sample_feat_v
                torch.cuda.empty_cache()
                feat_in.append(attn)
                del attn
                torch.cuda.empty_cache()
            feat_in = torch.cat(feat_in, dim=-1)
            feat_back = F.grid_sample(
                feat_q[:, :, sample_idx, ...], hr_coord3d.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 1, 3, 4).reshape(bs, q_sample, fc)
            feat_in = torch.cat([feat_in, feat_back, rel_cell], dim=-1)
            del feat_back, hr_coord3d
            torch.cuda.empty_cache()

            # Select appropriate MLP based on dynamic r_t
            if self.use_adaptive_temporal:
                # Use the MLP corresponding to the current r_t_dynamic
                mlp_to_use = self.mlps[str(r_t_dynamic)]
            else:
                # Use the single fixed MLP
                mlp_to_use = self.mlp

            pred = mlp_to_use(feat_in).permute(0, 2, 1).reshape(
                bs, 3, coords.shape[1], coords.shape[2])  # b, c, h, w
            del feat_in
            torch.cuda.empty_cache()

            sr_image_list.append(pred)
            del pred
            torch.cuda.empty_cache()

        return sr_image_list
