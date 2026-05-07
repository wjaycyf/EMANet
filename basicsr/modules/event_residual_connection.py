import torch
import torch.nn as nn
import torch.nn.functional as F


def get_WHT_coords(t: float, h: int, w: int):
    assert t >= -1 and t <= 1, f"Time t should be in [-1, 1], but got {t}."
    grid_map = torch.zeros(1, h, w, 3) + t

    h_coords = torch.linspace(-1, 1, h)
    w_coords = torch.linspace(-1, 1, w)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords], indexing='ij')
    grid_map[:, :, :, 1:] = torch.stack((mesh_w, mesh_h), 2)
    return grid_map.float()


class EventResidualConnection(nn.Module):
    def __init__(self, sample_number, event_channels, out_channels, offset, layers=2):
        super(EventResidualConnection, self).__init__()
        self.sample_number = sample_number
        self.event_channels = event_channels
        in_channels = (sample_number * 2 + 1) * event_channels
        self.offset = offset

        if layers == 2:
            self.sampler_module = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, 1, 1, 0),
            )
        elif layers == 3:
            self.sampler_module = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(64, out_channels, 1, 1, 0),
            )
        else:
            raise NotImplementedError

    def forward(self, event_features, key_timestamp, feature_h, feature_w):
        if len(event_features.shape) == 5:
            # event_features: (b, t, c, h, w)
            bz, event_t, event_c, h, w = event_features.shape
            # Reshape to (b, c, h, w, t) for grid_sample
            event_bchwt = event_features.permute(0, 2, 3, 4, 1)
        elif len(event_features.shape) == 4:
            # event_features: (b, t*c, h, w) - need to reshape
            bz, tc, h, w = event_features.shape
            event_t = tc // self.event_channels
            event_features = event_features.reshape(bz, event_t, self.event_channels, h, w)
            event_bchwt = event_features.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f"Unexpected event_features shape: {event_features.shape}")

        dt = self.offset / self.sample_number
        sampled_events = []

        for i in range(2 * self.sample_number + 1):
            t = key_timestamp + (i - self.sample_number) * dt
            t = min(1, max(t, -1))

            coord = get_WHT_coords(t, feature_h, feature_w)
            coord = coord.unsqueeze(0).repeat(bz, 1, 1, 1, 1).cuda()


            sampled_event = F.grid_sample(
                input=event_bchwt,
                grid=coord,
                align_corners=True,
                mode="bilinear",
            )
            sampled_events.append(sampled_event)


        sampled_events = torch.cat(sampled_events, dim=1)
        sampled_events = sampled_events.squeeze(2)

        sampled_event_feature = self.sampler_module(sampled_events)

        return sampled_event_feature


def get_frame_timestamp(frame_idx, total_frames):

    if total_frames == 1:
        return 0.0
    return -1.0 + 2.0 * frame_idx / (total_frames - 1)
