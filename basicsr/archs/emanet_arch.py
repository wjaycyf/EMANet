import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import make_layer, ResidualBlockNoBN
from .smtp import SMTP
from .edtd import EDTD
from .daea import ConvBlock, DAEA
from basicsr.modules.event_residual_connection import EventResidualConnection, get_frame_timestamp
from basicsr.modules.scale_aware_smtp import ScaleAwareSMTPLayer


@ARCH_REGISTRY.register()
class EMANetArch(nn.Module):
    def __init__(
            self,
            event_channels,
            channels,
            n_feats,
            front_RBs,
            base_dim,
            head,
            r,
            r_t,
            use_event_residual=True,
            event_residual_sample_number=8,
            event_residual_offset=0.0125,
            event_residual_layers=2,
            use_scale_aware_smtp=True,
            scale_aware_mode='full',
            use_adaptive_temporal=False,
            adaptive_sampler_type='density',
            r_t_min=1,
            r_t_max=3,
    ):
        super(EMANetArch, self).__init__()

        self.n_feats = n_feats
        self.use_event_residual = use_event_residual
        self.use_scale_aware_smtp = use_scale_aware_smtp
        self.scale_aware_mode = scale_aware_mode

        self.image_head = nn.Conv2d(channels, n_feats, 5, padding=2)
        self.event_head = nn.Conv2d(event_channels, n_feats, 5, padding=2)

        self.feature_extraction = make_layer(
            ResidualBlockNoBN, front_RBs, num_feat=n_feats)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.daea = DAEA(front_RBs=front_RBs, inf=event_channels, nf=n_feats)

        if use_scale_aware_smtp:
            self.backward_smtp = ScaleAwareSMTPLayer(
                n_feats, n_feats,
                fuse_two_direction=False,
                use_scale_aware=True,
                scale_aware_mode=scale_aware_mode
            )
            self.forward_smtp = ScaleAwareSMTPLayer(
                n_feats, n_feats,
                fuse_two_direction=True,
                use_scale_aware=True,
                scale_aware_mode=scale_aware_mode
            )
        else:
            self.backward_smtp = SMTP(
                n_feats, n_feats,
                fuse_two_direction=False,
                use_atten_fuse=True
            )
            self.forward_smtp = SMTP(
                n_feats, n_feats,
                fuse_two_direction=True,
                use_atten_fuse=True
            )

        if use_event_residual:
            self.event_residual_connection = EventResidualConnection(
                sample_number=event_residual_sample_number,
                event_channels=n_feats,
                out_channels=n_feats,
                offset=event_residual_offset,
                layers=event_residual_layers
            )
        else:
            self.event_residual_connection = None

        self.decoder = EDTD(
            in_dim=n_feats,
            base_dim=base_dim,
            head=head,
            r=r,
            r_t=r_t,
            use_adaptive_temporal=use_adaptive_temporal,
            adaptive_sampler_type=adaptive_sampler_type,
            r_t_min=r_t_min,
            r_t_max=r_t_max,
            event_channels=n_feats
        )

    def forward(self, image, event, scale, times):
        if len(image.shape) == 4:
            image.unsqueeze(0)
        if len(event.shape) == 4:
            event.unsqueeze(0)
        image_b, image_t, image_c, image_h, image_w = image.shape
        event_b, event_t, event_c, event_h, event_w = event.shape

        image_head_feature = self.image_head(
            image.view(-1, image_c, image_h, image_w))
        event_head_feature = self.event_head(
            event.view(-1, event_c, event_h, event_w))
        event_head_feature = event_head_feature.view(event_b, event_t, -1, event_h, event_w)

        L1_fea = self.lrelu(image_head_feature)
        L1_fea = self.feature_extraction(L1_fea)
        L1_fea = L1_fea.view(image_b, image_t, -1, image_h, image_w)
        fea1 = L1_fea[:, 0, :, :, :].clone()
        fea2 = L1_fea[:, 1, :, :, :].clone()
        del L1_fea
        torch.cuda.empty_cache()

        image_feature = self.daea(event, fea1, fea2)
        T = image_feature.shape[1]

        # backward propagation
        backward_states = []
        for frame_idx in range(T - 1, -1, -1):
            image_cur = image_feature[:, frame_idx, :, :, :]

            if frame_idx == T - 1 or frame_idx == 0:
                event_cur = None
            else:
                event_cur = event_head_feature[:, frame_idx - 1, :, :, :]

            if self.use_event_residual and event_cur is not None:
                timestamp = get_frame_timestamp(frame_idx, T)
                event_residual_feature = self.event_residual_connection(
                    event_head_feature, timestamp, image_h, image_w
                )
                image_cur = image_cur + event_residual_feature

            if frame_idx == T - 1:
                if self.use_scale_aware_smtp:
                    _, state = self.backward_smtp(
                        x=image_cur, y=event_cur, prev_state=None,
                        bi_direction_state=None, scale=scale
                    )
                else:
                    _, state = self.backward_smtp(
                        x=image_cur, y=event_cur, prev_state=None
                    )
            else:
                if self.use_scale_aware_smtp:
                    _, state = self.backward_smtp(
                        x=image_cur, y=event_cur, prev_state=state,
                        bi_direction_state=None, scale=scale
                    )
                else:
                    _, state = self.backward_smtp(
                        x=image_cur, y=event_cur, prev_state=state
                    )
            backward_states.append(state)

        # forward propagation
        pro_feature = []
        for frame_idx in range(0, T):
            image_cur = image_feature[:, frame_idx, :, :, :]

            if frame_idx == 0 or frame_idx == T - 1:
                event_cur = None
            else:
                event_cur = event_head_feature[:, frame_idx - 1, :, :, :]

            if self.use_event_residual and event_cur is not None:
                timestamp = get_frame_timestamp(frame_idx, T)
                event_residual_feature = self.event_residual_connection(
                    event_head_feature, timestamp, image_h, image_w
                )
                image_cur = image_cur + event_residual_feature

            if frame_idx == 0:
                if self.use_scale_aware_smtp:
                    x, state = self.forward_smtp(
                        x=image_cur, y=event_cur, prev_state=None,
                        bi_direction_state=backward_states[T - 1 - frame_idx],
                        scale=scale
                    )
                else:
                    x, state = self.forward_smtp(
                        x=image_cur, y=event_cur, prev_state=None,
                        bi_direction_state=backward_states[T - 1 - frame_idx]
                    )
            else:
                if self.use_scale_aware_smtp:
                    x, state = self.forward_smtp(
                        x=image_cur, y=event_cur, prev_state=state,
                        bi_direction_state=backward_states[T - 1 - frame_idx],
                        scale=scale
                    )
                else:
                    x, state = self.forward_smtp(
                        x=image_cur, y=event_cur, prev_state=state,
                        bi_direction_state=backward_states[T - 1 - frame_idx]
                    )
            pro_feature.append(x)

        pro_feature = torch.stack(pro_feature, dim=1)

        out = pro_feature + image_feature
        out = out.permute(0, 2, 1, 3, 4)
        del pro_feature, image_feature
        torch.cuda.empty_cache()

        out = self.decoder(out, scale, times, event_features=event_head_feature)
        out = torch.stack(out, dim=1)
        return out
