#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Motion-Adaptive Temporal Sampling Module

This module implements dynamic temporal window adjustment based on event density,
which naturally reflects motion intensity in event-based vision.

Key Idea:
- High motion (dense events) → larger temporal window (r_t↑) for more temporal context
- Low motion (sparse events) → smaller temporal window (r_t↓) for efficiency
"""

import torch
import torch.nn as nn


class MotionEstimator(nn.Module):
    def __init__(self, event_channels=64, mode='global'):
        super(MotionEstimator, self).__init__()
        self.mode = mode

        # Global pooling-based estimator (lightest)
        self.estimator = nn.Sequential(
            nn.Conv3d(event_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # Global pooling to [B, 16, 1, 1, 1]
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output [0, 1]
        )

    def forward(self, event_features):
        # Convert to [B, C, T, H, W] for 3D convolution
        x = event_features.permute(0, 2, 1, 3, 4)

        motion = self.estimator(x)
        return motion.squeeze()  # Scalar or [B]


class MotionAdaptiveTemporalSampler(nn.Module):
    def __init__(
        self,
        event_channels=64,
        r_t_min=1,
        r_t_max=3,
        motion_mode='global',
        use_learnable_threshold=False
    ):
        super(MotionAdaptiveTemporalSampler, self).__init__()
        self.r_t_min = r_t_min
        self.r_t_max = r_t_max
        self.motion_mode = motion_mode

        # Motion estimator
        self.motion_estimator = MotionEstimator(
            event_channels=event_channels,
            mode=motion_mode
        )

        # Learnable thresholds for multi-level r_t selection
        if use_learnable_threshold:
            num_levels = r_t_max - r_t_min
            self.thresholds = nn.Parameter(
                torch.linspace(0.3, 0.7, num_levels).float()
            )
        else:
            self.register_buffer(
                'thresholds',
                torch.linspace(0.3, 0.7, r_t_max - r_t_min).float()
            )

    def get_dynamic_r_t(self, motion_score):
        r_t = self.r_t_min + motion_score * (self.r_t_max - self.r_t_min)
        r_t = int(torch.round(r_t).item())
        r_t = max(self.r_t_min, min(self.r_t_max, r_t))
        return r_t

    def forward(self, event_features, target_time_idx, total_frames):
        # Estimate motion intensity
        motion_score = self.motion_estimator(event_features)

        # Handle batch dimension
        if len(motion_score.shape) > 0 and motion_score.shape[0] > 1:
            # Use mean over batch for global r_t decision
            motion_score_global = motion_score.mean()
        else:
            motion_score_global = motion_score

        # Get dynamic r_t
        r_t = self.get_dynamic_r_t(motion_score_global)

        # Compute temporal sampling indices
        center_time = round(target_time_idx)
        sample_idx = self._get_idxlist(center_time, r_t, 0, total_frames - 1)

        # Prepare motion info for visualization and debugging
        motion_info = {
            'motion_score': motion_score_global.item() if torch.is_tensor(motion_score_global) else motion_score_global,
            'r_t': r_t,
            'window_size': len(sample_idx),
            'sample_indices': sample_idx
        }

        return sample_idx, r_t, motion_info

    def _get_idxlist(self, idx, r_t, min_idx, max_idx):
        idxlist = []
        for i in range(-r_t, r_t + 1):
            frame_idx = idx + i
            frame_idx = max(min_idx, min(max_idx, frame_idx))
            idxlist.append(frame_idx)
        return idxlist


class DensityBasedTemporalSampler(nn.Module):
    def __init__(self, r_t_min=1, r_t_max=3, density_percentile=0.5):
        super(DensityBasedTemporalSampler, self).__init__()
        self.r_t_min = r_t_min
        self.r_t_max = r_t_max
        self.density_percentile = density_percentile

    def forward(self, event_features, target_time_idx, total_frames):
        density = torch.abs(event_features).mean(dim=2)  # [B, T, H, W]  计算空间密度图

        # Global density score
        density_score = density.mean()

        # Normalize to [0, 1] using adaptive percentile  相对运动强度
        density_max = torch.quantile(density.flatten(), self.density_percentile + 0.3)
        density_min = torch.quantile(density.flatten(), self.density_percentile - 0.3)
        density_normalized = (density_score - density_min) / (density_max - density_min + 1e-6)
        density_normalized = torch.clamp(density_normalized, 0, 1)

        # Map to r_t
        r_t = self.r_t_min + density_normalized * (self.r_t_max - self.r_t_min)
        r_t = int(torch.round(r_t).item())
        r_t = max(self.r_t_min, min(self.r_t_max, r_t))

        # Get sample indices
        center_time = round(target_time_idx)
        sample_idx = self._get_idxlist(center_time, r_t, 0, total_frames - 1)

        motion_info = {
            'density_score': density_score.item(),
            'density_normalized': density_normalized.item(),
            'r_t': r_t,
            'window_size': len(sample_idx)
        }

        return sample_idx, r_t, motion_info

    def _get_idxlist(self, idx, r_t, min_idx, max_idx):
        idxlist = []
        for i in range(-r_t, r_t + 1):
            frame_idx = idx + i
            frame_idx = max(min_idx, min(max_idx, frame_idx))
            idxlist.append(frame_idx)
        return idxlist


def build_temporal_sampler(sampler_type='learnable', **kwargs):
    if sampler_type == 'learnable':
        return MotionAdaptiveTemporalSampler(**kwargs)
    elif sampler_type == 'density':
        return DensityBasedTemporalSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Must be 'learnable' or 'density'.")
