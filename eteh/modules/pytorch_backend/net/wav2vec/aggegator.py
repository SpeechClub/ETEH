# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import norm_block

class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))

class ConvAggegator(nn.Module):
    def __init__(self, conv_layers, embed, dropout, skip_connections, residual_scale, non_affine_group_norm, conv_bias,
                 zero_pad):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                nn.ReLU(),
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for i, (dim, k, stride) in enumerate(conv_layers):
            if in_d != dim and skip_connections:
                self.residual_proj.append(
                    nn.Conv1d(in_d, dim, 1, bias=False),
                )
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(
                block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x