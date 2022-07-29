# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import norm_block

class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers, dropout, log_compression, skip_connections, residual_scale, non_affine_group_norm):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm),
                nn.ReLU(),
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, (dim, k, stride) in enumerate(conv_layers):
            self.conv_layers.append(
                block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., ::r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x