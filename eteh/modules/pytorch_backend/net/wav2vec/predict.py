# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import norm_block



class Wav2VecPredictionsModel(nn.Module):
    def __init__(self, in_dim, out_dim, prediction_steps, n_negatives, cross_sample_negatives, sample_distance,
                 dropout, offset, balanced_classes):
        super().__init__()

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance

        self.project_to_steps = nn.ConvTranspose2d(in_dim, out_dim, (1, prediction_steps))
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        if self.cross_sample_negatives:
            high = tsz * bsz
            assert self.sample_distance is None, 'sample distance is not supported with cross sampling'
        else:
            high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        if self.sample_distance is not None and self.sample_distance < tsz:
            neg_idxs += torch.cat(
                [torch.arange(start=1, end=tsz - self.sample_distance, device=neg_idxs.device, dtype=neg_idxs.dtype),
                 torch.arange(start=tsz - self.sample_distance, end=tsz - self.sample_distance * 2 - 1, step=-1,
                              device=neg_idxs.device, dtype=neg_idxs.dtype)])

        if not self.cross_sample_negatives:
            for i in range(1, bsz):
                neg_idxs[i] += i * high

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(fsz, bsz, self.n_negatives, tsz).permute(2, 1, 0, 3)  # to NxBxCxT

        return negs

    def forward(self, x, y):
        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)
        #print(targets.size())
        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)  # BxCxTxS
        x = self.dropout(x)
        x = x.unsqueeze(0).expand(targets.size(0), -1, -1, -1, -1)
        #print(x.size())        
        copies, bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)
        predictions = x.new(bsz * copies * (tsz - self.offset + 1) * steps - ((steps + 1) * steps // 2) * copies * bsz)
        labels = torch.zeros_like(predictions)
        weights = torch.full_like(labels, 1 / self.n_negatives) if self.balanced_classes else None

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            pos_num = (end - start) // copies
            predictions[start:end] = (x[..., :-offset, i] * targets[..., offset:]).sum(dim=2).flatten()
            labels[start:start + pos_num] = 1.
            if weights is not None:
                weights[start:start + pos_num] = 1.
            start = end
        assert end == predictions.numel(), '{} != {}'.format(end, predictions.numel())

        if weights is not None:
            labels = (labels, weights)

        return predictions, labels
