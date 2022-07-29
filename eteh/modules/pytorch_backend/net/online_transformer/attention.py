#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn

from eteh.modules.pytorch_backend.net.transformer.attention import MultiHeadedAttention


def safe_cumprod(x, *args, **kwargs):
    """Computes cumprod of x in logspace using cumsum to avoid underflow.
    The cumprod function and its gradient can result in numerical instabilities
    when its argument has very small and/or zero values.  As long as the argument
    is all positive, we can instead compute the cumulative product as
    exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.
    Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.
    Returns:
        Cumulative product of x, the first element is 1.
    """
    tiny_value = float(numpy.finfo(torch.tensor(0, dtype=x.dtype).numpy().dtype).tiny)
    exclusive_cumprod = torch.exp(torch.cumsum(torch.log(torch.clamp(x[..., :-1], tiny_value, 1.)), *args, **kwargs))
    shape = list(exclusive_cumprod.size())
    shape[-1] = 1
    return torch.cat([torch.ones(shape, dtype=x.dtype).to(x.device), exclusive_cumprod], dim=-1)

class MTMultiHeadedAttention(MultiHeadedAttention):
    """ Monotonic Truncated Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate, bias_init=0.0, sigmoid_noise=1.0):
        """Construct an MultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.src_att_bias = nn.Parameter(torch.Tensor(1,1))
        nn.init.constant_(self.src_att_bias, bias_init)
        self.sigmoid_noise = sigmoid_noise

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if self.sigmoid_noise > 0 and self.training:
            noise = torch.normal(mean=torch.zeros(scores.shape), std=1).to(scores.device)
            scores += self.sigmoid_noise * noise
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            p_choose_i = torch.sigmoid(scores).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            p_choose_i = torch.sigmoid(scores)  # (batch, head, time1, time2)
        cumprod_1mp_choose_i = safe_cumprod(1-p_choose_i, dim=-1)
        self.attn = p_choose_i * cumprod_1mp_choose_i
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def decode_attention(self, value, scores, endpoint):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (1, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (1, n_head, 1, time2).
            endpoint (list): End-pointer (head).

        Returns:
            torch.Tensor: Transformed value (1, 1, d_model)
                weighted by the attention score (1, 1, time2).

        """
        n_batch = value.size(0)
        p_choose_i = torch.sigmoid(scores)
        cumprod_1mp_choose_i = safe_cumprod(1-p_choose_i, dim=-1)
        self.attn = p_choose_i * cumprod_1mp_choose_i
        if not isinstance(endpoint, list):
            endpoint = [endpoint] * self.h
        self.endpoint = endpoint
        for h in range(self.h):
            for j in range(endpoint[h] + 1, scores.size(3)):
                if scores[0,h,0,j] > 0:
                    self.endpoint[h] = j
                    break
            self.attn[0,h,0,self.endpoint[h]+1:] = 0.0
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, ep=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            ep (list): End pointers of last step

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) + self.src_att_bias
        if ep is None:
            return self.forward_attention(v, scores, mask)
        else:
            return self.decode_attention(v, scores, ep)
