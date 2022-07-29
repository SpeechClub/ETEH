import torch

from torch import nn

from eteh.modules.pytorch_backend.net.transformer.layer_norm import LayerNorm


class StreamEncoderLayer(nn.Module):
    """Stream Encoder layer module
       Split the utterance into chunks (e.g. memory + target)

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param int hop_len: hop size for each chunk
    :param int mem_len: memory size for each chunk, stored and reused
    :param int tgt_len: target size for each chunk, equal to output size and involves back propagation
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool use_grad: whether the memory involves back propagation
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout_rate,
                 hop_len, mem_len, tgt_len, normalize_before=True, concat_after=False, use_grad=False):
        super(StreamEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.use_grad = use_grad
        self.hop_len = hop_len
        self.mem_len = mem_len
        self.tgt_len = tgt_len
        self.use_mem = mem_len > 0
        self.register_buffer('mems', None)
        
    def init_mems(self):
        param = next(self.parameters())
        if self.use_mem:
            self.mems = torch.empty(0, dtype=param.dtype, device=param.device)
        else:
            self.mems = None
    
    def update_mems(self, x):
        if self.mems.dim() > 1:
            mem_len = self.mems.size(1)
        else:
            mem_len = 0
        end_idx = mem_len + max(0, self.hop_len)
        beg_idx = max(0, end_idx - self.mem_len)
        if self.use_grad:
            self.mems = torch.cat([self.mems, x], dim=1)[:, beg_idx:end_idx]              
        else:
            with torch.no_grad():
                self.mems = torch.cat([self.mems, x], dim=1)[:, beg_idx:end_idx].detach().to(x.device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class CashedEncoderLayer(StreamEncoderLayer):
    """Cashed Encoder layer module with absolute position embedding

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param int hop_len: hop size for each chunk
    :param int mem_len: memory size for each chunk, stored and reused
    :param int tgt_len: target size for each chunk, equal to output size and involves back propagation
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool use_grad: whether the memory involves back propagation
    """

    def __init__(self, *args, **kwargs):
        super(CashedEncoderLayer, self).__init__(*args, **kwargs)
            
    def forward(self, x, mask):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.use_mem and self.mems is None:
            self.init_mems()
        if self.mems is not None:
            kx = torch.cat([self.mems, x], dim=1)
        else:
            kx = x
        if self.mems is not None and self.mems.dim() > 1:
            mem_mask = torch.ones(mask.size(0), 1, self.mems.size(1)).byte().to(mask.device)
            kmask = torch.cat([mem_mask, mask], dim=-1)
        else:
            kmask = mask
        if self.use_mem:
            self.update_mems(x)            
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, kx, kx, kmask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, kx, kx, kmask))            
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        return x, mask
  