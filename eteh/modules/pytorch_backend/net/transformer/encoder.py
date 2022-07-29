import torch

from .attention import MultiHeadedAttention
from .embedding import PositionalEncoding
from .encoder_layer import EncoderLayer
from .layer_norm import LayerNorm
from .positionwise_feed_forward import PositionwiseFeedForward
from .repeat import repeat
from .subsampling import Conv2dSubsampling
from .embedding import PositionalEncoding

class Encoder(torch.nn.Module):
    """Transformer encoder module

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, idim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="conv2d",
                 pos_enc=None,
                 normalize_before=True,
                 concat_after=False):
        super(Encoder, self).__init__()
        if pos_enc is None:
            pos_enc = PositionalEncoding(attention_dim, positional_dropout_rate)
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate, pos_enc)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim),
                pos_enc
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc,
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
        
class DualEncoder(torch.nn.Module):
    """Transformer encoder module for dual version

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int attention_chunk: the number of sub-sampled frames in each chunk
    :param int attention_left: the number of chunks to look on the left, -1 for look all
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, idim,
                 attention_dim=256,
                 attention_heads=4,
                 attention_chunk=16,
                 attention_left=-1,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="conv2d",
                 pos_enc=None,
                 normalize_before=True,
                 concat_after=False):
        super(DualEncoder, self).__init__()
        if pos_enc is None:
            pos_enc = PositionalEncoding(attention_dim, positional_dropout_rate)
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate, pos_enc)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim),
                pos_enc
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc,
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        self.register_buffer('att_mask', None)
        self.register_mask(attention_left, attention_chunk)
        self.chunk = attention_chunk

    def register_mask(self, left, chunk):
        param = next(self.parameters())
        num = (5000 + chunk - 1) // chunk
        ret = torch.tril(torch.ones(num, num, device=param.device, dtype=torch.uint8))
        if num > left >= 0:
            ret[left:, :num - left] = torch.triu(ret[left:, :num - left])
        chk = torch.ones(chunk, chunk, device=param.device, dtype=torch.uint8)
        self.att_mask = torch.einsum("ab,cd->acbd", ret, chk).view(1, num * chunk, -1)

    def forward(self, xs, masks):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        # offline version
        xs_off, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs_off = self.after_norm(xs_off)
        # online version
        hlen = xs.size(1)
        if masks is None:
            masks_on = self.att_mask[:, :hlen, :hlen]
        else:
            masks_on = self.att_mask[:, :hlen, :hlen] & masks
        xs_on, masks_on = self.encoders(xs, masks_on)
        if self.normalize_before:
            xs_on = self.after_norm(xs_on)
        return xs_off, xs_on, masks

    def forward_offline(self, xs, masks):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
        
    def forward_online(self, xs, masks):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        hlen = xs.size(1)
        if masks is None:
            masks = self.att_mask[:, :hlen, :hlen]
        else:
            masks = self.att_mask[:, :hlen, :hlen] & masks
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
        
    def forward_per_chunk(self, xs, masks, cache=None):
        """Encode input frames by chunk.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        if cache is None:
            cache = [None for _ in range(len(self.encoders) + 1)]
        new_cache = []
        if cache[0] is None:
            offset = 0
            xs_q = xs
        else:
            offset = cache[0].size(1)
            idx =  offset * 4 - xs.size(1)
            xs_q = xs[:, idx:, :]
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs_q, masks, offset)
        else:
            xs = self.embed(xs_q)
        if cache[0] is not None:
            xs = torch.cat([cache[0], xs], dim=1)
        new_cache.append(xs)
        hlen = xs.size(1)
        if masks is None:
            masks = self.att_mask[:, :hlen, :hlen]
        else:
            masks = self.att_mask[:, :hlen, :hlen] & masks[:, :hlen, :hlen]
        chunk = hlen - cache[0].size(1) if cache[0] is not None else hlen
        for c, e in zip(cache[1:], self.encoders):
            xs, masks = e(xs, masks, c, chunk)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs[:, -chunk:])
        return xs, new_cache