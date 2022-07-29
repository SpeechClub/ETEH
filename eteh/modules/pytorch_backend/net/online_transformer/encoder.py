import torch
from eteh.modules.pytorch_backend.net.transformer.attention import MultiHeadedAttention
from eteh.modules.pytorch_backend.net.transformer.embedding import PositionalEncoding
from eteh.modules.pytorch_backend.net.transformer.layer_norm import LayerNorm
from eteh.modules.pytorch_backend.net.transformer.positionwise_feed_forward import PositionwiseFeedForward
from eteh.modules.pytorch_backend.net.transformer.repeat import repeat
from eteh.modules.pytorch_backend.net.transformer.subsampling import Conv2dSubsampling
from eteh.modules.pytorch_backend.net.transformer.embedding import PositionalEncoding
from eteh.modules.pytorch_backend.net.transformer.encoder_layer import EncoderLayer
from .encoder_layer import CashedEncoderLayer
import random



class ChunkEncoder(torch.nn.Module):
    """TransformerXL encoder module,@ref "Transformer-XL_AttentiveLanguageModels BeyondaFixed-LengthContext"
        input streams are segmented into chunks (left + current + right) with hop size
        
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
    :param int left_len: left frames legnth, to provide futrue infomation
    :param int cur_len: current frames legnth, only current frames involves back propagation
    :param int right_len: right frames length, to provide futrue infomation
    :param int hop_len: right frames length, to provide futrue infomation
    :param bool use_mem: whether to reuse computed history infomation
    :param bool use_grad: whether to backward to memory
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
                 concat_after=False,
                 left_len=64,  
                 cur_len=64, 
                 right_len=64,
                 hop_len=64,
                 use_mem=True,
                 use_grad=False,
                 ):
        super(ChunkEncoder, self).__init__()
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

        self.cur_len = cur_len
        self.left_len = left_len if not use_mem else 0
        self.right_len = right_len
        self.hop_len = hop_len
        self.use_mem = use_mem
        self.mem_len = left_len if use_mem else 0
        self.chunk_len = self.left_len + self.cur_len + self.right_len
        
        if input_layer == "conv2d":
            self.cur_len_sub = self.cur_len // 4
            self.left_len_sub = self.left_len // 4
            #self.right_len_sub = self.right_len // 4
            self.hop_len_sub = self.hop_len // 4
            self.mem_len_sub = self.mem_len // 4
        else:
            self.cur_len_sub = self.cur_len
            self.left_len_sub = self.left_len
            #self.right_len_sub = self.right_len
            self.hop_len_sub = self.hop_len
            self.mem_len_sub = self.mem_len          
        
        self.encoders = repeat(
            num_blocks,
            lambda lnum: CashedEncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                self.hop_len_sub,
                self.mem_len_sub,
                self.cur_len_sub, 
                normalize_before,
                concat_after,
                use_grad,
            )
        )            
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        
    def _forward(self, xs, masks, pos):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor，(batch,time,dim)
        :param torch.Tensor masks: (batch,1,time),1 means data,zero means padding
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks, offset=pos)
        else:
            xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def chunk_iter(self, xs, masks):
        r_xs = torch.zeros(xs.size(0), self.right_len + 6, xs.size(2)).to(xs.device)
        r_masks = torch.zeros(masks.size(0), 1, self.right_len + 6).byte().to(masks.device)
        l_xs = torch.zeros(xs.size(0), self.left_len, xs.size(2)).to(xs.device)
        l_masks = torch.zeros(masks.size(0), 1, self.left_len).byte().to(masks.device)
        xs = torch.cat([l_xs, xs, r_xs], dim=1)
        masks = torch.cat([l_masks, masks, r_masks], dim=2)
        i = 0
        while (i + self.chunk_len) < xs.size(1) - 6 + self.hop_len:
            yield xs[:, i:i+self.chunk_len], \
                  masks[:, :, i:i+self.chunk_len]
            i += self.hop_len

    def forward(self, xs, masks):
        if masks is None:
            masks = torch.ones(xs.size(0), 1, xs.size(1)).byte().to(xs.device)           
        if self.cur_len_sub == 0: # basic transformer
            xs, masks = self._forward(xs, masks, 0)
        else:
            xs_list = []
            mask_list = []
            iter = self.chunk_iter(xs, masks)
            for i, (chunk_xs, chunk_masks) in enumerate(iter):
                _chunk_xs, _chunk_masks = self._forward(chunk_xs, chunk_masks, i * self.hop_len_sub)
                xs_list.append(_chunk_xs[:, self.left_len_sub:self.left_len_sub+self.cur_len_sub])
                mask_list.append(_chunk_masks[:, :, self.left_len_sub:self.left_len_sub+self.cur_len_sub])
            xs = torch.cat(xs_list, dim=1)
            masks = torch.cat(mask_list, dim=2)
            # prepare for next batch
            if self.use_mem:
                for layer in self.encoders._modules.values():
                    if isinstance(layer, CashedEncoderLayer):
                        layer.mems = None
        return xs, masks
        
class ParallelDynamicDualEncoder(torch.nn.Module):
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
        super().__init__()
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
        param = next(self.parameters())
        self.att_mask = torch.zeros(17, 1250, 1250, device=param.device, dtype=torch.uint8)
        for i in range(17):
            self.register_mask(attention_left, attention_chunk, i)
        self.chunk = attention_chunk

    def register_mask(self, left, chunk, idx):
        chunk += idx - 8
        param = next(self.parameters())
        num = (1250 + chunk - 1) // chunk
        ret = torch.tril(torch.ones(num, num, device=param.device, dtype=torch.uint8))
        if num > left >= 0:
            ret[left:, :num - left] = torch.triu(ret[left:, :num - left])
        chk = torch.ones(chunk, chunk, device=param.device, dtype=torch.uint8)
        att_mask = torch.einsum("ab,cd->acbd", ret, chk).view(1, num * chunk, -1)
        self.att_mask[idx:idx+1] = att_mask[:, :1250, :1250]

    def forward(self, xs, masks):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        batch_size = xs.shape[0]
        
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        assert masks is not None


        # prepare online masks
        hlen = xs.size(1)
        idx = random.randint(0,16)
        if masks is None:
            masks_on = self.att_mask[idx:idx+1, :hlen, :hlen]
        else:
            masks_on = self.att_mask[idx:idx+1, :hlen, :hlen] & masks

        # print(masks.shape)
        # print(masks_on.shape)

        

        masks_all = torch.cat([masks.expand(-1, masks.size(-1), -1), masks_on], dim=0)
        xs_repeat = xs.repeat(2, 1, 1)

        # print(masks_all.shape)
        # print(xs_repeat.shape)

        # forward_all
        xs_repeat, masks_all = self.encoders(xs_repeat, masks_all, None)

        if self.normalize_before:
            xs_repeat = self.after_norm(xs_repeat)

        # 这里未使用 encoder 输出的 mask
        # assert (masks == masks_all[:batch_size]).all()

        return xs_repeat, masks

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
            masks = self.att_mask[8:9, :hlen, :hlen]
        else:
            masks = self.att_mask[8:9, :hlen, :hlen] & masks
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
        
    def forward_per_chunk(self, xs, masks, cache=None, right=0):
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
        right //= 4
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
        if right > 0:
            new_cache.append(xs[:, :-right])
        else:
            new_cache.append(xs)
        hlen = xs.size(1)
        if masks is None:
            masks = self.att_mask[8:9, :hlen, :hlen]
        else:
            masks = self.att_mask[8:9, :hlen, :hlen] & masks[:, :hlen, :hlen]
        chunk = hlen - cache[0].size(1) if cache[0] is not None else hlen
        for c, e in zip(cache[1:], self.encoders):
            xs, masks = e(xs, None, c, chunk)
            if right > 0:
                new_cache.append(xs[:, :-right])
            else:
                new_cache.append(xs)
        if self.normalize_before:
            if right > 0:
                xs = self.after_norm(xs[:, -chunk:-right])
            else:
                xs = self.after_norm(xs[:, -chunk:])
        return xs, new_cache

