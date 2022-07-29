#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

import torch
import torch.nn.functional as F
from eteh.modules.pytorch_backend.net import transformer
from eteh.models.e2e_ctc_att.e2e_base import E2E_CTC_ATT
from eteh.utils.mask import make_pad_mask, target_mask

class EncoderMidoutput(transformer.encoder.Encoder):
    def __init__(self, *args, layer_midoutput=6, **kwargs):
        super(EncoderMidoutput, self).__init__(*args, **kwargs)
        self.layer_midoutput = len(self.encoders[:layer_midoutput])
        self.layer_finoutput = len(self.encoders[layer_midoutput:])

    def forward_midlayer(self, xs, masks):
        from eteh.modules.pytorch_backend.net.transformer.subsampling import Conv2dSubsampling
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        xs_mid, masks_mid = self.encoders[:self.layer_midoutput](xs, masks) \
            if self.layer_midoutput > 0 else (xs, masks)
        xs, masks = self.encoders[self.layer_midoutput:](xs_mid, masks_mid) \
            if self.layer_finoutput > 0 else (xs_mid, masks_mid)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, xs_mid, masks_mid

class DecoderMidoutput(transformer.decoder.Decoder):
    def __init__(self, *args, layer_midoutput=6, **kwargs):
        super(DecoderMidoutput, self).__init__(*args, **kwargs)
        self.layer_midoutput = len(self.decoders[:layer_midoutput])
        self.layer_finoutput = len(self.decoders[layer_midoutput:])

    def forward_midlayer(self, tgt, tgt_mask, memory, memory_mask):
        x = self.embed(tgt)

        x_mid, tgt_mask_mid, memory_mid, memory_mask_mid = \
            self.decoders[:self.layer_midoutput](x, tgt_mask, memory, memory_mask) \
            if self.layer_midoutput > 0 else (x, tgt_mask, memory, memory_mask)
        x, tgt_mask, memory, memory_mask = \
            self.decoders[self.layer_midoutput:](x_mid, tgt_mask_mid, memory_mid, memory_mask_mid) \
            if self.layer_finoutput > 0 else (x_mid, tgt_mask_mid, memory_mid, memory_mask_mid)

        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask, x_mid, tgt_mask_mid
    
    def forward_one_step_midlayer(self, tgt, tgt_mask, memory, cache=None):
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        if self.layer_midoutput == 0:
            x_mid, tgt_mask_mid = x, tgt_mask
        for i, (c, decoder) in enumerate(zip(cache, self.decoders),1):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)
            if i == self.layer_midoutput:
                x_mid, tgt_mask_mid = x, tgt_mask

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache, x_mid, tgt_mask_mid

class E2E_Transformer_CTC_TSE(E2E_CTC_ATT):
    def __init__(self,idim=13, odim=26, 
                 encoder_attention_dim=256, encoder_attention_heads=4, encoder_linear_units=2048, 
                 encoder_num_blocks_share=12, encoder_num_blocks_asr=0, encoder_num_blocks_tse=1, 
                 encoder_input_layer="conv2d", encoder_dropout_rate=0.1, encoder_attention_dropout_rate=0,
                 decoder_attention_dim=256, decoder_attention_heads=4, decoder_linear_units=2048, 
                 decoder_num_blocks_share=6, decoder_num_blocks_asr=0, decoder_num_blocks_tse=0, 
                 decoder_input_layer="embed", decoder_dropout_rate=0.1, decoder_src_attention_dropout_rate=0, decoder_self_attention_dropout_rate=0,
                 ctc_dropout=0.1, tse_dim=256, fix_asr_params=False):
        torch.nn.Module.__init__(self)
        self.encoder = EncoderMidoutput(
            idim=idim,
            attention_dim=encoder_attention_dim,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks_share+encoder_num_blocks_asr,
            input_layer=encoder_input_layer,
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=encoder_attention_dropout_rate,
            layer_midoutput=encoder_num_blocks_share
        )
        self.encoder_tse = transformer.encoder.Encoder(
            idim=idim,
            attention_dim=encoder_attention_dim,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks_tse,
            input_layer=torch.nn.Identity(),
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=encoder_attention_dropout_rate,
            pos_enc=torch.nn.Identity()
        ) # if encoder_num_blocks_tse > 0 else None
        self.decoder = DecoderMidoutput(
            odim=odim,
            attention_dim=decoder_attention_dim,
            attention_heads=decoder_attention_heads,
            linear_units=decoder_linear_units,
            num_blocks=decoder_num_blocks_share+decoder_num_blocks_asr,
            input_layer=decoder_input_layer,
            dropout_rate=decoder_dropout_rate,
            positional_dropout_rate=decoder_dropout_rate,
            src_attention_dropout_rate=decoder_src_attention_dropout_rate,
            self_attention_dropout_rate=decoder_self_attention_dropout_rate,
            layer_midoutput=decoder_num_blocks_share
        )
        self.decoder_tse = transformer.encoder.Encoder(
            idim=odim,
            attention_dim=decoder_attention_dim,
            attention_heads=decoder_attention_heads,
            linear_units=decoder_linear_units,
            num_blocks=decoder_num_blocks_tse,
            input_layer=torch.nn.Identity(),
            dropout_rate=decoder_dropout_rate,
            positional_dropout_rate=decoder_dropout_rate,
            attention_dropout_rate=decoder_self_attention_dropout_rate,
            pos_enc=torch.nn.Identity()
        ) # if decoder_num_blocks_tse > 0 else None
        
        self.ctc = torch.nn.Sequential(
                    torch.nn.Dropout(ctc_dropout),
                    torch.nn.Linear(encoder_attention_dim,odim)
                )
        
        self.tse_linear_enc = torch.nn.Linear(encoder_attention_dim, tse_dim)
        self.tse_linear_dec = torch.nn.Linear(decoder_attention_dim, tse_dim)

        if fix_asr_params:
            for k,v in self.named_parameters():
                if '.tse.' not in k and '.tse_' not in k and '_tse.' not in k:
                    v.requires_grad_(False)
                    v.detach_()

    def forward(self, x, xlen, ys_in, ylen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(
            xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, enc_tse, enc_tse_mask = self.encoder.forward_midlayer(xs_pad, src_mask)
        ys_mask = target_mask(ys_in)
        att_out, _, dec_tse, _ = self.decoder.forward_midlayer(ys_in, ys_mask, hs_pad, hs_mask)

        ctc_out = self.ctc(hs_pad)
        if self.encoder_tse is not None:
            enc_tse, enc_tse_mask = self.encoder_tse(enc_tse, enc_tse_mask)
        if self.decoder_tse is not None:
            dec_tse, _ = self.decoder_tse(dec_tse, None)
        enc_tse = self.tse_linear_enc(enc_tse)
        dec_tse = self.tse_linear_dec(dec_tse)
        tse_out = torch.matmul(dec_tse,enc_tse.transpose(1,2))
        
        return att_out, ctc_out, self.subfunction(hs_mask), tse_out

    def train_forward(self, input_dict):
        att_out, ctc_out, hs_len, tse_out = self.forward(
            x=input_dict["x"],
            xlen=input_dict["xlen"],
            ys_in=input_dict["ys_in"],
            ylen=input_dict["ylen"]
        )
        return {
            "att_out": att_out,
            "ctc_out": ctc_out,
            "hs_len": hs_len,
            "tse_out": tse_out
        }
    
    def encoder_forward_tse(self, x, xlen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(
            xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, enc_tse, enc_tse_mask = self.encoder.forward_midlayer(xs_pad, src_mask)
        if self.encoder_tse is not None:
            enc_tse, enc_tse_mask = self.encoder_tse(enc_tse, enc_tse_mask)
        enc_tse = self.tse_linear_enc(enc_tse)
        return hs_pad, hs_mask, enc_tse, enc_tse_mask
    
    def decoder_forward_tse(self, y, ys_mask=None, hs_pad=None, hs_mask=None):
        if ys_mask is None and hs_pad is None and hs_mask is None:
            dec_tse = y
            if self.decoder_tse is not None:
                dec_tse, _ = self.decoder_tse(dec_tse, None)
            dec_tse = self.tse_linear_dec(dec_tse)
            return dec_tse
        att_out, _, dec_tse, _ = self.decoder.forward_midlayer(y, ys_mask, hs_pad, hs_mask)
        if self.decoder_tse is not None:
            dec_tse, _ = self.decoder_tse(dec_tse, None)
        dec_tse = self.tse_linear_dec(dec_tse)
        return att_out, dec_tse

    def decoder_forward_onestep_midlayer(self, y, ys_mask, hs_pad, cache=None):
        att_out, new_cache, dec_midlayer, _ = self.decoder.forward_one_step_midlayer(
            y, ys_mask, hs_pad, cache)
        return att_out, new_cache, dec_midlayer

    def tse_forward(self, enc_tse, dec_tse):
        return torch.matmul(dec_tse,enc_tse.transpose(1,2))
    
    def get_input_dict(self):
        return {"x": "(B,T,D)", "xlen": "(B)", "ys_in": "(B,N)", "ylen": "(B)"}

    def get_out_dict(self):
        return {"att_out": "(B,N,O)", "ctc_out": "(B,T,O)", "hs_len": "(B)", "tse_out": "(B,N,T)"}
        

    # __encoder_midoutput:
    # def forward_one_step(self, xs, masks, cache=None):
    #     """Encode input frame.

    #     Args:
    #         xs (torch.Tensor): Input tensor.
    #         masks (torch.Tensor): Mask tensor.
    #         cache (List[torch.Tensor]): List of cache tensors.

    #     Returns:
    #         torch.Tensor: Output tensor.
    #         torch.Tensor: Mask tensor.
    #         List[torch.Tensor]: List of new cache tensors.

    #     """
    #     from eteh.modules.pytorch_backend.net.transformer.subsampling import Conv2dSubsampling
    #     if isinstance(self.embed, Conv2dSubsampling):
    #         xs, masks = self.embed(xs, masks)
    #     else:
    #         xs = self.embed(xs)
    #     if cache is None:
    #         cache = [None for _ in range(len(self.encoders))]
    #     new_cache = []
    #     if len(self.encoders[:self.layer_midoutput])>0:
    #         for c, e in zip(cache, self.encoders[:self.layer_midoutput]):
    #             xs, masks = e(xs, masks, cache=c)
    #             new_cache.append(xs)
    #     xs_mid, masks_mid = xs, masks
    #     if len(self.encoders[self.layer_midoutput:])>0:
    #         for c, e in zip(cache, self.encoders[self.layer_midoutput:]):
    #             xs, masks = e(xs, masks, cache=c)
    #             new_cache.append(xs)
    #     if self.normalize_before:
    #         xs = self.after_norm(xs)
    #     return xs, masks, new_cache, xs_mid, masks_mid