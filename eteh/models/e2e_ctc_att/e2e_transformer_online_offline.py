import torch
from eteh.utils.mask import make_pad_mask, target_mask
from eteh.modules.pytorch_backend.net.online_transformer.encoder import ParallelDynamicDualEncoder
from eteh.modules.pytorch_backend.net.online_transformer.decoder import StreamDecoder
from eteh.models.e2e_ctc_att.e2e_base import E2E_CTC_ATT

class E2E_Transformer_CTC_Univ_Dynamic(E2E_CTC_ATT):

    def __init__(self,idim=13,odim=26, 
                 encoder_attention_dim=256, encoder_attention_heads=4, encoder_attention_chunk=16, encoder_attention_left=-1,encoder_linear_units=2048, 
                 encoder_num_blocks=12, encoder_input_layer="conv2d", encoder_dropout_rate=0.1, encoder_attention_dropout_rate=0,
                 decoder_attention_dim=256, decoder_self_attention_heads=4, decoder_src_attention_heads=4, decoder_linear_units=2048, decoder_num_block=6, 
                 decoder_input_layer="embed", decoder_dropout_rate=0.1, decoder_src_attention_dropout_rate=0, decoder_self_attention_dropout_rate=0,
                 decoder_src_attention_bias_init=0, decoder_src_attention_sigmoid_noise=1,
                 ctc_dropout=0.1):
        torch.nn.Module.__init__(self)
        self.encoder = ParallelDynamicDualEncoder(
            idim=idim,
            attention_dim=encoder_attention_dim,
            attention_heads=encoder_attention_heads,
            attention_chunk=encoder_attention_chunk,
            attention_left=encoder_attention_left,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks,
            input_layer=encoder_input_layer,
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=encoder_attention_dropout_rate
        )
        self.decoder = StreamDecoder(
            odim=odim,
            attention_dim=decoder_attention_dim,
            self_attention_heads=decoder_self_attention_heads,
            src_attention_heads=decoder_src_attention_heads,
            linear_units=decoder_linear_units,
            num_blocks=decoder_num_block,
            input_layer=decoder_input_layer,
            dropout_rate=decoder_dropout_rate,
            positional_dropout_rate=decoder_dropout_rate,
            src_attention_dropout_rate=decoder_src_attention_dropout_rate,
            self_attention_dropout_rate=decoder_self_attention_dropout_rate,
            src_attention_bias_init=decoder_src_attention_bias_init,
            src_attention_sigmoid_noise=decoder_src_attention_sigmoid_noise,
        )
        
        self.ctc = torch.nn.Sequential(
                    torch.nn.Dropout(ctc_dropout),
                    torch.nn.Linear(encoder_attention_dim,odim)
                )

    def forward(self,x,xlen,y_in,ylen):

        batch_size = x.shape[0]
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(xs_pad.device).unsqueeze(-2)
        hs_all, hs_mask = self.encoder(xs_pad, src_mask)

        ys_mask = target_mask(y_in)
        
        # ys_mask = ys_mask.repeat(2, )

        y_all = y_in.repeat(2, 1)
        ys_mask_all = ys_mask.repeat(2, 1, 1)
        hs_mask_all = hs_mask.repeat(2, 1, 1)

        att_all, _ = self.decoder(y_all, ys_mask_all, hs_all, hs_mask_all)

        att_out_off = att_all[:batch_size]
        att_out_on = att_all[batch_size:]

        ali_out = [decoder.src_attn.attn[batch_size:] for decoder in self.decoder.decoders if decoder.src_attn is not None]        
        ali_out = torch.cat(ali_out, dim=1)

        ctc_out_all = self.ctc(hs_all)
        ctc_out_off = ctc_out_all[:batch_size]
        ctc_out_on = ctc_out_all[batch_size:]
        
        hs_len = self.subfunction(hs_mask)
        return att_out_on, ctc_out_on, ali_out, att_out_off, ctc_out_off, hs_len

    def train_forward(self, input_dict):
        att_out_on, ctc_out_on, ali_out, att_out_off, ctc_out_off, hs_len = self.forward(
            x=input_dict["x"],
            xlen=input_dict["xlen"],
            y_in=input_dict["ys_in"],
            ylen=input_dict["ylen"]
        )
        return {
            "att_out_on": att_out_on,
            "ctc_out_on": ctc_out_on,
            "ali_out": ali_out,
            "att_out_off": att_out_off,
            "ctc_out_off": ctc_out_off,
            "hs_len": hs_len
        }

    def encoder_forward(self,x,xlen,online=False):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen, max_length=xs_pad.size(1))).to(xs_pad.device).unsqueeze(-2)
        if online:
            hs_pad, hs_mask = self.encoder.forward_online(xs_pad, src_mask)
        else:
            hs_pad, hs_mask = self.encoder.forward_offline(xs_pad, src_mask)

        return hs_pad, hs_mask

    def encoder_forward_online(self,x,xlen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder.forward_online(xs_pad, src_mask)
        return hs_pad, hs_mask

    def decoder_forward_online(self, y, ys_mask, hs_pad, cache=None):
        ys_pad, new_cache = self.decoder.forward_one_step_online(y, ys_mask, hs_pad, cache)
        return ys_pad, new_cache
