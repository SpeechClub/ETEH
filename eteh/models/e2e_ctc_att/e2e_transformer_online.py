# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Haoran Miao     miaohaoran@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import torch, torch.nn
from eteh.models.e2e_ctc_att.e2e_base import E2E_CTC_ATT
from eteh.utils.mask import make_pad_mask
from eteh.modules.pytorch_backend.net.online_transformer.encoder import ChunkEncoder
from eteh.modules.pytorch_backend.net.online_transformer.decoder import StreamDecoder

class E2E_Transformer_CTC_Online(E2E_CTC_ATT):
    def __init__(self,idim=13,odim=26, 
                 encoder_attention_dim=256, encoder_attention_heads=4, encoder_left_chunk=64, encoder_center_chunk=64, encoder_right_chunk=64,
                 encoder_linear_units=2048, encoder_num_blocks=12, encoder_input_layer="conv2d", encoder_dropout_rate=0.1, encoder_attention_dropout_rate=0,
                 decoder_attention_dim=256, decoder_self_attention_heads=4, decoder_src_attention_heads=4, decoder_linear_units=2048, decoder_num_block=6,
                 decoder_input_layer="embed", decoder_dropout_rate=0.1, decoder_src_attention_dropout_rate=0, decoder_self_attention_dropout_rate=0,
                 decoder_src_attention_bias_init=0, decoder_src_attention_sigmoid_noise=1,
                 ctc_dropout=0.1):
        torch.nn.Module.__init__(self)
        self.encoder = ChunkEncoder(
            idim=idim,
            attention_dim=encoder_attention_dim,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks,
            input_layer=encoder_input_layer,
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=encoder_attention_dropout_rate,
            left_len=encoder_left_chunk,
            cur_len=encoder_center_chunk,
            right_len=encoder_right_chunk,
            hop_len=encoder_center_chunk,
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
                    torch.nn.Linear(encoder_attention_dim, odim)
                )

    def encoder_forward_online(self,x,xlen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder.forward(xs_pad, src_mask)
        return hs_pad, hs_mask

    def decoder_forward_online(self, y, ys_mask, hs_pad, cache=None):
        ys_pad, new_cache = self.decoder.forward_one_step_online(y, ys_mask, hs_pad, cache)
        return ys_pad, new_cache