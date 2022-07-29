import torch
import torch.nn.functional as F
from eteh.modules.pytorch_backend.net import transformer
from eteh.models.e2e_ctc_att.e2e_base import E2E_CTC_ATT

class E2E_Transformer_CTC(E2E_CTC_ATT):

    def __init__(self,idim=13, odim=26, 
                 encoder_attention_dim=256, encoder_attention_heads=4, encoder_linear_units=2048, 
                 encoder_num_blocks=12, encoder_input_layer="conv2d", encoder_dropout_rate=0.1, encoder_attention_dropout_rate=0,
                 decoder_attention_dim=256, decoder_attention_heads=4, decoder_linear_units=2048, decoder_num_block=6, 
                 decoder_input_layer="embed", decoder_dropout_rate=0.1, decoder_src_attention_dropout_rate=0, decoder_self_attention_dropout_rate=0,
                 ctc_dropout=0.1):
        torch.nn.Module.__init__(self)
        self.encoder = transformer.encoder.Encoder(
            idim=idim,
            attention_dim=encoder_attention_dim,
            attention_heads=encoder_attention_heads,
            linear_units=encoder_linear_units,
            num_blocks=encoder_num_blocks,
            input_layer=encoder_input_layer,
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=encoder_attention_dropout_rate
        )
        self.decoder = transformer.decoder.Decoder(
            odim=odim,
            attention_dim=decoder_attention_dim,
            attention_heads=decoder_attention_heads,
            linear_units=decoder_linear_units,
            num_blocks=decoder_num_block,
            input_layer=decoder_input_layer,
            dropout_rate=decoder_dropout_rate,
            positional_dropout_rate=decoder_dropout_rate,
            src_attention_dropout_rate=decoder_src_attention_dropout_rate,
            self_attention_dropout_rate=decoder_self_attention_dropout_rate
        )
        
        self.ctc = torch.nn.Sequential(
                    torch.nn.Dropout(ctc_dropout),
                    torch.nn.Linear(encoder_attention_dim,odim)
                )

