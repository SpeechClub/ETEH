# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import torch
import torch.nn.functional as F
from eteh.utils.mask import make_pad_mask, target_mask
from eteh.models.model_interface import Model_Interface


class E2E_CTC_ATT(torch.nn.Module, Model_Interface):
    def __init__(self, encoder, decoder, attention, ctc):
        torch.nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

    def forward(self, x, xlen, ys_in, ylen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(
            xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        ys_mask = target_mask(ys_in)
        att_out, _ = self.decoder(ys_in, ys_mask, hs_pad, hs_mask)
        ctc_out = self.ctc(hs_pad)
        return att_out, ctc_out, self.subfunction(hs_mask)

    def train_forward(self, input_dict):
        att_out, ctc_out, hs_len = self.forward(
            x=input_dict["x"],
            xlen=input_dict["xlen"],
            ys_in=input_dict["ys_in"],
            ylen=input_dict["ylen"]
        )
        return {
            "att_out": att_out,
            "ctc_out": ctc_out,
            "hs_len": hs_len
        }

    def get_input_dict(self):
        return {"x": "(B,T,D)", "xlen": "(B,T)", "ys_in": "(B,N)", "ylen": "(B)"}

    def get_out_dict(self):
        return {"att_out": "(B,N,O)", "ctc_out": "(B,T,O)", "hs_len": "(B)"}

    def subfunction(self, hs_mask):
        xlen = torch.sum(hs_mask.byte(), dim=-1).squeeze(-1)
        return xlen

    def get_ctc_prob(self, x, xlen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(
            xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        ctc_out = self.ctc(hs_pad)        
        return ctc_out

    def ctc_forward(self, enc_out):
        ctc_out = self.ctc(enc_out)
        return ctc_out

    def att_forward(self, x, xlen, y, ylen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(
            xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        ys_mask = target_mask(y)
        att_out, _ = self.decoder(y, ys_mask, hs_pad, hs_mask)
        return att_out

    def encoder_forward(self, x, xlen):
        xs_pad = x
        src_mask = (~make_pad_mask(xlen.tolist(), max_length=xs_pad.size(1))).to(
            xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        return hs_pad, hs_mask

    def decoder_forward(self, y, ys_mask, hs_pad, hs_mask):
        att_out, _ = self.decoder(y, ys_mask, hs_pad, hs_mask)
        return att_out

    def decoder_forward_onestep(self, y, ys_mask, hs_pad, cache=None):
        att_out, new_cache = self.decoder.forward_one_step(
            y, ys_mask, hs_pad, cache)
        return att_out, new_cache

    def decoder_forward_onestep_batch(self, y, ys_mask, hs_pad, hs_mask, cache=None):
        ys_pad, new_cache = self.decoder.forward_one_step_batch(
            y, ys_mask, hs_pad, hs_mask, cache)
        return ys_pad, new_cache

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, y_len):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, y_len)
        ret = dict()
        for name, m in self.named_modules():
            from eteh.models.pytorch_backend.net.transformer.attention import MultiHeadedAttention
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.detach().cpu().numpy()
        return ret
