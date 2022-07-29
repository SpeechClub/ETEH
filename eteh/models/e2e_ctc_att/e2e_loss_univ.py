# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from eteh.modules.pytorch_backend.criterion.customize_loss import KL_Loss
from eteh.modules.pytorch_backend.criterion.ali_loss import Align_Loss
from eteh.models.e2e_ctc_att.e2e_loss import E2E_Loss
from eteh.utils.mask import make_pad_mask
from eteh.utils.data_utils import calcurate_cer, calculate_cer_ctc

class CTC_CE_Univ_Loss(E2E_Loss):
    def __init__(self, size, padding_idx, smoothing, rate=0.5, kl_rate=1.0, ali_rate=1.0, ali_type='mid'):
        super(CTC_CE_Univ_Loss, self).__init__(size, padding_idx, smoothing, rate)
        self.ali_rate = ali_rate
        self.kl_rate = kl_rate
        self.kl_loss = KL_Loss(size)
        self.ali_loss = Align_Loss(ali_type, padding_idx)
        self.padding_idx = padding_idx

    def forward(self, att_out, ctc_out, ali_out, att_out_off, ctc_out_off, data_len, att_label, ctc_label, ctc_len, label_beg=None, label_end=None):
        #with autocast(): //增加这个修饰才能正确运行amp
        att_loss = self.att_loss(att_out, att_label)
        att_loss_off = self.att_loss(att_out_off, att_label)
        kl_loss = self.kl_loss(att_out, att_out_off, att_label == self.padding_idx)
        ctc_loss = self.ctc_loss(
            ctc_out, ctc_len, ctc_label)
        ctc_loss_off = self.ctc_loss(
            ctc_out_off, ctc_len, ctc_label)
        ctc_mask = (make_pad_mask(ctc_len.tolist(), max_length=ctc_out.size(1))).to(ctc_out.device)
        kl_loss += self.kl_loss(ctc_out, ctc_out_off, ctc_mask)
        ali_loss = self.ali_loss(ali_out, label_beg, label_end, ctc_mask, ctc_out_off, ctc_label, ctc_len) \
            if label_beg is not None else att_loss.new_zeros(att_loss.size())
        return (1 - self.rate) * (att_loss + att_loss_off) \
               + self.rate * (ctc_loss + ctc_loss_off) \
               + self.ali_rate * ali_loss \
               + self.kl_rate * kl_loss, \
               att_loss, ctc_loss, ali_loss, kl_loss

               
    def train_forward(self, input_dict):
        loss_main, att_loss, ctc_loss, ali_loss, kl_loss = self.forward(
            att_out=input_dict["att_out_on"],
            ctc_out=input_dict["ctc_out_on"],
            ali_out=input_dict["ali_out"],
            att_out_off=input_dict["att_out_off"],
            ctc_out_off=input_dict["ctc_out_off"],
            att_label=input_dict["ys_out"],
            ctc_label=input_dict["y"],
            data_len=input_dict["hs_len"],
            ctc_len=input_dict["hs_len"],
            label_beg = input_dict["y_beg"] if "y_beg" in input_dict else None,
            label_end = input_dict["y_end"] if "y_end" in input_dict else None,
        )
        att_corr_on = calcurate_cer(input_dict["att_out_on"].detach().cpu().numpy(), input_dict["ys_out"].detach().cpu().numpy())            
        att_corr_off = calcurate_cer(input_dict["att_out_off"].detach().cpu().numpy(), input_dict["ys_out"].detach().cpu().numpy())            
        
        return {
            "loss_main": loss_main,
            "att_loss": att_loss.item(),
            "ctc_loss": ctc_loss.item(),
            "ali_loss": ali_loss.item(),
            "kl_loss": kl_loss.item(),
            "att_corr_on": att_corr_on,
            "att_corr_off": att_corr_off,
        }
        
    def valid_forward(self, input_dict):
        valid_dict = self.train_forward(input_dict)
        ctc_corr_off = calculate_cer_ctc(input_dict["ctc_out_off"].cpu().numpy(), input_dict["y"].cpu().numpy(), xs_len=input_dict["hs_len"])
        ctc_corr_on = calculate_cer_ctc(input_dict["ctc_out_on"].cpu().numpy(), input_dict["y"].cpu().numpy(), xs_len=input_dict["hs_len"])
        valid_dict["ctc_cer_off"] = ctc_corr_off
        valid_dict["ctc_cer_on"] = ctc_corr_on
        return valid_dict
