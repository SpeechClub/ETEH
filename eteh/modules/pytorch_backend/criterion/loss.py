import torch
from eteh.models.pytorch_backend.criterion.cross_entropy import CTC_Loss, LabelSmoothingLoss
from eteh.models.pytorch_backend.criterion.customize_loss import Align_Loss, KL_Loss
from eteh.utils.mask import make_pad_mask


class E2E_Loss(torch.nn.Module):
    def __init__(self, size, padding_idx, smoothing, rate=0.5, ctc_type='builtin'):
        super(E2E_Loss, self).__init__()
        self.ctc_loss = CTC_Loss(ctc_type=ctc_type)
        self.att_loss = LabelSmoothingLoss(size, padding_idx, smoothing, False)
        self.rate = rate


    def forward(self, att_out, ctc_out, data_len, att_label, ctc_label, ctc_len):
        #with autocast(): //增加这个修饰才能正确运行amp
        att_loss = self.att_loss(att_out, att_label)
        ctc_loss = self.ctc_loss(
            ctc_out, ctc_len, ctc_label)
        return (1 - self.rate) * att_loss + self.rate * ctc_loss, att_loss, ctc_loss
        
class CTC_CE_Online_Loss(E2E_Loss):
    def __init__(self, size, padding_idx, smoothing, rate=0.5, ali_rate=1.0, ali_type='mid'):
        super(CTC_CE_Online_Loss, self).__init__(size, padding_idx, smoothing, rate)
        self.ali_rate = ali_rate
        self.ali_loss = Align_Loss(ali_type, padding_idx)

    def forward(self, att_out, ctc_out, ali_out, att_label, ctc_label, ctc_len, label_beg, label_end):
        #with autocast(): //增加这个修饰才能正确运行amp
        att_loss = self.att_loss(att_out, att_label)
        ctc_loss = self.ctc_loss(ctc_out, ctc_len, ctc_label)
        ali_loss = self.ali_loss(ali_out, label_beg, label_end, ctc_out.size(1)) \
            if self.ali_rate > 0 else att_loss.new_zeros(att_loss.size())
        return (1 - self.rate) * att_loss \
               + self.rate * ctc_loss \
               + self.ali_rate * ali_loss, \
               att_loss, ctc_loss, ali_loss

class CTC_CE_Univ_Loss(E2E_Loss):
    def __init__(self, size, padding_idx, smoothing, rate=0.5, ali_rate=1.0, ali_type='mid', kl_rate=1.0):
        super(CTC_CE_Univ_Loss, self).__init__(size, padding_idx, smoothing, rate)
        self.ali_rate = ali_rate
        self.ali_loss = Align_Loss(ali_type, padding_idx)
        self.kl_rate = kl_rate
        self.kl_loss = KL_Loss(size)
        self.padding_idx = padding_idx

    def forward(self, att_out, ctc_out, ali_out, att_out_off, ctc_out_off, data_len, att_label, ctc_label, ctc_len, label_beg, label_end):
        #with autocast(): //增加这个修饰才能正确运行amp
        att_loss = self.att_loss(att_out, att_label)
        att_loss_off = self.att_loss(att_out_off, att_label)
        kl_loss = self.kl_loss(att_out, att_out_off, att_label == self.padding_idx)
        ctc_loss = self.ctc_loss(
            ctc_out, ctc_len, ctc_label)
        ctc_loss_off = self.ctc_loss(
            ctc_out_off, ctc_len, ctc_label)
        ctc_mask = (make_pad_mask(ctc_len.squeeze(-1).tolist(), max_length=ctc_out.size(1))).to(ctc_out.device)
        kl_loss += self.kl_loss(ctc_out, ctc_out_off, ctc_mask)
        ali_loss = self.ali_loss(ali_out, label_beg, label_end, ctc_out.size(1)) \
            if self.ali_rate > 0 else att_loss.new_zeros(att_loss.size())
        return (1 - self.rate) * (att_loss + att_loss_off) \
               + self.rate * (ctc_loss + ctc_loss_off) \
               + self.ali_rate * ali_loss \
               + self.kl_rate * kl_loss, \
               att_loss, ctc_loss, ali_loss, kl_loss
