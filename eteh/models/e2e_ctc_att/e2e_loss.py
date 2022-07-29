# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import torch
from eteh.modules.pytorch_backend.criterion.cross_entropy import CTC_Loss, LabelSmoothingLoss
from eteh.models.model_interface import Model_Interface
from eteh.utils.data_utils import calcurate_cer, calculate_cer_ctc

class E2E_Loss(torch.nn.Module, Model_Interface):
    def __init__(self, size, padding_idx, smoothing, rate=0.5, ctc_type='builtin'):
        super(E2E_Loss, self).__init__()
        self.ctc_loss = CTC_Loss(ctc_type=ctc_type)
        self.att_loss = LabelSmoothingLoss(size, padding_idx, smoothing, False)
        self.rate = rate


    def forward(self, att_out, ctc_out, ys_out, y, hs_len):
        #with autocast(): //增加这个修饰才能正确运行amp
        att_loss = self.att_loss(att_out, ys_out)
        ctc_loss = self.ctc_loss(
            ctc_out, hs_len, y)
        return (1 - self.rate) * att_loss + self.rate * ctc_loss, att_loss, ctc_loss

    def get_input_dict(self):
        raise NotImplementedError("")

    def get_out_dict(self):
        raise NotImplementedError("")

    def train_forward(self, input_dict):
        loss_main, att_loss, ctc_loss = self.forward(
            att_out=input_dict["att_out"],
            ctc_out=input_dict["ctc_out"],
            ys_out=input_dict["ys_out"],
            y=input_dict["y"],
            hs_len=input_dict["hs_len"],

        )
        att_corr = calcurate_cer(input_dict["att_out"].detach().cpu().numpy(), input_dict["ys_out"].detach().cpu().numpy())            
        
        return {
            "loss_main": loss_main,
            "att_loss": att_loss.item(),
            "ctc_loss": ctc_loss.item(),
            "att_corr": att_corr
        }


    def valid_forward(self, input_dict):
        valdi_dict = self.train_forward(input_dict)
        ctc_corr = calculate_cer_ctc(input_dict["ctc_out"].cpu().numpy(), input_dict["y"].cpu().numpy(), xs_len=input_dict["hs_len"])
        valdi_dict["ctc_cer"] = ctc_corr
        return valdi_dict