import torch
import torch.nn.functional as F
from eteh.models.model_interface import Model_Interface

class LM_Loss(torch.nn.Module, Model_Interface):
    def __init__(self, padding_idx=-1):
        super(LM_Loss, self).__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.padding_idx = padding_idx

    def forward(self, y_out, lm_out):        
        sequence_length = y_out.size(1)
        loss, count = 0, 0
        for i in range(sequence_length):
            loss_batch = self.loss_func(lm_out[:,i], y_out[:,i])
            non_zeros = torch.sum(y_out[:, i] != self.padding_idx, dtype=torch.float)
            loss += loss_batch * non_zeros 
            count += int(non_zeros)
        loss = loss.mean()
        ppl = torch.exp(loss/count)
        return loss, ppl

    def train_forward(self, input_dict):
        loss, ppl = self.forward(
            y_out = input_dict["y_out"],
            lm_out = input_dict["lm_out"],
        )
        return {
            "loss_main": loss,
            "ppl": ppl.item(),
        }

    def get_input_dict(self):
        return {"y_out": "(B,T)", "lm_out": "(B,T)"}

    def get_out_dict(self):
        return {"loss_main":"(1)", "ppl": "(1)"}