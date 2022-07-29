import torch
from eteh.models.model_interface import Model_Interface

class LM_Base(torch.nn.Module, Model_Interface):
    def __init__(self, predictor):
        torch.nn.Module.__init__(self)
        self.predictor = predictor

    def forward(self,x,state=None):
        if x.dim() <=1:
            state,y = self.predictor(state, x)
        else:
            x_len = x.size(1)
            y_list = []
            for i in range(x_len):
                state,y = self.predictor(state, x[:,i])
                y_list.append(y.unsqueeze(1))
            y = torch.cat(y_list,dim=1)
        return state, y

    def train_forward(self, input_dict):
        state, y = self.forward(
            x = input_dict["y_in"],
            state = None,
        )

        return {
            "lm_out": y,
            "state_new": state
        }

    def get_input_dict(self):
        return {"y_in": "(B) or (B,T)"}

    def get_out_dict(self):
        return {"lm_out":"(B) or (B,T)", "state_new": "new state"}