import torch
import torch.nn as nn

from .lm_base import LM_Base
from eteh.modules.pytorch_backend.net.rnn.lstm import RNNCellStack
from eteh.modules.pytorch_backend.net.transformer.encoder import Encoder as TransformerEncoder
from eteh.utils.data_utils import to_device
from eteh.utils.mask import target_mask


class RNNLM(LM_Base):
    def __init__(self, n_vocab, n_layers, n_units, typ="lstm"):
        torch.nn.Module.__init__(self)
        self.predictor = RNNCellStack(n_vocab,n_vocab,n_layers,n_units,typ,input_layer="embed")


class SelfAttLM(LM_Base):
    def __init__(self, n_vocab, n_layers, attention_dim, attention_heads, linear_units, 
                    dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0):
        torch.nn.Module.__init__(self)
        self.predictor = TransformerEncoder(n_vocab,
                 attention_dim=attention_dim,
                 attention_heads=attention_heads,
                 linear_units=linear_units,
                 num_blocks=n_layers,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="embed",
                 normalize_before=True,
                 concat_after=False)
        self.linear = torch.nn.Sequential(
                    torch.nn.Dropout(dropout_rate),
                    torch.nn.Linear(attention_dim,n_vocab)
                )

    def forward(self,x,state=None): 
        if x.dim() <= 1:
            x = x.unsqueeze(1)
            x_len = 1
            unsq = True
        else:
            x_len = x.size(1)
            unsq = False
        x = to_device(self,x)        
        if state is not None:
            x = torch.cat([state,x],dim=1)
        xs_mask = target_mask(x)
        y, _ = self.predictor(x, xs_mask)
        y = self.linear(y)[:,-x_len:]
        if unsq:
            y = y.squeeze(1)
        return x, y