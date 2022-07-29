import torch
from eteh.tools.interface.basicinterface import EtehInterface
from eteh.models.model_interface import Model_Interface

class TH_ModelFace(torch.nn.Module, EtehInterface):

    def __init__(self, model, amp=False):
        torch.nn.Module.__init__(self)
        EtehInterface.__init__(self, 'pytorch')
        self.amp = amp
        self.model = model
        assert isinstance(model, Model_Interface)
        assert isinstance(model, torch.nn.Module)

    def forward(self, *args, valid=False, **kwargs):
        if valid:
            return self.valid_forward(*args, **kwargs)
        if(self.amp):
            from torch.cuda.amp import autocast
            with autocast():
                output = self.model.train_forward(*args, **kwargs)
        else:
            output = self.model.train_forward(*args, **kwargs)
        return output

    def valid_forward(self, *args, **kwargs):
        with torch.no_grad():
            output = self.model.valid_forward(*args, **kwargs)
        return output

class TH_CriterionFace(torch.nn.Module, EtehInterface):
    def __init__(self, criterion, amp=False):
        torch.nn.Module.__init__(self)
        EtehInterface.__init__(self, 'pytorch')
        self.amp = amp
        self.criterion = criterion
        assert isinstance(criterion, Model_Interface)
        assert isinstance(criterion, torch.nn.Module)


    def forward(self, *args, valid=False, **kwargs):
        if valid:
            return self.valid_forward(*args, **kwargs)
        if(self.amp):
            from torch.cuda.amp import autocast
            with autocast():
                output = self.criterion.train_forward(*args, **kwargs)
        else:
            output = self.criterion.train_forward(*args, **kwargs)
        return output

    def valid_forward(self, *args, **kwargs):
        with torch.no_grad():
            output = self.criterion.valid_forward(*args, **kwargs)
        return output

class TH_TrainModelFace(torch.nn.Module, EtehInterface):
    '''
        由于很多地方都按照model forward得到损失函数的设置，因此增加了一个train model，
        负责将模型和损失函数拼接起来
    '''
    def __init__(self, model, criterion):
        torch.nn.Module.__init__(self)
        EtehInterface.__init__(self, 'pytorch')
        self.model = model
        self.criterion = criterion
        assert isinstance(criterion, TH_CriterionFace)
        assert isinstance(model, TH_ModelFace)

    def forward(self, input_data, valid=False):
        data = {}
        data.update(input_data)
        model_out = self.model(input_data, valid=valid)
        data.update(model_out)
        loss_out = self.criterion(data, valid=valid)
        data.update(loss_out)
        return data