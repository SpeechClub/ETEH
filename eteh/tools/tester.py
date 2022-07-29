# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import torch

def MaxDecider(data):
    return torch.argmax(data,dim=-1).int()

def SameChecker(pre,label):
    return pre == label


class Tester(torch.nn.Module):
    def __init__(self,model,decider=None,checker=SameChecker):
        super(Tester, self).__init__()
        self.model = model
        self.decider = decider
        self.checker = checker

    def forward(self,data,*arg,**args):
        self.model.eval()
        if hasattr(self.model, 'recognize'):
            out = self.model.recognize(data,*arg,**args)
        else:
            out = self.model.forward(data,*arg,**args)
        
        return out

    def recognize(self,data,*arg,**args):
        out = self.forward(data,*arg,**args)
        if self.decider is not None:
            out = self.decider(out)
        return out

    def accurate(self,data,label,*arg,**args):
        predict = self.recognize(data,*arg,**args)
        dev = self.checker(predict,label)
        return torch.mean(dev.view(-1).float())
