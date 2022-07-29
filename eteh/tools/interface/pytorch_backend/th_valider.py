
# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import torch
from eteh.tools.valider import Valider

class TH_Valider(Valider):
    def __init__(self,dataloader,model, criterion):
        super(TH_Valider, self).__init__(dataloader,model, criterion)
        