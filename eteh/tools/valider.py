# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import torch
import numpy as np
from eteh.component import EtehComponent


class Valider(EtehComponent):
    def __init__(self, dataloader, model, criterion=None):
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion

    def valid(self, *args, **kwargs):
        self.model.eval()
        result = []
        for i, data in enumerate(self.dataloader, 0):
            data = self.pack_valid_data(data)
            result.append(self.valid_function(
                data, self.model, *args, **kwargs))
        return self.pack_valid_result(result)

    def get_validdata_iter(self):
        for i, data in enumerate(self.dataloader, 0):
            yield data        

    def pack_valid_data(self, data):
        return data

    def valid_function(self, data=None, model=None):
        raise NotImplementedError("")

    def pack_valid_result(self, result):
        raise NotImplementedError("")
