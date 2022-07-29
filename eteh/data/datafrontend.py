# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import numpy as np
from eteh.utils.dynamic_import import dynamic_import
from eteh.ops.transform.transformation import Transformation


def GetFrontEnd(trans_config={}, frontend=None):
    '''
        :param dict trans_config: trans_config是一个字典，必须包含name和kwargs两个元素
            trans_config[name]是BasicFrontend子类的路径，使用generatemodule获取对应的类
            trans_config[kwargs]是构造BasicFrontend子类所需要的参数
        :param str frontend: eteh.data.datafrontend:BasicFrontend
    '''
    if frontend is None:
        return BasicFrontend(trans_config)
    else:
        cls = dynamic_import(frontend)
        return cls(trans_config)

class BasicFrontend(object):
    def __init__(self, trans_config={}):
        if trans_config is not None and len(trans_config) > 0:
            trans_config = {"process": trans_config}
            self.preprocessing = Transformation(trans_config)
        else:
            # If conf doesn't exist, this function don't touch anything.
            self.preprocessing = None

    def transfer_data(self, data, data_len):
        if self.preprocessing is not None:
            data = self.preprocessing(data, train=True)
        return data, data_len

    def transfer_label(self, label, label_len):
        return label, label_len

class MultiChannelFrontend(BasicFrontend):
    #同时返回前端处理之前和处理之后的数据，因此输出增加了一个维度
    def __init__(self, trans_config={}):
        super(MultiChannelFrontend, self).__init__(trans_config)

    def transfer_data(self, data, data_len):
        data_aug, len_aug = super(MultiChannelFrontend, self).transfer_data(data, data_len)
        # data = np.concatenate((data, data_aug), axis=0)
        # data_len = np.concatenate((data_len, len_aug), axis=0)        
        data = np.stack((data, data_aug), axis=0)
        data_len = np.stack((data_len, len_aug), axis=0)
        return data, data_len

    def transfer_label(self, label, label_len):
        # label = np.concatenate((label, label))
        # label_len = np.concatenate((label_len, label_len))
        label = np.stack((label, label))
        label_len = np.stack((label_len, label_len))
        return label, label_len

class KaldiFrontend(object):
    def __init__(self, trans_config={}):
        self.trans_config = trans_config
        self.trans_preprocessing = {}
        for k in trans_config:
            self.trans_preprocessing[k] = Transformation({"process": trans_config[k]})

    def transfer_kaldi(self, data):
        for k in self.trans_preprocessing:
            if k in data:
                data[k]["data"] = self.trans_preprocessing[k](data[k]["data"], train=True)
                # 如何解决长度问题呢？？咩咩咩。 让trans_preprocessing返回修改之后返回新的长度
                # 如何提取特征呢？？
        return data

    def transfer_Data_by_key(self, data, key):
        if key not in self.trans_preprocessing:
            return data
        else:
            value = self.trans_preprocessing[key]([data.getvalue()], train=True)
            data.setValue(value[0])
            return data

    def transfer_Label_by_key(self, label, key):
        return label