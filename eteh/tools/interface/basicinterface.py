# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from eteh.component import EtehComponent

class EtehInterface(EtehComponent):
    def __init__(self, back_ground):
        super(EtehInterface, self).__init__()
        self.back_ground = back_ground