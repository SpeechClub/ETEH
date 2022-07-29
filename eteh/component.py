from __future__ import print_function


class EtehComponent(object):
    def __init__(self):
        self.distributed = False
        self.parallel = False
        self.rank_id = -1
        self.rank_size = 0
        pass

    def to_distributed(self, nodes, node_id, gpus, gpu):
        raise NotImplementedError("")

    def to_parallel(self, cuda_id, cuda_groups):
        raise NotImplementedError("")

