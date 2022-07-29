# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import time
from eteh.component import EtehComponent
from eteh.tools.reporter import Reporter
from eteh.utils import utils

class IObserver(EtehComponent):
    def __init__(self, print_freq=100):
        super(IObserver, self).__init__()
        self.time = time.time()
        self.print_time = 0
        self.print_freq = print_freq

    def regist_variables(self, variables):
        self.notify_string("do not need regist")

    def to_distributed(self, nodes, node_id, gpus, gpu):
        self.rank_id = node_id*gpus + gpu

    def notify(self, msg):
        if(self.print_time%self.print_freq==0):
            self.notify_msg(msg)
        self.print_time += 1

    def notify_model(self, model):
        pass
        # self.notify_string(str(model))

    def notify_msg(self, msg):
        raise NotImplementedError("")

    def notify_string(self, s):
        raise NotImplementedError("")

    def notify_station(self, task):
        raise NotImplementedError("")

    def clear(self):
        pass

class StdIOObserver(IObserver):
    def notify_msg(self, msg):
        print("" + str(time.time()-self.time) + ":" + str(msg))

class ReporterObserver(IObserver):
    def __init__(self, outpath, filename='train.log', print_freq=100, distributed=False):
        super(ReporterObserver, self).__init__(print_freq=print_freq)
        self.outpath = outpath
        self.filename = filename
        self.outfile = self.outpath + '/' + "rank{}_".format(self.rank_id) + self.filename
        self.distributed = distributed
        if self.distributed:
            self.reporter = Reporter(log_file=self.outfile)
        else:
            self.reporter = Reporter(log_file=self.outfile)
        self.preifx=""
        self.num_baches=0
        self.global_step = 0
        self.model = None
        self.reporter.register(['Time', 'lr'], [utils.AVERAGE_VAL, utils.AVERAGE_VAL], [':6.3f', ':.4e'])

    def to_distributed(self, nodes, node_id, gpus, gpu):
        super(ReporterObserver,self).to_distributed(nodes, node_id, gpus, gpu)
        self.outfile = self.outpath + '/' + "rank{}_".format(self.rank_id) + self.filename
        self.re_open(self.outfile)

    def re_open(self, path):
        self.reporter.re_open(path)

    def regist_variables(self, variables):
        for v in variables:
            self.reporter.register_key(*v)

    def notify(self, msg):
        assert type(msg) == dict
        self.reporter.updata({"Time": time.time() - self.time})
        self.reporter.updata(msg)
        if(self.print_time%self.print_freq==0):
            self.notify_msg(msg)
        self.print_time += 1
        self.global_step += 1

    def clear(self):
        self.reporter.clear()
        self.print_time = 0
        self.time = time.time()

    def notify_msg(self, msg):
        self.reporter.print(self.print_time, prefix=self.preifx, num_banches=self.num_baches)
        self.reporter.logging(self.print_time, prefix=self.preifx, num_banches=self.num_baches)
        self.reporter.clear()

    def notify_station(self, task):
        self.preifx = "Epoch: [{}] ".format(task.epoch)
        self.num_baches = task.data_num
        self.global_step = task.step

    def notify_string(self, s):
        self.reporter.printstring(self.preifx + s)
        self.reporter.logstring(self.preifx + s)

class TensorBoardObserver(ReporterObserver):
    def __init__(self, outpath, filename='train.log', print_freq=100, distributed=False):
        super(TensorBoardObserver, self).__init__(outpath, filename, print_freq, distributed)
        from tensorboardX import SummaryWriter
        try:
            self.boardwriter = SummaryWriter(outpath+'/tensorboard_{}'.format(self.rank_id))
        except:
            self.boardwriter = None
            pass

    def to_distributed(self, nodes, node_id, gpus, gpu):
        from tensorboardX import SummaryWriter
        super(TensorBoardObserver, self).to_distributed(nodes, node_id, gpus, gpu)  
        if self.boardwriter:
            self.boardwriter.close()
        self.boardwriter =  SummaryWriter(self.outpath+'/tensorboard_{}'.format(self.rank_id))
    
    def notify_msg(self, msg):
        for k in self.reporter.MeterDict:
            self.boardwriter.add_scalar(k, self.reporter.getvalue(k), self.global_step)
        self.boardwriter.flush()
        super(TensorBoardObserver,self).notify_msg(msg)

    def notify_model(self, model):
        pass
        # super(TensorBoardObserver,self).notify_model(model)
        # self.boardwriter.add_graph(model, input_to_model=None, verbose=False)
        # self.boardwriter.flush()