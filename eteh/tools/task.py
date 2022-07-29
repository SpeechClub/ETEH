# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

from eteh.data.datacenter import DataCenter
from eteh.component import EtehComponent
from eteh.tools.config import ModelConfig, CriterionConfig, OptimConfig
from eteh.tools.observer import ReporterObserver


class EtehTask(EtehComponent):
    def __init__(self, name, cuda_id, exp_path,
                 model_config, data_config, set_config,
                 criterion_config, optim_config, train_config, valid_config=None,
                 other_config=None, random_seed=0, use_npu=False):
        super(EtehTask, self).__init__()
        self.task_name = name
        self.cuda_id = cuda_id
        self.rand_seed = random_seed
        self.use_npu = use_npu
        self.exp_path = exp_path
        # config member
        self.model_config = ModelConfig(model_config)
        self.criterion_config = CriterionConfig(criterion_config)
        self.optim_config = OptimConfig(optim_config)
        self.data_config = data_config
        self.train_config = train_config
        self.set_config = set_config
        self.other_config = other_config
        self.valid_config = valid_config
        # component member
        self.model = None
        self.loss = None
        self.optimizer = None
        self.trainer = None
        self.valider = None
        self.data_center = None
        # observer
        self.observer = None
        # task train member
        self.train_set = None
        self.valid_set = None
        self.epoch = -1
        self.data_num = 0
        self.set_num = 0
        self.step = 0
        self.lr = 0
        self.best_model = (0, 1e10)
        # DataLoader config
        self.b_size = self.set_config.pop(
            "b_size") if "b_size" in self.set_config else 1
        self.data_loader_threads = self.set_config.pop(
            "data_loader_threads") if "data_loader_threads" in self.set_config else 1

    def task_init(self):
        self.generate_model()
        self.generate_optimizer()
        self.generate_criterion()
        self.generate_trainer()
        self.generate_datacenter()
        self.generate_valider()
        self.generate_observer()

    def generate_model(self, model_config=None):
        if(model_config is not None):
            self.model_config = ModelConfig(model_config)
        self.model = self.model_config.generateExample()
        if(self.cuda_id >= 0):
            self.model.cuda(self.cuda_id)

    def generate_datacenter(self, data_config=None):
        if(data_config is not None):
            self.data_config = data_config
        self.data_center = DataCenter(
            self.data_config, distributed=self.distributed, rank_size=self.rank_size, rank_id=self.rank_id, rand_seed=self.rand_seed)

    def generate_optimizer(self, optim_config=None):
        if(optim_config is not None):
            self.optim_config = OptimConfig(optim_config)
        self.optimizer = self.optim_config.generateExample(self.model)

    def generate_criterion(self, criterion_config=None):
        if(criterion_config is not None):
            self.criterion_config = CriterionConfig(criterion_config)
        self.loss = self.criterion_config.generateExample()

    def generate_trainer(self, train_config=None):
        raise NotImplementedError("")

    def generate_valider(self, valid_config=None):
        if valid_config is not None:
            self.valid_config = valid_config
        self.valid_set = self.data_center.GetUnDistributedDataLoader(1, 1, source="valid", data_config = self.valid_config)

    def generate_observer(self, observer_conf=None):
        self.observer = ReporterObserver(self.exp_path)

    def is_finish(self):
        model_ok = self.model is not None
        loss_ok = self.loss is not None
        opt_ok = self.optimizer is not None
        train_ok = self.trainer is not None
        data_ok = self.data_center is not None
        return model_ok and loss_ok and opt_ok and train_ok and data_ok

    def train_epoch(self, epoch=0, split=False, multistream=False):
        if split:
            self.train_epoch_split(epoch)
        else:
            self.train_epoch_all(epoch)

    def train_epoch_split(self, epoch=0):
        self.data_center.refresh(epoch)
        self.epoch = epoch
        train_iter = self.data_center.GetDataLoaderIter(
            b_size=self.b_size, data_loader_threads=self.data_loader_threads, data_config=self.set_config)
        for j, train_dataloader in enumerate(train_iter, 0):
            self.set_num, self.data_num = j, len(train_dataloader)
            if self.observer is not None:
                self.observer.notify_station(self)
                self.observer.clear()
            for i, data in enumerate(train_dataloader, 0):
                self.train_step(data)

    def train_epoch_all(self, epoch=0):
        self.data_center.refresh(epoch)
        self.epoch = epoch
        if self.train_set == None:
            self.train_set = self.data_center.GetDataSet(data_config=self.set_config)
        train_dataloader = self.data_center.MakeLoaderFromSet(
            self.train_set, self.b_size, self.data_loader_threads)
        self.data_num = len(train_dataloader)
        if self.observer is not None:
            self.observer.notify_station(self)
            self.observer.clear()
        for i, data in enumerate(train_dataloader, 0):
            self.train_step(data)

    def train_step(self, data):
        data = self.pack_data(data)
        result = self.trainer.train_batch(data)        
        self.lr = self.trainer.get_lr()
        self.step += 1
        result = self.pack_result(result)
        if self.observer is not None:
            self.observer.notify(result)

    def do_valid(self):
        if self.valid_set is None:
            return
        else:
            # v_msg  = self.valider.valid()
            self.model.eval()
            results = []
            for data in self.valid_set:
                data = self.pack_data(data)
                results.append(self.pack_valid_result(
                    self.trainer.do_valid(data)))
            v_msg = self.valid_msg(results)
            if self.observer is not None:
                self.observer.notify_string(v_msg)
            else:
                print(v_msg)

    def valid_msg(self, results):
        results_all = {}
        if len(results) <= 0:
            return "no valid resylts"
        for key in results[0].keys():
            results_all[key] = 0
            for i in range(len(results)):
                results_all[key] += results[i][key]
            results_all[key] /= len(results)
        self.best_model = self.best_model if results_all["loss_main"] > self.best_model[1] else (self.epoch, results_all["loss_main"])
        return str(results_all) + "\n" + "Epoch {} ".format(self.epoch) + "Current Loss:{} ".format(results_all["loss_main"]) + \
            "Valid Best Model: checkpoint.{}".format(self.best_model[0]) + " Best Loss:{}".format(self.best_model[1])

    def save_checkpoint(self, ck_name=""):
        print("Not Implement save function")

    def load_checkpoint(self, ck_name="", resume_optimizer=False, resume_progress=False):
        print("Not Implement save function")

    def pack_data(self, data):
        return data

    def pack_result(self, result):
        report_result = {}
        report_result["lr"] = self.lr
        report_result["loss_main"] = result["loss_main"].item()
        for k in result:
            if type(result[k]) in [int, float]:
                report_result[k] = result[k]        
        return report_result

    def pack_valid_result(self, results):
        return self.pack_result(results)


class EtehTaskDecorator(EtehTask):
    def __init__(self, eteh_task):
        '''
            FIXME: the EtehTaskDecorator is not complate as it don't has the parameters.
        '''
        self.eteh_task = eteh_task

    def task_init(self):
        self.eteh_task.task_init()

    def generate_model(self, model_config=None):
        self.eteh_task.generate_model(model_config)

    def generate_datacenter(self, data_config=None):
        self.eteh_task.generate_datacenter(data_config)

    def generate_optimizer(self, optim_config=None):
        self.eteh_task.generate_optimizer(optim_config)

    def generate_criterion(self, criterion_config=None):
        self.eteh_task.generate_criterion(criterion_config)

    def generate_trainer(self, train_config=None):
        self.eteh_task.generate_trainer(train_config)

    def generate_valider(self, valid_config=None):
        self.eteh_task.generate_valider(valid_config)

    def is_finish(self):
        return self.eteh_task.is_finish()

    def train_epoch(self, epoch=0, split=False, multistream=False):
        self.eteh_task.train_epoch(epoch, split, multistream)

    def train_epoch_split(self, epoch=0):
        self.eteh_task.train_epoch_split(epoch)

    def train_epoch_all(self, epoch=0):
        self.eteh_task.train_epoch_all(epoch)

    def train_step(self, data):
        self.eteh_task.train_step(data)

    def do_valid(self):
        self.eteh_task.do_valid()

    def save_checkpoint(self, ck_name=""):
        self.eteh_task.save_checkpoint(ck_name)

    def load_checkpoint(self, ck_name="", resume_optimizer=False, resume_progress=False):
        self.eteh_task.load_checkpoint(
            ck_name, resume_optimizer, resume_progress)

    def pack_data(self, data):
        return self.eteh_task.pack_data(data)

    def pack_result(self, result):
        return self.eteh_task.pack_result(result)
