import os
import copy
from distutils.version import LooseVersion
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from eteh.tools.task import EtehTask, EtehTaskDecorator
from eteh.tools.interface.pytorch_backend.th_model import TH_ModelFace, TH_CriterionFace
from eteh.tools.interface.pytorch_backend.th_trainer import TH_Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp


class TH_Task(EtehTask):
    def __init__(self, name, cuda_id, exp_path,
                 model_config, data_config, set_config,
                 criterion_config, optim_config, train_config, valid_config,
                 other_config, random_seed=0, use_npu=False):
        super(TH_Task, self).__init__(name, cuda_id, exp_path,
                                      model_config, data_config, set_config,
                                      criterion_config, optim_config, train_config, valid_config,
                                      other_config, random_seed, use_npu)

    def to_parallel(self, cuda_id, cuda_groups):
        self.parallel = True
        self.cuda_id = cuda_id
        self.trainer.to_parallel(cuda_id, cuda_groups)


    def to_distributed(self, nodes, node_id, gpus, gpu):
        self.distributed = True
        self.cuda_id = gpu
        self.trainer.to_distributed(nodes, node_id, gpus, gpu, self.use_npu)
        self.data_center.to_distributed(nodes, node_id, gpus, gpu)        
        if self.observer is not None:
            self.observer.to_distributed(nodes, node_id, gpus, gpu)

    def to_deepspeed(self, args):
        import torch.distributed as dist
        self.distributed = True
        self.trainer.to_deepspeed(args)
        self.data_center.distributed = True
        self.data_center.rank_size = dist.get_world_size()
        self.data_center.rank_id = dist.get_rank()
        # self.data_center.to_distributed(args.num_nodes, node_id, gpus, gpu)        

    def generate_model(self, model_config=None):
        super(TH_Task, self).generate_model(model_config)
        self.model = TH_ModelFace(self.model)

    def generate_criterion(self, criterion_config=None):
        super(TH_Task, self).generate_criterion(criterion_config)
        self.loss = TH_CriterionFace(self.loss)

    def generate_trainer(self, train_config=None):
        if(train_config is not None):
            self.train_config = train_config
        self.trainer = TH_Trainer(self.model, self.loss, self.optimizer, 
                        acc_grad=self.train_config['accum_grad'], amp=self.train_config['amp'])

    def save_checkpoint(self, ck_name=""):
        checkpoint = {}
        checkpoint['distributed'] = self.distributed
        checkpoint['epoch'] = self.epoch
        checkpoint['step'] = self.step
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
            torch.save(checkpoint, ck_name,
                       _use_new_zipfile_serialization=False)
        else:
            torch.save(checkpoint, ck_name)

    def load_checkpoint(self, ck_name="", resume_optimizer=False, resume_progress=False):
        if len(ck_name)==0 or ck_name=="None":
            return
        checkpoint = torch.load(
            ck_name, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, False)
            print("missing_keys: {}".format(','.join(missing_keys)))
            print("unexpected_keys: {}".format(','.join(unexpected_keys)))
        else:
            self.model.load_state_dict(state_dict, False)
        if resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if resume_progress:
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            self.trainer.step = self.step
        print("resumed {}".format(ck_name))

    def get_dist_task(self, nodes, gpus, node_id, master_ip, args,
                      master_port='8888', dist_method='nccl', init_method="env://",
                      seed=0, valid_freq=1):
        return TH_Distributed_Task(self, nodes, gpus, node_id, master_ip, args,
                                   master_port=master_port, dist_method=dist_method, init_method=init_method,
                                   seed=seed, valid_freq=valid_freq)

    def get_deepspeed_task(self, args, seed=0, valid_freq=1):
        return TH_DeepSpeed_Task(self, args, seed=seed, valid_freq=valid_freq)

class TH_Distributed_Task(EtehTaskDecorator):
    def __init__(self, th_task, nodes, gpus, node_id, master_ip, args,
                 master_port='8888', dist_method='nccl', init_method="env://",
                 seed=0, valid_freq=1):
        super(TH_Distributed_Task, self).__init__(th_task)
        self.world_size = nodes * gpus
        self.nodes = nodes
        self.gpus = gpus
        self.node_id = node_id
        self.master_ip = master_ip
        self.master_port = master_port
        self.dist_method = dist_method
        self.init_method = init_method
        self.seed = seed
        self.valid_freq = valid_freq
        os.environ['MASTER_ADDR'] = self.master_ip
        os.environ['MASTER_PORT'] = self.master_port

        mp.spawn(self.distributed_train, nprocs=gpus, args=(args, ))

    def distributed_train(self, gpu, args):
        # 虽然看上去是多个rank对一个task进行了修改，但实际上不是，用多线程开启的时候拷贝了多份task
        # print(self.eteh_task)
        rank = self.node_id * self.gpus + gpu
        dist.init_process_group(
            backend=self.dist_method,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=rank
        )
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.eteh_task.task_init()
        self.eteh_task.to_distributed(self.nodes, self.node_id, self.gpus, gpu)
        if args.checkpoint is not None and args.checkpoint != "None" :
            if args.param_only:
                self.load_checkpoint(args.checkpoint, False, False)
                print(
                    'Warning! --param_only will be disgarded, use --resume_optimizer and --resume_progress instead')
            else:
                self.load_checkpoint(
                    args.checkpoint, args.resume_optimizer, args.resume_progress)
        start_epoch = self.eteh_task.epoch + 1
        for i in range(start_epoch, args.num_epochs):
            dist.barrier()
            self.eteh_task.train_epoch(i, args.split, args.multistream)
            if i % self.valid_freq == 0:
                self.eteh_task.do_valid()
            if rank == 0:
                self.save_checkpoint(
                    args.exp_dir+"/checkpoint."+str(i))
        pass

class TH_DeepSpeed_Task(EtehTaskDecorator):

    def __init__(self, th_task, args,
                 seed=0, valid_freq=1):
        super(TH_DeepSpeed_Task, self).__init__(th_task)
        self.eteh_task.to_deepspeed(args)
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def load_checkpoint(self, ck_name="", resume_optimizer=False, resume_progress=False):
        if len(ck_name)==0 or ck_name=="None":
            return
        _, client_sd =  self.eteh_task.trainer.train_model.load_checkpoint(ck_name, 0)
        if not resume_optimizer:
            print("Deepspeed must load the optimizer")
        if resume_progress:
            self.eteh_task.step = client_sd['step']
            self.eteh_task.epoch = client_sd['epoch']
            self.eteh_task.trainer.step = self.step

    def save_checkpoint(self, ck_name=""): 
        client_sd  = {}
        # client_sd ['distributed'] = self.distributed
        client_sd['epoch'] = self.eteh_task.epoch
        client_sd['step'] = self.eteh_task.step
        ckpt_id = 0
        self.eteh_task.trainer.train_model.save_checkpoint(ck_name, ckpt_id, client_state = client_sd)
