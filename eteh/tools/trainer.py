# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import torch
import math
from eteh.component import EtehComponent

class Trainer(EtehComponent):
    '''Trainer is inhert from torch.nn.Module. It can represent a deep neural network training task. For one 
        machine task, three components should be defined, including model, criterion and optimizer.For this
        Trainer class, model and criterion should be a torch.nn.Module and optimizer should be a torch.nn.optimizer
        Call train_batch funtion to train one step according to the data offered
        For a specific training task, you'd better override this class

    :param torch.nn.Module model: the model to be trained. Examples in models
    :param torch.nn.Module criterion: the objective function of the training. Examples in models.criterion
    :param torch.nn.optimizer optimizer: the optimizer for the training. Examples in models.optimizer
    :param bool distributed: if True, the optimizer will call synchronize() at each step, guarantee the optimizer
                                is a instance of hvd.DistributedOptimizer
    '''

    def __init__(self, model, criterion, optimizer, acc_grad=1, max_grad_norm=5):
        super(Trainer, self).__init__()
        self.step = 0
        self.model = model
        self.loss_fn = criterion
        self.optimizer = optimizer
        self.acc_grad = acc_grad
        self.max_grad_norm = max_grad_norm
        self.valid = None

    def forward(self, data, *arg, **args):
        self.model.train()
        return self.model(data, *arg, **args)

    def train_batch(self, input_data):
        data = {}
        data.update(input_data)
        model_out = self.ModelForward(input_data)
        data.update(model_out)
        loss_out = self.LossCompute(data)
        data.update(loss_out)
        assert "loss_main" in loss_out
        self.TrainerUpdata()
        self.LossBackward(loss_out["loss_main"] / self.acc_grad)
        return data

    def do_valid(self, input_data):
        data = {}
        data.update(input_data)
        model_out = self.ValidForward(input_data)
        data.update(model_out)
        loss_out = self.LossCompute(data, valid=True)
        data.update(loss_out)
        return data

    def ModelForward(self, data):
        self.model.train()
        return self.model(data)

    def ValidForward(self, data):
        self.model.eval()
        return self.model(data, valid=True)

    def LossCompute(self, data, valid=False):
        return self.loss_fn(data, valid=valid) 

    def LossBackward(self, loss):
        self._backward(loss, self.max_grad_norm)

    def TrainerUpdata(self):
        self.step += 1
        self.up_data_optim()

    def get_lr(self):
        return 0

    def _backward(self, loss, max_grad_norm):
        '''
            you can override this method to realize different trainer
        '''
        raise NotImplementedError("")

    def up_data_optim(self):
        if hasattr(self.optimizer, 'set_step'):
            self.optimizer.set_step(self.step, self.acc_grad)
