import torch
from eteh.tools.trainer import Trainer
from eteh.tools.interface.pytorch_backend.th_model import TH_TrainModelFace

class TH_Trainer(Trainer):
    def __init__(self, model, criterion, optimizer, acc_grad=1, amp="None"):
        super(TH_Trainer, self).__init__(
            model, criterion, optimizer, acc_grad)
        self.amp = amp == "torch.amp"
        self.model.amp = self.amp
        self.loss_fn.amp = self.amp
        self.deep_speed = amp == "deepspeed" 
        self.apex = amp in ["apex:O1", "apex:O2", "apex:O3"]
        if self.amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.train_model = TH_TrainModelFace(self.model, self.loss_fn)
        if self.apex:
            self.apex_level = amp.split(":")[1]
            from apex import amp
            self.apex = amp


    def train_batch(self, input_data):
        self.train_model.train()
        data = self.train_model(input_data)
        assert "loss_main" in data
        self.TrainerUpdata()
        self.LossBackward(data["loss_main"] / self.acc_grad)
        return data

    def do_valid(self, input_data):
        self.train_model.eval()
        data = self.train_model(input_data, valid=True)
        return data

    def _backward(self, loss, max_grad_norm):
        '''
            you can override this method to realize different trainer
        '''
        self.up_data_optim()
        # Gradient Clipping
        import math
        if self.deep_speed:
             self.train_model.backward(loss)
             self.train_model.step()
        elif self.apex:
            with self.apex.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm)
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                self.optimizer.zero_grad()
                return
            if self.step%self.acc_grad==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        elif self.scaler is None:            
            loss.backward()
            if max_grad_norm != 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm)
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    self.optimizer.zero_grad()
                    return
            if self.step%self.acc_grad==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place     
            if self.step%self.acc_grad==0:
            #acc grad can not be used when amp is used
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                if max_grad_norm != 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print(RuntimeWarning("grad with NAN !"))       
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def to_distributed(self, nodes, node_id, gpus, gpu, use_npu=False):
        if use_npu:
            loc = 'npu:{}'.format(gpu)
            torch.npu.set_device(loc)
            self.train_model.to(gpu)
        else:
            torch.cuda.set_device(node_id * gpus + gpu)
            self.train_model.cuda()
        if self.apex:
            from apex.parallel import DistributedDataParallel as DDP
            self.train_model.model, self.optimizer=self.apex.initialize(self.train_model.model, self.optimizer, opt_level=self.apex_level)# 这里是“欧一”，不是“零一”
            # self.train_model = torch.nn.parallel.DistributedDataParallel(self.train_model, device_ids=[
                             # gpu], find_unused_parameters=True, output_device=node_id*gpus + gpu)            
            self.train_model = DDP(self.train_model)
        else:   
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.train_model = DDP(self.train_model, device_ids=[
                                         gpu], find_unused_parameters=True)
  

    def to_parallel(self, cuda_id, cuda_groups):
        torch.cuda.set_device(cuda_id)
        self.train_model.cuda(cuda_id) 
        if self.apex:
            self.train_model.model, self.optimizer=self.apex.initialize(self.train_model.model, self.optimizer, opt_level=self.apex_level)# 这里是“欧一”，不是“零一”
        self.train_model = torch.nn.DataParallel(
            self.train_model, device_ids=cuda_groups)

 
    def to_deepspeed(self, args):
        import deepspeed
        self.deep_speed = True
        self.train_model.cuda()
        model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=self.train_model,
                                                     model_parameters=self.train_model.parameters())
        self.train_model, self.optimizer = model_engine, optimizer