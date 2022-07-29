import torch


'''
def GetOptimeter(model,*args,opt_type='adam',**kwargs):
    if opt_type == 'adadelta':
        opti = torch.optim.Adadelta(
            model.parameters(),*args,**kwargs)
    elif opt_type == 'adam':
        opti = torch.optim.Adam(model.parameters(),
                                     *args,**kwargs)
    elif opt_type == 'noam':
        opti = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    elif opt_type == 'sgd':    
        opti = torch.optim.SGD(model.parameters(), *args,**kwargs)
    else:
        raise NotImplementedError("unknown optimizer: " + opt_type)
    return opti
'''


class Noam(torch.optim.Adam):
    """Optim wrapper that implements rate."""

    def __init__(self, params, model_size, factor, warm_step, offset=0, offstep=0):
        """Construct an NoamOpt object."""
        # self.optimizer = torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9)
        super(Noam, self).__init__(params, lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warm_step
        self.factor = factor
        self.model_size = model_size
        self.offset = offset
        self.offstep= offstep
        self._rate = 0

    # @property
    # def param_groups(self):
    #     """Return param_groups."""
    #     return self.optimizer.param_groups

    def set_step(self, step, acc_grad):
        self._step = step // acc_grad

    def step(self):
        """Update parameters and rate."""
        # self._step += 1 # 使用trainer控制step，不要自己记录自己的学习率
        rate = self.rate()
        for p in self.param_groups:
            p["lr"] = rate
        self._rate = rate
        super(Noam, self).step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        step += self.offstep
        return (
            self.offset + self.factor
            * self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    # def zero_grad(self):
    #     """Reset gradient."""
    #     self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": super(Noam, self).state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                super(Noam, self).load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)