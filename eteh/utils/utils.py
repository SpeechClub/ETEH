
"""
  AverageMeter and ProgressMeter are borrowed from the PyTorch Imagenet example:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
AVERAGE_VAL = 0x08
AVERAGE_SUM = 0x04
AVERAGE_AVG = 0x02
AVERAGE_COUNT = 0x01
AVERAGE_VAL_AVG = AVERAGE_VAL|AVERAGE_AVG

FMT_E4 = ':.4e'
FMT_F6 = ':.3f'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', mode = AVERAGE_VAL_AVG):
        self.name = name
        self.fmt = fmt
        self.mode = mode
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def getmodevalue(self):
        if self.mode&AVERAGE_VAL:
            return self.val
        if self.mode&AVERAGE_SUM:
            return self.sum
        if self.mode&AVERAGE_AVG:
            return self.avg
        if self.mode&AVERAGE_COUNT:
            return self.count 
        else:
            return 0      

    def __str__(self):
        fmtstr = '{name}'
        if self.mode&AVERAGE_VAL:
            fmtstr += ' {val' + self.fmt + '}'
        if self.mode&AVERAGE_SUM:
            fmtstr += ' [{sum' + self.fmt + '}]'
        if self.mode&AVERAGE_AVG:
            fmtstr += ' ({avg' + self.fmt + '})'
        if self.mode&AVERAGE_COUNT:
            fmtstr += ' _{count}_'
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def updata(self,num_batches=None,prefix=None):
        if num_batches is not None:
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        if prefix is not None:
            self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def filelog(self,batch,filename):
        with open(filename,'a') as f:
            f.write(self.getstr(batch)+"\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def getstr(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)