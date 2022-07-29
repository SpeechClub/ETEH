import torch
import numpy as np

class SeqCrossEntorpy(torch.nn.CrossEntropyLoss):
    '''
        x:[B,T,D],y:[B,T,1]
    '''
    def forward(self,x,y):
        #super(self,x.view(-1, x.shape[2]), y.view(-1))
        x = x.contiguous()
        y = y.contiguous()
        return super(SeqCrossEntorpy, self).forward(x.view(-1, x.shape[2]), y.view(-1))


class CTC_Loss(torch.nn.Module):
    def __init__(self, ctc_type='builtin', ignore_id=-1, reduce=True):
        super(CTC_Loss, self).__init__()
        self.loss = None
        self.ctc_type = ctc_type
        self.ignore_id = ignore_id
        if self.ctc_type == 'builtin':
            reduction_type = 'sum' if reduce else 'none'
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == 'warpctc':
            import warpctc_pytorch as warp_ctc
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)
        else:
            raise ValueError('ctc_type must be "builtin" or "warpctc": {}'
                             .format(self.ctc_type))

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        if self.ctc_type == 'builtin':
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            # Batch-size average
            loss = loss / th_pred.size(1)
            return loss
        elif self.ctc_type == 'warpctc':
            return self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        else:
            raise NotImplementedError

    def forward(self, out_pad, out_len, label_pad):
        ys_hat = out_pad.transpose(0, 1)
        hlens = torch.from_numpy(np.fromiter(out_len, dtype=np.int32))

        ys = [y[y != self.ignore_id] for y in label_pad]
        ys_true = torch.cat(ys).cpu().int()        
        olens = torch.from_numpy(np.fromiter(
            (x.size(0) for x in ys), dtype=np.int32))
        with torch.backends.cudnn.flags(enabled=False):
            self.loss = self.loss_fn(ys_hat, ys_true, hlens, olens).to(out_pad.device)
        return self.loss

class LabelSmoothingLoss(torch.nn.Module):
    """Label-smoothing loss

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, padding_idx, smoothing, normalize_length=False, criterion=torch.nn.KLDivLoss(reduce=False)):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0).to(x.device)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        ignore = ignore.to(x.device)
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom