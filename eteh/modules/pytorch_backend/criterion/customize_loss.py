import torch
from eteh.utils.mask import make_pad_mask
 
class KL_Loss(torch.nn.Module):
    def __init__(self, size, normalize_length=False, criterion=torch.nn.KLDivLoss(reduce=False)):
        super(KL_Loss, self).__init__()
        self.criterion = criterion
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, y, mask):
        assert x.size() == y.size(), "{} {}".format(x.size(), y.size())
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        with torch.no_grad():
            y = torch.softmax(y.view(-1, self.size), dim=1).detach().to(y.device)
        kl = self.criterion(torch.log_softmax(x, dim=1), y)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(mask.view(-1, 1), 0).sum() / denom