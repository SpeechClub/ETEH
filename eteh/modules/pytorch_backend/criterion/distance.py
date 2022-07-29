import torch
import numpy as np 

class SeqCosineSimilarity(torch.nn.Module):
    '''
        x:[B,T,D],y:[B,T,D]
    '''
    def forward(self,feature1,feature2):
        assert feature1.dim() == feature2.dim()
        assert feature1.dim()>=2
        loss_cosine = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")
        loss = torch.zeros(1).to(feature1.device)
        feature1 = feature1.reshape(-1, feature1.size()[2])
        feature2 = feature2.reshape(-1, feature2.size()[2])
        y = torch.ones(feature2.shape[0]).to(feature1.device)
        loss = loss_cosine(feature2, feature1, y)
        return loss

class SeqPairwiseDistance(torch.nn.PairwiseDistance):
    '''
        x:[B,T,D],y:[B,T,D]
    '''
    def forward(self,x,y):
        #super(self,x.view(-1, x.shape[2]), y.view(-1))
        assert x.dim() == y.dim()
        assert x.dim()>=2
        shape = x.size()
        x = x.contiguous().view(-1, x.shape[-1])
        y = y.contiguous().view(-1, x.shape[-1])
        dist = super(SeqPairwiseDistance, self).forward(x, y)
        return torch.mean(dist)

class SeqKLDistance(torch.nn.KLDivLoss):

    def __init__(self,reduction ='batchmean'):
        super(SeqKLDistance,self).__init__(reduction = reduction)
    '''
        x:[B,T,D],y:[B,T,D]
        x, y must be the real pdf not the log pdf
    '''
    def forward(self,x,y):
        #super(self,x.view(-1, x.shape[2]), y.view(-1))
        assert x.dim() == y.dim()
        assert x.dim()>=2
        logx = torch.log(x)
        logy = torch.log(y)
        x,logx = x.contiguous(),logx.contiguous()
        y,logy = y.contiguous(),logy.contiguous()
        dist1 = super(SeqKLDistance, self).forward(logx.view(-1, x.shape[2]).t(), y.view(-1, x.shape[2]).t())
        dist2 = super(SeqKLDistance, self).forward(logy.view(-1, x.shape[2]).t(), x.view(-1, x.shape[2]).t())
        return (dist1+dist2)/2

class SeqCEDistance(torch.nn.Module):

    def __init__(self,reduction='mean'):
        super(SeqCEDistance,self).__init__()
        self.reduction = reduction

    def forward(self,x,y):
        assert x.dim() == y.dim()
        assert x.dim()>=2
        logx = torch.log(x)
        logx,y = logx.contiguous(),y.contiguous()
        ce = -torch.sum(y*logx,-1).view(-1)
        if self.reduction=="mean":
            return torch.mean(ce,-1)
        elif self.reduction=="sum":
            return torch.sum(ce,-1)
        else:
            return ce

if __name__ == "__main__":
    a = torch.rand(3,2,5)
    den = a.sum(dim=-1)[0].unsqueeze(-1)
    k1 = a/den
    log1 = torch.log(k1)
    a = torch.rand(3,2,5)
    den = a.sum(dim=-1)[0].unsqueeze(-1)
    k2 = a/den
    log2 = torch.log(k2)
    loss = SeqKLDistance("batchmean")
    sim = loss(k1,k2)
    print('diff data:',sim)

    sim = loss(k1,k1)
    print('same data:',sim)

    loss = SeqCEDistance("mean")
    sim = loss(k1,k2)
    print('diff data:',sim)

    sim = loss(k1,k1)
    print('same data:',sim)