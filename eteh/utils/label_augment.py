import random
import torch

def tensor_labaug(label_in,label_len,charnum,replace=0.15, swap=0.1):
    assert label_in.dim()>=2
    o_label = torch.zeros_like(label_in)
    for i in range(label_in.size(0)):
        o_label[i] = label_aug(label_in[i],label_len[i],charnum,replace=replace, swap=swap)
    return o_label.detach()


def label_aug(in_label,lens,charnum,replace=0.15, swap=0.1):
    label = in_label.detach().squeeze().clone()
    mask = label.new_empty(label.size())
    mask = mask.bernoulli_(replace).byte()   
    mask[lens-1:] = 0 
    fill_data = torch.rand_like(label[mask==1].float())
    for i in range(fill_data.size(0)):
        fill_data[i] = random.randint(1,charnum-1)
    label[mask==1] = fill_data.long()
    
    for i in range(lens):
        if random.uniform(0,1) < swap:
            if i>1:
                pos = random.randint(0,i)
                temp = label[i].item()
                label[i] = label[pos].item()
                label[pos] = temp
                
    return label.detach()


if __name__ == "__main__":
    x = torch.LongTensor([[1,4,5,67,5,1,3,7,-1,-1,-1],[4,4,5,54,5,13,3,7,4,2,1]])
    print(x.size())
    xlen = [8,11]
    xaug = tensor_labaug(x,xlen,100,replace=0,swap=0.3)
    print(x)
    print(xaug)
