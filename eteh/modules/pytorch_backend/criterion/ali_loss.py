import torch
import numpy as np
import math
from eteh.utils.mask import make_pad_mask

def ctc_force_align(logits, target, input_len, target_len):
    bsz, vocab_size, times = logits.size()
    max_target_len = len(target[0])
    logits = logits.detach().cpu().numpy()
    #ctc_alignment = np.zeros((bsz, times), dtype=np.int32)
    align = np.zeros((bsz, max_target_len), dtype=np.float32)
    for b in range(bsz):
        T = input_len[b]
        L = target_len[b]
        N = L * 2 + 1
        labels = [0] * N
        for i in range(L):
            labels[2*i+1] = target[b][i]
        log_probs = logits[b, :, :times]
        a = np.zeros((T, N), dtype=log_probs.dtype)
        v = np.zeros((T, N), dtype=np.int32)
        for i in range(N):
            if i < 2:
                v[0, i] = i
                a[0, i] = log_probs[labels[i], 0]
            else:
                v[0, i] = 0
                a[0, i] = -math.inf
        for t in range(1, T):
            for i in range(N):
                if (i / 2 > t or L - i / 2 > T - t):
                    v[t, i] = 0
                    a[t, i] = -math.inf
                    continue
                elif i == 0:
                    v[t, 0] = v[t-1, 0]
                    a[t, 0] = a[t-1, 0] + log_probs[labels[0], t]
                    continue
                elif i == 1:
                    if a[t-1, 1] > a[t-1, 0]:
                        val = 1
                    else:
                        val = 0
                elif i % 2 == 0:
                    if a[t-1, i] > a[t-1, i-1]:
                        val = i
                    else:
                        val = i - 1
                elif labels[i] == labels[i-2]:
                    if a[t-1, i] > a[t-1, i-1]:
                        val = i
                    else:
                        val = i - 1
                else:
                    if a[t-1, i] > a[t-1, i-1] and a[t-1, i] > a[t-1, i-2]:
                        val = i
                    elif a[t-1, i-1] > a[t-1, i] and a[t-1, i-1] > a[t-1, i-2]:
                        val = i - 1
                    else:
                        val = i - 2
                v[t, i] = val
                a[t, i] = a[t-1, val] + log_probs[labels[i], t]
                
        if a[T-1, N-1] > a[T-1, N-2]:
            pre_idx = N - 1
        else:
            pre_idx = N - 2
            align[b, L - 1] = T
        #ctc_alignment[b, T - 1] = labels[pre_idx]
        for t in range(T - 2, -1, -1):
            pre_idx = v[t+1, pre_idx]
            #ctc_alignment[b, t] = labels[pre_idx]
            if pre_idx % 2:
                align[b, pre_idx//2] = t + 1
        
    return align

class Align_Loss(torch.nn.Module):
    def __init__(self, ali_type='mid', ignore_id=-1, exp_dist=3):
        super(Align_Loss, self).__init__()
        self.ali_type = ali_type
        self.ignore_id = ignore_id
        self.exp_dist = exp_dist
        
    def forward(self, ali_out, ali_beg, ali_end, enc_mask, ctc_out=None, ctc_label=None, ctc_len=None): 
        if self.ali_type == 'google':
            hlen = enc_mask.size(1)
            ylens = torch.sum((ali_beg != self.ignore_id), dim=1) # (batch)
            mask = (make_pad_mask(ylens + 1)).to(ali_out.device).unsqueeze(1).unsqueeze(-1) # (batch, 1, olen + 1, 1)
            mask = mask | enc_mask.unsqueeze(1).unsqueeze(2)
            batch, layers, olen, ilen = ali_out.size()
            align = ali_out.new_zeros(batch, olen, ilen) # (batch, olen + 1, ilen)
            expand_ali_beg = torch.clamp(ali_beg - self.exp_dist - 1, 0, hlen) #(batch, olen)
            expand_ali_end = torch.clamp(ali_end + self.exp_dist, 0, hlen)
            for b in range(batch):
                align[b,:-1,:] = make_pad_mask(expand_ali_beg[b], max_length=hlen) & ~make_pad_mask(expand_ali_end[b], max_length=hlen)
                align[b, ylens[b], expand_ali_beg[b, ylens[b] - 1]:] = 1 # eos
            align = align.unsqueeze(1)
            loss = (ali_out * (1 - align)).masked_fill(mask, 0)
            return loss.sum() / (mask.numel() - mask.sum()) / layers
        elif self.ali_type == 'qua':
            ylens = torch.sum((ali_beg != self.ignore_id), dim=1) + 1 # (batch)
            mask = (make_pad_mask(ylens)).to(ali_out.device).unsqueeze(1).unsqueeze(-1) # (batch, 1, olen + 1, 1)
            batch, layers, olen, ilen = ali_out.size()
            ali_out = ali_out.masked_fill(mask, 0)
            loss = (ylens.unsqueeze(1) - ali_out.sum(dim=[2,3])).sum() / (batch * layers)
            return loss
        elif self.ali_type == 'norm':
            ylens = torch.sum((ali_beg != self.ignore_id), dim=1) + 1 # (batch)
            tokens = ylens.sum().item()
            mask = (make_pad_mask(ylens)).to(ali_out.device).unsqueeze(1).unsqueeze(-1) # (batch, 1, olen + 1, 1)
            batch, layers, olen, ilen = ali_out.size()
            ali_out = ali_out.masked_fill(mask, 0)
            loss = (1 - ali_out.sum(dim=3)).sum() / (layers * tokens)
            return loss
        elif self.ali_type == 'ctc':
            ctc_logits = ctc_out.log_softmax(dim=2).transpose(1, 2).contiguous()
            label_len = torch.sum((ctc_label != self.ignore_id), dim=1).tolist()
            #ali = ctc_force_alignment(ctc_logits, ctc_label.int(), ctc_len.int(), label_len).float()
            ali = ctc_force_align(ctc_logits, ctc_label.tolist(), ctc_len.tolist(), label_len)
            ali = torch.from_numpy(ali).to(ctc_logits)
            hlen = enc_mask.size(1)
            pos_vec = torch.arange(1, hlen + 1, dtype=ali_out.dtype, device=ali_out.device).unsqueeze(1)
            ali_out = torch.matmul(ali_out, pos_vec).squeeze(3)[:,:,:-1] # not inc. eos
            ylens = torch.sum((ali_beg != self.ignore_id).float(), dim=1, keepdim=True) # batch x 1, not inc. eos
            mask = (make_pad_mask(ylens.reshape(-1).tolist())).to(ali_out.device).unsqueeze(1)
            total = mask.eq(0).float().sum() * ali_out.size(1)
            lat = (ali_out - ali.unsqueeze(1)).masked_fill(mask, 0.0)
            return torch.pow(lat, 2).sum() / total / hlen
        else:
            hlen = enc_mask.size(1)
            pos_vec = torch.arange(1, hlen + 1, dtype=ali_out.dtype, device=ali_out.device).unsqueeze(1)
            ali_out = torch.matmul(ali_out, pos_vec).squeeze(3)[:,:,:-1] # not inc. eos
            ylens = torch.sum((ali_beg != self.ignore_id).float(), dim=1, keepdim=True) # batch x 1, not inc. eos
            mask = (make_pad_mask(ylens.reshape(-1).tolist())).to(ali_out.device).unsqueeze(1)
            total = mask.eq(0).float().sum() * ali_out.size(1)
            # lat: batch x time1, not inc.
            if self.ali_type == 'mid':
                ali = (ali_beg.float() + ali_end.float()) / 2
                lat = (ali_out - ali.unsqueeze(1)).masked_fill(mask, 0.0)
            elif self.ali_type == 'end':
                ali = ali_end.float()
                lat = (ali_out - ali.unsqueeze(1)).masked_fill(mask, 0.0)
            elif self.ali_type == 'beg':
                ali = ali_beg.float()
                lat = (ali_out - ali.unsqueeze(1)).masked_fill(mask, 0.0)        
            else:
                print("NOT implement align loss type %s"%self.ali_type)
                exit(1)
            return torch.pow(lat, 2).sum() / total / hlen