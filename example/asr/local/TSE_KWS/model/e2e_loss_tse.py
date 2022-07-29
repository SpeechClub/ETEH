#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

from eteh.models.e2e_ctc_att.e2e_loss import E2E_Loss
from eteh.utils.mask import make_pad_mask
from eteh.utils.data_utils import calcurate_cer, calculate_cer_ctc
import torch

def sub4(y):
    return ((y-1)//2-1)//2

class Tse_Loss(torch.nn.Module):
    def __init__(self, delay=True, aggr_func="sum", enc_sub_func="sub4", ignore_id=-1):
        super(Tse_Loss, self).__init__()
        self.delay = delay
        self.aggr_func = aggr_func
        self.enc_sub_func = enc_sub_func
        self.ignore_id = ignore_id

    def forward(self, tse_mat, word_beg, word_end, enc_len, vad=None):
        ylen = (word_beg != self.ignore_id).sum(dim=1) + 1
        if self.enc_sub_func == "sub4":
            word_beg = sub4(word_beg)
            word_end = sub4(word_end)
        elif self.enc_sub_func == "":
            pass
        else:
            raise NotImplementedError

        token2word=torch.zeros(tse_mat.shape[0],ylen.max(),tse_mat.shape[1])
        for i in range(tse_mat.shape[0]):
            token2word[i,0 if self.delay else ylen[i]-1,0 if self.delay else ylen[i]-1]=1.0
            for k,(st,ed) in enumerate(zip(torch.nonzero(word_beg[i]!=self.ignore_id),torch.nonzero(word_end[i]!=self.ignore_id)+1)):
                if self.delay:
                    if self.aggr_func == "sum":
                        token2word[i,k+1,st+1:ed+1]=1.0
                    elif self.aggr_func == "mean":
                        token2word[i,k+1,st+1:ed+1]=1.0/(ed-st)
                    else:
                        raise NotImplementedError
                else:
                    if self.aggr_func == "sum":
                        token2word[i,k,st:ed]=1.0
                    elif self.aggr_func == "mean":
                        token2word[i,k,st:ed]=1.0/(ed-st)
                    else:
                        raise NotImplementedError
        tse_mat=torch.matmul(token2word.to(tse_mat.device),tse_mat)

        assert ylen.max()==tse_mat.shape[1]

        target=torch.zeros(tse_mat.shape)
        for i in range(target.shape[0]):
            word_beg_i=word_beg[i][torch.nonzero(word_beg[i]!=self.ignore_id)]
            word_end_i=word_end[i][torch.nonzero(word_end[i]!=self.ignore_id)]
            for j in range(word_beg_i.shape[0]):
                target[i,j+1 if self.delay else j,word_beg_i[j]:word_end_i[j]]=1.0
            if vad is not None:
                vadstt,vadend=vad[0][i],vad[1][i]
                for vs,ve in zip(vadstt,vadend):
                    if vs==self.ignore_id: break
                    target[i,:,vs:ve if ve != -1 else None]=0.0
                #torch.set_printoptions(threshold=np.inf)
                #if i==0: print(vadstt,vadend,target[i])

            target[i,0 if self.delay else ylen[i]-1]=torch.max(torch.zeros_like(target[i,0]),1.0-target[i].sum(dim=0))
            #target[i,0 if self.delay else ylen[i]-1]=1.0-target[i].sum(dim=0)
            #if any(target[i,0 if self.delay else ylen[i]-1]<0.0):
            #    print(target[i])
            #    print(word_beg[i],word_end[i],ylen[i])
        target=target.to(tse_mat.device)
        
        mask1=((~make_pad_mask(ylen)).unsqueeze(2).expand(tse_mat.shape))
        mask2=(~make_pad_mask(enc_len.tolist(), max_length=tse_mat.size(2)).unsqueeze(1).expand(tse_mat.shape))
        mask=(mask1&mask2).to(tse_mat.device)
 
        return torch.nn.functional.binary_cross_entropy_with_logits(tse_mat, target, mask)

class CTC_CE_TSE_Loss(E2E_Loss):
    def __init__(self, size, padding_idx, smoothing, rate=0.5, tse_rate=1.0, tse_delay=False, tse_aggr_func="sum"):
        super(CTC_CE_TSE_Loss, self).__init__(size, padding_idx, smoothing, rate)
        self.tse_rate = tse_rate
        self.tse_loss = Tse_Loss(delay=tse_delay, aggr_func=tse_aggr_func, enc_sub_func="sub4", ignore_id=-1)

    def forward(self, att_out, ctc_out, tse_out, ys_out, y, hs_len, word_beg, word_end):
        #with autocast(): //å¢å è¿ä¸ªä¿®é¥°æè½æ­£ç¡®è¿è¡amp
        tse_loss = self.tse_loss(tse_out, word_beg, word_end, hs_len)
        if self.tse_loss!=1.0:
            att_loss = self.att_loss(att_out, ys_out)
            ctc_loss = self.ctc_loss(ctc_out, hs_len, y)
            return (1-self.tse_rate)*(1 - self.rate) * att_loss + (1-self.tse_rate)*self.rate * ctc_loss \
                + self.tse_rate*tse_loss, att_loss, ctc_loss, tse_loss
        else:
            return tse_loss, None, None, tse_loss

    def train_forward(self, input_dict):
        loss_main, att_loss, ctc_loss, tse_loss = self.forward(
            att_out=input_dict["att_out"],
            ctc_out=input_dict["ctc_out"],
            tse_out=input_dict["tse_out"],
            ys_out=input_dict["ys_out"],
            y=input_dict["y"],
            hs_len=input_dict["hs_len"],
            word_beg=input_dict["y_beg"],
            word_end=input_dict["y_end"]
        )
        if self.tse_rate != 1.0:
            att_corr = calcurate_cer(input_dict["att_out"].detach().cpu().numpy(), input_dict["ys_out"].detach().cpu().numpy())
            return {
                "loss_main": loss_main,
                "att_loss": att_loss.item(),
                "ctc_loss": ctc_loss.item(),
                "tse_loss": tse_loss.item(),
                "att_corr": att_corr
            }
        else:
            return {
                "loss_main": loss_main,
                "tse_loss": tse_loss.item()
            }

    def valid_forward(self, input_dict):
        valdi_dict = self.train_forward(input_dict)
        if self.tse_rate != 1.0:
            ctc_cer = calculate_cer_ctc(input_dict["ctc_out"].cpu().numpy(), input_dict["y"].cpu().numpy(), xs_len=input_dict["hs_len"])
            valdi_dict["ctc_cer"] = ctc_cer
        return valdi_dict
