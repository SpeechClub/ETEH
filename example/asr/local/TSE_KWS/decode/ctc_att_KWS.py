#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

from __future__ import print_function
import torch
import logging
import numpy as np
from eteh.utils.mask import subsequent_mask
import math
import collections
from eteh.tools.decode.ctc_att_decoder import CTC_ATT_Decoder

NEG_INF = -float("inf")

def maxargmax(*x):
    m=NEG_INF
    am=-1
    for i,j in enumerate(x):
        if j > m:
            m=j
            am=i
    return (m,am)

class Trie(object):
    def __init__(self):
        self.root = {}
    def insert(self, word):
        current = self.root
        for l in word:
            if l not in current:
                current[l] = {}
            current = current[l]
        current[None]=1
    def match(self, text):
        ans=[]
        current = self.root
        for i,l in enumerate(text):
            if None in current:
                ans.append(i)
            if l not in current:
                return ans
            current = current[l]
        if None in current:
            ans.append(len(text))
        return ans


class CTC_ATT_Decoder_KWS(CTC_ATT_Decoder):
    def decode_feat_tse(self, feat, f_len):
        feat = torch.as_tensor(feat).unsqueeze(0)
        enc_output, _, enc_tse, _ = self.model.encoder_forward_tse(feat, f_len)
        enc_tse_trsp = enc_tse.cpu().squeeze(0).transpose(0,1)
        if self.ctc_weight > 0.0:
            lpz = self.model.ctc_forward(enc_output)
            lpz = torch.log_softmax(lpz, -1)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)
        beam = self.beam
        penalty = self.penalty
        ctc_weight = self.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if self.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(self.maxlenratio * h.size(0)))
        minlen = int(self.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        rnnlm = self.rnnlm
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [
                y], 'rnnlm_prev': None, 'att_prev': None, 'score_this': [0.0]}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'att_prev': None, 'score_this': [0.0]}
        if lpz is not None:
            import numpy
            from eteh.ops.ctc_prefix_score import CTCPrefixScore
            ctc_prefix_score = CTCPrefixScore(
                lpz.detach().cpu().numpy(), 0, self.eos, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                CTC_SCORING_RATIO = 1.5
                ctc_beam = min(lpz.shape[-1], self.ctc_beam)
            else:
                ctc_beam = lpz.shape[-1] - 1
        hyp['attscoreseq']=[]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(
                    i + 1).unsqueeze(0).to(enc_output.device)
                ys = torch.tensor(hyp['yseq']).unsqueeze(
                    0).to(enc_output.device)
                local_att_scores, att_prev, dec_midlayer = self.model.decoder_forward_onestep_midlayer(ys, ys_mask, enc_output, hyp['att_prev'])
                local_att_scores = local_att_scores.cpu()
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(
                        vy, hyp['rnnlm_prev'])
                    local_lm_scores = local_lm_scores.cpu()
                    local_scores = local_att_scores + self.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores[:,1:], ctc_beam, dim=1)
                    local_best_ids = local_best_ids + 1
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * \
                        torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += self.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores[:,1:], beam, dim=1)
                    local_best_ids = local_best_ids + 1

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score_this'] = [0] * (1 + len(hyp['score_this']))
                    new_hyp['score_this'][:len(
                        hyp['score_this'])] = hyp['score_this']
                    new_hyp['score_this'][len(hyp['score_this'])] = float(
                        local_best_scores[0, j])
                    new_hyp['score'] = hyp['score'] + \
                        float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(
                        local_best_ids[0, j])
                    new_hyp['dec_midlayer'] = dec_midlayer
                    new_hyp['att_prev'] = att_prev
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    
                    #new_hyp['attscoreseq']=[0.0] * (1 + len(hyp['attscoreseq']))
                    #new_hyp['attscoreseq'][:len(hyp['attscoreseq'])]=hyp['attscoreseq']
                    #new_hyp['attscoreseq'][len(hyp['attscoreseq'])]=float(local_att_scores[0,local_best_ids[0,j]])
                    new_hyp['attscoreseq'] = hyp['attscoreseq'] + [float(local_att_scores[0,local_best_ids[0,j]])]

                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            #logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                #logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += 0
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if self.end_detect(ended_hyps, i) and self.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break
            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), self.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                'there is no N-best results, perform recognition again with smaller minlenratio.')
            self.minlenratio = max(0.0, self.minlenratio - 0.1)
            return self.decode_feat_tse(feat, f_len)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' +
                     str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps, enc_tse_trsp
    
    def kws_tse_1hyp(self, hyp, enc_tse_trsp, char_list, kwlist, tse_delay=False, tse_aggr_func="sum"):
        rec_output = hyp['yseq'][1:-1]
        rec_output_attscore = hyp['attscoreseq']
        rec_output_cemscore = None
        if len(rec_output)==0:
            return []
        dec_tse = self.model.decoder_forward_tse(hyp['dec_midlayer'])[0]
        
        rec_output
        #(hyp['tse_dec_emb'],tse_enc_emb_trsp,hyp['yseq'][1:-1],char_list,kwlist,True,      True,               hyp['attscoreseq'],hyp['cemscoreseq'])
        #(dec_tse,           enc_tse_trsp,    rec_output,       char_list,kwlist,tse_delay,tse_sumtrue_meanfalse,rec_output_attscore,rec_output_cemscore=None)

        if len(rec_output)==0:
            return []
        tse_output=torch.sigmoid(torch.matmul(dec_tse,enc_tse_trsp))
        if tse_delay:
            tse_output_eos=tse_output[0]
            tse_output=tse_output[1:]
        else:
            tse_output_eos=tse_output[-1]
            tse_output=tse_output[:-1]
        if len(rec_output_attscore)==1+len(rec_output):
            rec_output_attscore=rec_output_attscore[:-1]
        assert len(rec_output)==len(rec_output_attscore), (rec_output,len(rec_output),len(rec_output_attscore))
        assert tse_output.shape[0]==len(rec_output), (tse_output.shape[0],len(rec_output))
        if rec_output_cemscore is not None:
            if len(rec_output_cemscore)==1+len(rec_output):
               rec_output_cemscore=rec_output_cemscore[:-1]
            assert len(rec_output)==len(rec_output_cemscore), (rec_output,len(rec_output),len(rec_output_cemscore))
        
        rec_char=[char_list[id] for id in rec_output]
        reclist=[]
        for i in range(len(rec_char)):
            for j in kwlist.match(rec_char[i:]):
                word=''.join(rec_char[i:i+j])
                print("kw:",word)
                attscore=sum(rec_output_attscore[i:i+j])/(j)
                if rec_output_cemscore is not None:
                    cemscore=sum(rec_output_cemscore[i:i+j])/(j)
                else:
                    cemscore=None
                
                rec=[False]
                recz=[]
                for c in rec_char[:i]:
                    recz.append(len(rec))
                    rec.append(True)
                    rec.append(False)
                beg_char=len(rec)
                for c in rec_char[i:i+j]:
                    recz.append(len(rec))
                    rec.append(True)
                end_char=len(rec)
                rec.append(False)
                for c in rec_char[i+j:]:
                    recz.append(len(rec))
                    rec.append(True)
                    rec.append(False)
                reclist.append((rec,recz,word,beg_char,end_char,attscore,cemscore))
        
        res3=[]
        for (rec,recz,word,beg_char,end_char,attscore,cemscore) in reclist:
            mat=torch.ones(len(rec),tse_output.shape[1])
            mat[:]=torch.log(tse_output_eos*1.0)

    #        mat[1:,:]=-100000000.0 # not insert sil
            mat[recz]=torch.log(tse_output)
            route=torch.zeros(mat.shape,dtype=torch.int8)
            # route=mat.clone().zero_().int()
            # torch.save(mat,"/data/gaochangfeng/docker/project/eteh_egs/asr/local/code/e2e_ctc_att_tse/newta.mat",_use_new_zipfile_serialization=False)

            mat[2:,0]=-100000000.0
            #print(mat)
            
            for j in range(1,mat.size(1)):
                mat[0][j]+=mat[0][j-1]
                m,am=maxargmax(mat[1][j-1],mat[0][j-1])
                mat[1][j]+=m
                route[1][j]=am
                for i in range(2,mat.size(0)):
                    if rec[i-1]:
                        m,am=maxargmax(mat[i][j-1],mat[i-1][j-1])
                    else:
                        m,am=maxargmax(mat[i][j-1],mat[i-1][j-1],mat[i-2][j-1])
                    mat[i][j]+=m
                    route[i][j]=am
            m,am=maxargmax(mat[-2][-1],mat[-1][-1])
            
            i=mat.size(0)-2+am
            isinword=False
            end_frame=None
            for j in range(mat.size(1)-1,-1,-1):
                if not isinword and i==end_char-1:
                    end_frame=j+1
                    isinword=True
                i-=int(route[i][j])
                if isinword and i<beg_char:
                    beg_frame=j
                    isinword=False
            if isinword:
                beg_frame=0
            
            assert end_frame is not None, (recz,route)
            #if cemscore is not None:
            #    score=cemscore
            #else:
            #    score=float(mat[end_char-1][end_frame-1]-(mat[beg_char-route[beg_char][beg_frame]][beg_frame-1] if beg_frame>0 else 0.0))/(end_frame-beg_frame)
            res3.append((word,beg_frame,end_frame,attscore,cemscore))
        return res3
    
    def kws_tse(self, hyps, *args, **kwargs): # enc_tse_trsp, char_list, kwlist, tse_delay=False, tse_aggr_func="sum"):
        kws_res_final=[]
        kws_res_final_nbest=[]
        for hyp in reversed(hyps):
            if len(hyp['yseq'])<=2:
                continue
            # hyp['cemscoreseq']=cem(hyp['feat_cem']).sigmoid().log().squeeze(0).tolist() if cem is not None and hyp['feat_cem'] is not None else None
            kws_res = self.kws_tse_1hyp(hyp, *args, **kwargs)

            for (word,beg_frame,end_frame,attscore,cemscore) in kws_res:
                kws_res_final_nbest.append((word,beg_frame,end_frame,attscore,cemscore))
                score=attscore
                add=True
                rm=[]
                for (word0,beg_frame0,end_frame0,score0) in kws_res_final:
                    if word==word0 and not (beg_frame>=end_frame0 or beg_frame0>=end_frame):
                        if score>score0:
                            rm.append((word0,beg_frame0,end_frame0,score0))
                        else:
                            add=False
                            break
                if add:
                    for jj in rm:
                        kws_res_final.remove(jj)
                    kws_res_final.append((word,beg_frame,end_frame,score))
        return kws_res_final

    def decode_feat_online_tse(self, feat, f_len):
        feat = torch.as_tensor(feat).unsqueeze(0)
        # enc_output, _ = self.model.encoder_forward_online_per_chunk(feat, f_len, self.chunk, self.right)
        enc_output, _ = self.model.encoder_forward_online(feat, f_len)
        if self.ctc_weight > 0.0:
            lpz = self.model.ctc_forward(enc_output)
            lpz = torch.log_softmax(lpz, -1)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)
        beam = self.beam
        penalty = self.penalty
        ctc_weight = self.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if self.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(self.maxlenratio * h.size(0)))
        minlen = int(self.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        rnnlm = self.rnnlm
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None, 'att_prev': None, 'att_lm_score': 0.0, 'score_this': [0.0]}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'att_prev': None, 'att_lm_score': 0.0, 'score_this': [0.0]}
        if lpz is not None:
            from eteh.ops.ctc_prefix_score import TCTCPrefixScore
            ctc_prefix_score = TCTCPrefixScore(
                lpz.detach().cpu().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'], hyp['ctc_hist_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'], hyp['ctc_end'] = 0.0, 0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                CTC_SCORING_RATIO = 1.5
                ctc_beam = min(lpz.shape[-1], self.ctc_beam)
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []
        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(
                    i + 1).unsqueeze(0).to(enc_output.device)
                ys = torch.tensor(hyp['yseq']).unsqueeze(
                    0).to(enc_output.device)
                local_att_scores, att_prev = self.model.decoder_forward_online(ys, ys_mask, enc_output, hyp['att_prev'])
                local_att_scores = local_att_scores.cpu()
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(
                        vy, hyp['rnnlm_prev'])
                    local_lm_scores = local_lm_scores.cpu()
                    local_scores = local_att_scores + self.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states, ctc_hists, ctc_end = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'], hyp['ctc_hist_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * \
                        torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += self.lm_weight * \
                            local_lm_scores[:, local_best_ids[0]]
                        local_att_lm_scores = (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                                            + self.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    else:
                        local_att_lm_scores = (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score_this'] = [0] * (1 + len(hyp['score_this']))
                    new_hyp['score_this'][:len(
                        hyp['score_this'])] = hyp['score_this']
                    new_hyp['score_this'][len(hyp['score_this'])] = float(
                        local_best_scores[0, j])
                    new_hyp['score'] = hyp['score'] + \
                        float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(
                        local_best_ids[0, j])
                    new_hyp['att_prev'] = att_prev
                    new_hyp['att_lm_score'] = hyp['att_lm_score'] + local_att_lm_scores[0, joint_best_ids[0, j]]
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                        new_hyp['ctc_hist_prev'] = ctc_hists
                        new_hyp['ctc_end'] = ctc_end
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            #logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                #logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += 0
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if self.end_detect_online(ended_hyps, remained_hyps, i, h.size(0)) and self.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break
            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))
            
        # replace T-CTC with CTC prefix scores of ended hypotheses
        # important to prune too short hypotheses and no need for length normalization
        for idx, hyp in enumerate(ended_hyps):
            if hyp['ctc_end'] + 1 < h.size(0):
                ctc_rescore = ctc_prefix_score.rescore(hyp['yseq'], hyp['ctc_state_prev'])
                hyp['score'] = ctc_weight * ctc_rescore + hyp['att_lm_score']
                hyp['score'] = hyp['score'].float()

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), self.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                'there is no N-best results, perform recognition again with smaller minlenratio.')
            self.minlenratio = max(0.0, self.minlenratio - 0.1)
            return self.decode_feat_online_tse(feat, f_len)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' +
                     str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps
