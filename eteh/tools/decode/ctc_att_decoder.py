# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from __future__ import print_function
import torch
import logging
import numpy as np
from eteh.utils.mask import subsequent_mask
import math
import collections

NEG_INF = -float("inf")

class CTC_Decoder_LASRescore(object):
    def __init__(self, beam_size, ctc_beam=15, blank=0, sos=0, las_model=None, las_rate=0.0):
        self.beam_size = beam_size
        self.blank = blank
        self.las_model = las_model
        self.las_rate = las_rate
        self.sos = sos
        self.ctc_beam = ctc_beam
        if self.las_model is None:
            self.las_rate = 0.0
        else:
            self.las_model.eval()

    def make_new_beam(self):
        def fn(): return [NEG_INF, NEG_INF, None, None]
        return collections.defaultdict(fn)

    def logsumexp(self, *args):
        """
        Stable log sum exp.
        """
        if all(a == NEG_INF for a in args):
            return NEG_INF
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max)
                           for a in args))
        return a_max + lsp

    def decode_problike(self, probs, feat, f_len, do_log=False):
        """
        Performs inference for the given output probabilities.
        Arguments:
        probs: The output probabilities (e.g. post-softmax) for each
            time step. Should be an array of shape (time x output dim).
        beam_size (int): Size of the beam to use during inference.
        blank (int): Index of the CTC blank label.
        Returns the output label sequence and the corresponding negative
        log-likelihood estimated by the decoder.
        """
        # T表示时间，S表示词表大小
        T, S = probs.shape
        # 求概率的对数
        if self.ctc_beam == 0:
            ctc_beam = S
        else:
            ctc_beam = self.ctc_beam
        if do_log:
            probs = np.log(probs)

        # Elements in the beam are (prefix, (p_blank, p_no_blank))
        # Initialize the beam with the empty sequence, a probability of
        # 1 for ending in blank and zero for ending in non-blank
        # (in log space).
        if self.las_model is not None:
            enc_output, _ = self.las_model.encoder_forward(feat, f_len)
            ys_mask = subsequent_mask(1).unsqueeze(0).to(enc_output.device)
            ys = torch.tensor([self.sos]).unsqueeze(0).to(enc_output.device)
            local_att_scores = torch.log_softmax(self.las_model.decoder_forward(
                ys, ys_mask, enc_output, None), -1)[:, -1].cpu()[0]
            #start_state, start_lm = self.las_model.predict(np.array([self.sos]), None)
            beam = [[(self.sos,), [0.0, NEG_INF, None, local_att_scores]]]
        else:
            beam = [[(self.sos,), [0.0, NEG_INF, None, None]]]
        for t in range(T):  # Loop over time
            next_beam = self.make_new_beam()
            for prefix, (p_b, p_nb, lm_state, prefix_lm) in beam:  # Loop over beam
                # p_b表示前缀最后一个是blank的概率，p_nb是非blank的概率
                prob_t = np.copy(probs[t])
                for i in range(ctc_beam):  # Loop over vocab
                    s = np.argmax(prob_t)
                    p = prob_t[s]
                    prob_t[s] = NEG_INF
                    if s == self.blank:
                        # 增加的字母是blank
                        # n_p_b和n_p_nb第一n表示new是新创建的路径
                        n_p_b, n_p_nb, _, _ = next_beam[prefix]
                        n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = [
                            n_p_b, n_p_nb, lm_state, prefix_lm]
                        continue
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)  # new prefix
                    n_p_b, n_p_nb, _, _ = next_beam[n_prefix]
                    if prefix_lm is not None:
                        q = self.las_rate * prefix_lm[s]
                    else:
                        q = 0
                    if s != end_t:
                        # 如果s不和上一个不重复，则更新非空格的概率
                        n_p_nb = self.logsumexp(
                            n_p_nb, p_b + p + q, p_nb + p + q)
                    else:
                        # 如果s和上一个重复，也要更新非空格的概率
                        # We don't include the previous probability of not ending
                        # in blank (p_nb) if s is repeated at the end. The CTC
                        # algorithm merges characters not separated by a blank.
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p + q)
                    # *NB* this would be a good place to include an LM score.
                    next_beam[n_prefix] = [n_p_b, n_p_nb, lm_state, None]
                    # If s is repeated at the end we also update the unchanged
                    # prefix. This is the merging case.
                    if s == end_t:
                        n_p_b, n_p_nb, _, _ = next_beam[prefix]
                        n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = [
                            n_p_b, n_p_nb, lm_state, prefix_lm]
            # Sort and trim the beam before moving on to the
            # next time-step.
            # 根据概率进行排序，每次保留概率最高的beam_size条路径
            beam = sorted(next_beam.items(),
                          key=lambda x: self.logsumexp(*x[1][:-2]),
                          reverse=True)
            beam = beam[:self.beam_size]
            if self.las_model is not None:
                for b in beam:
                    if b[1][-1] is None:
                        hyseq = b[0]
                        ys_mask = subsequent_mask(len(hyseq)).unsqueeze(
                            0).to(enc_output.device)
                        ys = torch.tensor(hyseq).unsqueeze(
                            0).to(enc_output.device)
                        local_att_scores = torch.log_softmax(self.las_model.decoder_forward(
                            ys, ys_mask, enc_output, None), -1)[:, -1].cpu()[0]
                        prefix_lm = local_att_scores.numpy()
                        b[1][-2], b[1][-1] = None, prefix_lm
        N_best = []
        for b in beam:
            N_best.append((b[0], -self.logsumexp(*b[1][:-2])))
        return N_best


class CTC_ATT_Decoder(object):
    def __init__(self, model, sos, eos, beam=5, ctc_beam=15, nbest=1, maxlenratio=0, minlenratio=0, rnnlm=None, ctc_weight=0.5, penalty=0, lm_weight=0):
        self.model = model
        self.beam = beam
        self.nbest = nbest
        self.rnnlm = rnnlm
        if self.rnnlm is not None:
            self.rnnlm.eval()
        self.ctc_weight = ctc_weight
        self.lm_weight = lm_weight
        self.penalty = penalty
        self.sos = sos
        self.eos = eos
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.ctc_beam = ctc_beam
        self.model.eval()

    def decode_feat(self, feat, f_len):
        feat = torch.as_tensor(feat).unsqueeze(0)
        enc_output, _ = self.model.encoder_forward(feat, f_len)
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
                local_att_scores, att_prev = self.model.decoder_forward_onestep(ys, ys_mask, enc_output, hyp['att_prev'])
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
                    new_hyp['att_prev'] = att_prev
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
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
            return self.decode_feat(feat)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' +
                     str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def decode_feat_online(self, feat, f_len):
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
            return self.decode_feat(feat)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' +
                     str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps
    @staticmethod
    def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
        """End detection
        desribed in Eq. (50) of S. Watanabe et al
        "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

        :param ended_hyps:
        :param i:
        :param M:
        :param D_end:
        :return:
        """
        if len(ended_hyps) == 0:
            return False
        count = 0
        best_hyp = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[0]
        import six
        for m in six.moves.range(M):
            # get ended_hyps with their length is i - m
            hyp_length = i - m
            hyps_same_length = [
                x for x in ended_hyps if len(x['yseq']) == hyp_length]
            if len(hyps_same_length) > 0:
                best_hyp_same_length = sorted(
                    hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
                if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                    count += 1

        if count == M:
            return True
        else:
            return False


    @staticmethod                
    def end_detect_online(ended_hyps, remained_hyps, i, T, M=3, D_end=np.log(1 * np.exp(-10))):
        '''End detection for online scenario

        :param ended_hyps:
        :param i: max length - 2
        :return:
        '''
        if len(ended_hyps) == 0:
            return False
        if len(remained_hyps) == 0:
            return True
        
        # all ctc end point of remained hyps reaches T
        remained_hyps_ctc_end = sorted(remained_hyps, key=lambda x: x['ctc_end'])[0]['ctc_end'] + 1 # index from 0
        flag1 = True if remained_hyps_ctc_end == T else False
        
        # no higher score for longer ended hyps
        hyp_length = i + 2 # longest ended hyps, index from 0, plus <sos>
        hyps_long_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_long_length) > 0:
            best_hyp_long_length = sorted(hyps_long_length, key=lambda x: x['score'], reverse=True)[0]
        else:
            return False

        count = 0        
        for m in range(M):
            # get ended_hyps with their length is i + 1 - m 
            hyp_length = i + 1 - m
            hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
            if len(hyps_same_length) > 0:
                best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
                if best_hyp_long_length['score'] - best_hyp_same_length['score'] < D_end:
                    count += 1 
        flag2 = True if count == M else False
        
        if flag1 and flag2:
            return True
        else:
            return False

        if count == M:
            return True
        else:
            return False
