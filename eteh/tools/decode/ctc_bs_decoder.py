# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from __future__ import print_function
import numpy as np
import math
import collections

NEG_INF = -float("inf")


class CTC_Decoder(object):
    def __init__(self, beam_size, ctc_beam=15, blank=0, sos=0, rnn_lm=None, lm_rate=0.0):
        self.beam_size = beam_size
        self.blank = blank
        self.rnn_lm = rnn_lm
        self.lm_rate = lm_rate
        self.sos = sos
        self.ctc_beam = ctc_beam
        if self.rnn_lm is None:
            self.lm_rate = 0.0
        else:
            self.rnn_lm.eval()

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

    def decode_problike(self, probs, do_log=False):
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
        if self.rnn_lm is not None:
            start_state, start_lm = self.rnn_lm.predict(
                np.array([self.sos]), None)
            beam = [
                [(self.sos,), [0.0, NEG_INF, start_state, start_lm[-1].cpu().numpy()]]]
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
                        q = self.lm_rate * prefix_lm[s]
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
            if self.rnn_lm is not None:
                for b in beam:
                    if b[1][-1] is None:
                        new_state, prefix_lm = self.rnn_lm.predict(
                            np.array([b[0][-1]]), b[1][-2])
                        prefix_lm = prefix_lm[-1].cpu().numpy()
                        b[1][-2], b[1][-1] = new_state, prefix_lm
        N_best = []
        for b in beam:
            N_best.append((b[0], self.logsumexp(*b[1][:-2])))
        return N_best

