#!/usr/bin/env python

# Copyright 2018 Mitsubishi Electric Research Laboratories (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lm_utils import make_lexical_tree 
from eteh.reader.txtfile_reader import dict_reader
from eteh.utils.data_utils import to_device

class BasicLM(nn.Module):
    def __init__(self,model): 
        super(BasicLM, self).__init__()
        self.model = model

    def forward(self,x,state=None): 
        x = to_device(self.model,x)
        return self.model(x,state)

    def forward_onehot(self, x, state=None):
        x = to_device(self.model,x)
        return self.model.forward_onehot(x, state)

    def predict(self,x,state=None):      
        x = torch.as_tensor(x).long()
        #print('x size',x.size())
        state,y = self.forward(x, state)
        return state, F.log_softmax(y, dim=-1)
        
    def score(self,x,idx,state=None):
        x = torch.as_tensor(x).long()
        with torch.torch.no_grad():
            state,y = self.predict(x,state)
        return y[-1,idx].item(),state
        
    def rescore(self,x):
        x = torch.as_tensor(x).long()
        x_in,y_out = x[:-1], x[1:]
        with torch.torch.no_grad():
            _, x_out = self.predict(x,state=None)
        score_lm = 0
        for i in range(y_out.size(0)):
            score_lm = score_lm + x_out[i,y_out[i]]
        return score_lm/(y_out.size(0)+1)

    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict)


# Definition of a multi-level (subword/word) language model
class MultiLevelLM(BasicLM):
    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(self, model, submodel, word_dict="", subword_dict="",
                 subwordlm_weight=0.8, oov_penalty=1.0, open_vocab=True):
        super(MultiLevelLM, self).__init__(model)
        self.submodel = submodel
        if isinstance(word_dict,str):
            word_dict = dict_reader(word_dict)
        if isinstance(subword_dict,str):
            subword_dict = dict_reader(subword_dict)
        self.word_eos = word_dict['<eos>']
        self.word_unk = word_dict['<unk>']
        self.var_word_eos = torch.LongTensor([self.word_eos])
        self.var_word_unk = torch.LongTensor([self.word_unk])
        self.space = subword_dict['<space>']
        self.eos = subword_dict['<eos>']
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.log_oov_penalty = math.log(oov_penalty)
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.subwordlm_weight = subwordlm_weight
        self.normalized = True

    def forward(self, x, state=None):
        # update state with input label x
        if state is None:  # make initial states and log-prob vectors
            self.var_word_eos = to_device(self, self.var_word_eos)
            self.var_word_unk = to_device(self, self.var_word_eos)
            wlm_state, z_wlm = self.model(self.var_word_eos, None)
            wlm_logprobs = F.log_softmax(z_wlm, dim=1)
            clm_state, z_clm = self.submodel(x, None)
            log_y = F.log_softmax(z_clm, dim=1) * self.subwordlm_weight
            new_node = self.lexroot
            clm_logprob = 0.
            xi = self.space
        else:
            clm_state, wlm_state, wlm_logprobs, node, log_y, clm_logprob = state
            xi = int(x)
            if xi == self.space:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = to_device(self, torch.LongTensor([node[1]]))
                else:  # this node is not a word end, which means <unk>
                    w = self.var_word_unk
                # update wordlm state and log-prob vector
                wlm_state, z_wlm = self.model(w, wlm_state)
                wlm_logprobs = F.log_softmax(z_wlm, dim=1)
                new_node = self.lexroot  # move to the tree root
                clm_logprob = 0.
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
                clm_logprob += log_y[0, xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
                clm_logprob += log_y[0, xi]
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = to_device(self, torch.full((1, self.subword_dict_size), self.logzero))
                return (clm_state, wlm_state, wlm_logprobs, None, log_y, 0.), log_y

            clm_state, z_clm = self.submodel(x, clm_state)
            log_y = F.log_softmax(z_clm, dim=1) * self.subwordlm_weight

        # apply word-level probabilies for <space> and <eos> labels
        if xi != self.space:
            if new_node is not None and new_node[1] >= 0:  # if new node is word end
                wlm_logprob = wlm_logprobs[:, new_node[1]] - clm_logprob
            else:
                wlm_logprob = wlm_logprobs[:, self.word_unk] + self.log_oov_penalty
            log_y[:, self.space] = wlm_logprob
            log_y[:, self.eos] = wlm_logprob
        else:
            log_y[:, self.space] = self.logzero
            log_y[:, self.eos] = self.logzero

        return (clm_state, wlm_state, wlm_logprobs, new_node, log_y, float(clm_logprob)), log_y

    def final(self, state):
        clm_state, wlm_state, wlm_logprobs, node, log_y, clm_logprob = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = to_device(self, torch.LongTensor([node[1]]))
        else:  # this node is not a word end, which means <unk>
            w = self.var_word_unk
        wlm_state, z_wlm = self.model(w, wlm_state)
        return float(F.log_softmax(z_wlm, dim=1)[:, self.word_eos])

    def load_parameter(self,dict1,dict2):
        self.model.load_state_dict(dict1)
        self.submodel.load_state_dict(dict2)


# Definition of a look-ahead word language model
class LookAheadWordLM(BasicLM):
    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(
            self, model, word_dict, subword_dict, 
            oov_penalty=0.0001, open_vocab=True, lexicon=None,
            word_eos='<eos>', word_unk='<unk>', 
            space='<space>', subword_eos='<eos>', subword_unk='<unk>'
                        
    ):
        super(LookAheadWordLM, self).__init__(model)
        if isinstance(word_dict,str):
            word_dict = dict_reader(word_dict)
        if isinstance(subword_dict,str):
            subword_dict = dict_reader(subword_dict)
        self.word_eos = word_dict[word_eos]
        self.word_unk = word_dict[word_unk]
        self.var_word_eos = torch.LongTensor([self.word_eos])
        self.var_word_unk = torch.LongTensor([self.word_unk])
        self.space = subword_dict[space]
        self.eos = subword_dict[subword_eos]
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk, lexicon)
        self.oov_penalty = oov_penalty
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict) + 1
        self.zero_tensor = torch.FloatTensor([self.zero])
        self.normalized = True
        self.lexicon = lexicon

        self.new_word_label = [self.space]
        for subword in subword_dict:
            if subword.startswith(space):
                self.new_word_label.append(subword_dict[subword])

    def forward(self, x, state=None):
        # update state with input label x
        if state is None:  # make initial states and cumlative probability vector
            self.var_word_eos = to_device(self, self.var_word_eos)
            self.var_word_unk = to_device(self, self.var_word_eos)
            self.zero_tensor = to_device(self, self.zero_tensor)
            wlm_state, z_wlm = self.model(self.var_word_eos, None)
            cumsum_probs = torch.cumsum(F.softmax(z_wlm, dim=1), dim=1)
            new_node = self.lexroot
            xi = self.space
        else:
            wlm_state, cumsum_probs, node = state
            xi = int(x)
            if xi in self.new_word_label:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = to_device(self, torch.LongTensor([node[1]]))
                else:  # this node is not a word end, which means <unk>
                    w = self.var_word_unk
                # update wordlm state and cumlative probability vector
                wlm_state, z_wlm = self.model(w, wlm_state)
                cumsum_probs = torch.cumsum(F.softmax(z_wlm, dim=1), dim=1)                
                if xi != self.space:
                    new_node = self.lexroot[0][xi]
                else:
                    new_node = self.lexroot  # move to the tree root
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = to_device(self, torch.full((1, self.subword_dict_size), self.logzero))
                return (wlm_state, None, None), log_y

        if new_node is not None:
            succ, wid, wids = new_node
            # compute parent node probability
            sum_prob = (cumsum_probs[:, wids[1]] - cumsum_probs[:, wids[0]]) if wids is not None else 1.0
            if sum_prob < self.zero:
                log_y = to_device(self, torch.full((1, self.subword_dict_size), self.logzero))
                return (wlm_state, cumsum_probs, new_node), log_y
            # set <unk> probability as a default value
            unk_prob = cumsum_probs[:, self.word_unk] - cumsum_probs[:, self.word_unk - 1]
            y = to_device(self, torch.full((1, self.subword_dict_size), float(unk_prob) * self.oov_penalty))
            # compute transition probabilities to child nodes
            for cid, nd in succ.items():
                y[:, cid] = (cumsum_probs[:, nd[2][1]] - cumsum_probs[:, nd[2][0]]) / sum_prob
            # apply word-level probabilies for <space> and <eos> labels
            if wid >= 0:
                wlm_prob = (cumsum_probs[:, wid] - cumsum_probs[:, wid - 1]) / sum_prob
                if not self.lexicon:
                    y[:, self.space] = wlm_prob
                y[:, self.eos] = wlm_prob
            elif xi == self.space:
                y[:, self.space] = self.zero
                y[:, self.eos] = self.zero
            log_y = torch.log(torch.max(y, self.zero_tensor))  # clip to avoid log(0)
        else:  # if no path in the tree, transition probability is one
            log_y = to_device(self, torch.zeros(1, self.subword_dict_size))
        return (wlm_state, cumsum_probs, new_node), log_y
        
    def predict(self,x,state=None):      
        x = torch.as_tensor(x).long()
        state,y = self.forward(x, state)
        return state, y

    def final(self, state):
        wlm_state, cumsum_probs, node = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = to_device(self, torch.LongTensor([node[1]]))
        else:  # this node is not a word end, which means <unk>
            w = self.var_word_unk
        wlm_state, z_wlm = self.model(w, wlm_state)
        return float(F.log_softmax(z_wlm, dim=1)[:, self.word_eos])
