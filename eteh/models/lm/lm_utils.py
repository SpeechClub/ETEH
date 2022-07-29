#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import random
import six



def read_tokens(filename, label_dict):
    """Read tokens as a sequence of sentences

    :param str filename : The name of the input file
    :param dict label_dict : dictionary that maps token label string to its ID number
    :return list of ID sequences
    :rtype list
    """

    data = []
    for ln in open(filename, 'rb').readlines():
        data.append(np.array([label_dict[label]
                              if label in label_dict else label_dict['<unk>']
                              for label in ln.decode('utf-8').split()], dtype=np.int32))
    return data


def count_tokens(data, unk_id=None):
    """Count tokens and oovs in token ID sequences

    :param list[np.ndarray] data: list of token ID sequences
    :param int unk_id: ID of unknown token '<unk>'
    :return number of token occurrences, number of oov tokens
    :rtype int, int
    """

    n_tokens = 0
    n_oovs = 0
    for sentence in data:
        n_tokens += len(sentence)
        if unk_id is not None:
            n_oovs += np.count_nonzero(sentence == unk_id)
    return n_tokens, n_oovs


def compute_perplexity(result):
    """Computes and add the perplexity to the LogReport

    :param dict result: The current observations
    """
    # Routine to rewrite the result dictionary of LogReport to add perplexity values
    result['perplexity'] = np.exp(result['main/loss'] / result['main/count'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


# TODO(Hori): currently it only works with character-word level LM.
#             need to consider any types of subwords-to-word mapping.
def make_lexical_tree(word_dict, subword_dict, word_unk, lexicon=None):
    """Make a lexical tree to compute word-level probabilities

    """
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            if lexicon is not None:
                assert isinstance(lexicon, dict) and w in lexicon
                w_lex = lexicon[w]
            else:
                w_lex = list(w)
            for i, c in enumerate(w_lex):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w_lex) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root
