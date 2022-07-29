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


class Kaldi_Decoder(object):
    def __init__(self, beam, max_active, mdl, fst, word, acoustic_scale=0.1):
        from kaldi.asr import MappedLatticeFasterRecognizer
        from kaldi.decoder import LatticeFasterDecoderOptions
        from kaldi.matrix import Matrix
        from kaldi.util.table import SequentialMatrixReader
        decoder_opts = LatticeFasterDecoderOptions()
        decoder_opts.beam = beam
        decoder_opts.max_active = max_active
        #decoder_opts.lattice_beam = 7
        self.asr = MappedLatticeFasterRecognizer.from_files(
            mdl, fst, word,
            acoustic_scale=acoustic_scale, decoder_opts=decoder_opts)

    def decode_loglike(self, loglikes):
        from kaldi.matrix import Matrix
        out = self.asr.decode(Matrix(loglikes))
        return out

