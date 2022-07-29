# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from eteh.reader.txtfile_reader import dict_reader
import numpy
import math
import itertools as it

try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


class CTC_KenLM_Decoder(object):
    def __init__(self,
                 beam_size, beam_threshold, 
                 lexicon=None, tokens_dict=None, kenlm_model=None,
                 sos='<eos>', blk='<blank>', unk='<unk>', sil=None,
                 lm_weight=2.0, word_score=-1, unk_score=-math.inf, sil_score=0,
                 log_add=False,
                 ):

        lexicon = load_words(lexicon)
        word_dict = create_word_dict(lexicon)

        tokens_dict = dict_reader(tokens_dict, eos=sos)
        if blk not in tokens_dict:
            tokens_dict[blk] = 0
        if sil:
            self.silence = tokens_dict[sil]
        else:
            self.silence = tokens_dict[blk]

        self.blank = tokens_dict[blk]
        lm = KenLM(kenlm_model, word_dict)
        trie = Trie(len(tokens_dict), self.silence)

        start_state = lm.start(False)
        for i, (word, spellings) in enumerate(lexicon.items()):
            word_idx = word_dict.get_index(word)
            _, score = lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idxs = [tokens_dict[token] if token in tokens_dict else tokens_dict[unk] for token in spelling]
                trie.insert(spelling_idxs, word_idx, score)
        trie.smear(SmearingMode.MAX)

        options = LexiconDecoderOptions(
            beam_size, # number of top hypothesis to preserve at each decoding step
            len(tokens_dict), # restrict number of tokens by top am scores (if you have a huge token set)
            beam_threshold=beam_threshold, # preserve a hypothesis only if its score is not far away from the current best hypothesis score
            lm_weight=lm_weight, # language model weight for LM score
            word_score=word_score, # score for words appearance in the transcription
            unk_score= unk_score, # score for unknown word appearance in the transcription
            sil_score=sil_score, # score for silence appearance in the transcription
            log_add=log_add, # the way how to combine scores during hypotheses merging (log add operation, max)
            criterion_type=CriterionType.CTC # supports only CriterionType.ASG or CriterionType.CTC
            )

        self.blank_idx = tokens_dict[blk]
        unk_word = tokens_dict[unk]
        transitions = []
        #transitions = torch.FloatTensor(N, N).zero_()
        self.decoder = LexiconDecoder(options, trie, lm, self.silence, self.blank_idx, unk_word, transitions, False)

    def decode_problike(self, probs, do_log=False):
        T, N = probs.shape
        results = self.decoder.decode(probs.ctypes.data, T, N)        
        N_best = []
        for result in results:
            N_best.append((self.get_tokens(result.tokens), result.score))
        return N_best

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))     
        idxs = filter(lambda x: x != self.blank, idxs)

        return list(idxs)