import kenlm
import math

def char_listreader(path, sc=' ',append=True,eos='<eos>'):
    with open(path,'r',encoding='utf-8') as f:
        char_list = f.read().splitlines()
    for i in range(len(char_list)):
        if isinstance(char_list[i],str):
            char_list[i] = char_list[i].split(sc)[0]
        else:
            char_list[i] = str(char_list[i])
    char_list.insert(0,'<blank>')
    if append and eos not in char_list:
        char_list.append('<eos>')
    return char_list

class NgramLM(object):
    def __init__(self, subword_dict, word_dict, arpa_path, order=3, eos='<eos>', unk='<unk>', space='<space>'):
        self.model= kenlm.LanguageModel(arpa_path)
        self.eos = eos
        self.unk = unk
        self.space = space
        self.subword_list = char_listreader(subword_dict)
        self.word_list = char_listreader(word_dict)
        self.order = order
               
    def score_sentence(self, x):
        bos = x[0] == self.eos
        eos = x[-1] == self.eos
        x_string = self.format_string(self.id2string(x))     
        return self.model.score(x_string, bos = bos, eos = eos)
        
    def score_prefix(self, x, look_forward=False, add_unk=False):
        '''
        look forward可以用于subword建模的asr打分。
            例如"_how_are_you_fri"，直接使用score_sentence会得到"_how_are_you_<unk>_"的概率
                score_prefix(look_forward=True),得到"_how_are_you_fri*_"的概率
                score_prefix(look_forward=False),得到"_how_are_you_"的概率
        '''
        if len(x) == 1:
            return 0
        bos = x[0] == self.eos
        eos = x[-1] == self.eos
        x_string = self.format_string(self.id2string(x))        
        if x_string[-1] == ' ' or eos:    
            score = self.model.score(x_string, bos = bos, eos = eos)
        else:
            if look_forward:
                history_string = " ".join(x_string.split(" ")[:-1])
                prefix_words = self.get_prefix_word(x_string.split(" ")[-1])
                if add_unk:
                    prefix_words.append(self.unk)
                scores = []
                for word in prefix_words:
                    score_word = self.model.score(history_string + " " + word, bos = bos, eos = eos)
                    scores.append(score_word)
                score = math.log(sum([math.exp(x) for x in scores]))
            else:
                score = self.score_sentence(x[:-1])
        return score

    def score_step(self, x, look_forward=False, add_unk=False):
        word_num = min(len(x), self.order)
        x = x[-word_num:]
        bos = x[0] == self.eos
        eos = x[-1] == self.eos
        x_string = self.format_string(self.id2string(x))        
        if x_string[-1] == ' ' or eos:    
            full_score = self.model.full_scores(x_string, bos = bos, eos = eos)
            score, _, _ = list(full_score)[-1]
        else:
            if look_forward:
                history_string = " ".join(x_string.split(" ")[:-1])
                prefix_words = self.get_prefix_word(x_string.split(" ")[-1])
                if add_unk:
                    prefix_words.append(self.unk)
                scores = []
                for word in prefix_words:
                    full_score_word = self.model.full_scores(history_string + " " + word, bos = bos, eos = eos)
                    score_word, _, _ = list(full_score_word)[-1]
                    scores.append(score_word)
                score = math.log(sum([math.exp(x) for x in scores]))
            else:
                full_score = self.model.full_scores(history_string, bos = bos, eos = eos)
                score, _, _ = list(full_score)[-1]
        return score

    def id2string(self, x):
        xs = [self.subword_list[id] for id in x]
        return ''.join(xs)
        
    def format_string(self, x_string):
        return x_string.replace(self.space, ' ').replace(self.eos, '').lstrip()
        
    def get_prefix_word(self, prefix):
        prefix_word = []
        has_find = False
        for s in self.word_list:
            if s.startswith(prefix):
                prefix_word.append(s)
                has_find = True
            else:
                if has_find:
                    break
        return prefix_word