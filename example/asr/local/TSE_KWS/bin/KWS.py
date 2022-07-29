#!/bin/python3

# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Runyan Yang       yangrunyan@hccl.ioa.ac.cn       (Institute of Acoustics, Chinese Academy of Science)

# export OMP_NUM_THREADS=1
import torch
import yaml
import argparse
import os
import numpy as np
from eteh.tools.config import ModelConfig
from eteh.reader.txtfile_reader import char_listreader
from eteh.models.lm.decode_lm_interface import BasicLM
from eteh.reader.kaldi_reader import Read_KaldiFeat
# eteh.tools.decode.ctc_att_KWS

def word_listreader(path):
    with open(path,'r',encoding='utf-8') as f:
        word_list = f.read().splitlines()
    from local.TSE_KWS.decode.ctc_att_KWS import Trie
    trie = Trie()
    for w in word_list:
        w = ' '.join(w.split())
        if w:
            trie.insert(w)
    return trie

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model")
    parser.add_argument("-model_config")
    parser.add_argument("-lm_model")
    parser.add_argument("-lm_config")   
    parser.add_argument("-decode_config") 
    parser.add_argument("-data_list")
    parser.add_argument("-ref_list")
    parser.add_argument("-output_file")
    parser.add_argument("-char_list",type=str, help="the number of the character")
    parser.add_argument("-keyword_list",type=str, help="keywords")
    parser.add_argument("-kws_output_file",type=str, help="keyword search output")
    parser.add_argument("-gpu", type=int, default=0, help="the index of the gpu")
    args = parser.parse_args()
   
    # if not os.path.isdir(args.exp_dir):
    #     os.makedirs(args.exp_dir)

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
        criterion_config = model_config['criterion_config']
        model_config = model_config['model_config']

    with open(args.decode_config) as f:
        decode_config = yaml.safe_load(f)


    if(model_config is not None):
        model_config = ModelConfig(model_config)
        model = model_config.generateExample()

    if args.lm_config is not None and args.lm_config != "None":
        with open(args.lm_config) as f:
            lm_config = yaml.safe_load(f)["model_config"]
        if(lm_config is not None):
            lm_config = ModelConfig(lm_config)
            lm_model = lm_config.generateExample()
        checkpoint = torch.load(
            args.lm_model, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['model']
        state_dict = {k.split('.', maxsplit=1)[1]: v for k, v in state_dict.items()}
        lm_model.load_state_dict(state_dict)
        lm_model.eval()
        lm_model = BasicLM(lm_model)
    else:
        lm_model = None

    char_list = char_listreader(args.char_list)
    assert os.path.isfile(
        args.model), "ERROR: model file {} does not exit!".format(args.model)
    checkpoint = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['model']
    state_dict = {k.split('.', maxsplit=1)[1]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if args.gpu >= 0:
        if torch.cuda.device_count() <= args.gpu:
            print('Error!gpu is not enough')
            exit()
        else:
            device = args.gpu
            model = model.cuda(device)
            if lm_model is not None:
                lm_model = lm_model.cuda(device)
    else:
        device = "cpu" 

    kwlist = word_listreader(args.keyword_list)

    tse_config = {}
    for k in ['tse_delay', 'tse_aggr_func']:
        if k in criterion_config:
            tse_config[k] = criterion_config[k]

    decode_method = decode_config["decode_method"] if "decode_method" in decode_config else "ctc_att"

    if decode_method.startswith("ctc_att"):
        from local.TSE_KWS.decode.ctc_att_KWS import CTC_ATT_Decoder_KWS
        decoder = CTC_ATT_Decoder_KWS(
                model, model_config["odim"]-1, model_config["odim"]-1,
                beam=decode_config["beam"], ctc_beam=decode_config["ctc_beam"], 
                ctc_weight=decode_config["ctc_weight"],
                rnnlm=lm_model, lm_weight=decode_config["lm_rate"],
        )
    
    elif decode_method == "ctc_bs":
        raise NotImplementedError
        from eteh.tools.decode.ctc_bs_decoder import CTC_Decoder
        decoder = CTC_Decoder(
                beam_size=decode_config["beam"], ctc_beam=decode_config["ctc_beam"], 
                sos=model_config["odim"]-1,
                rnn_lm=lm_model, lm_rate=decode_config["lm_rate"]
        )
    elif decode_method == "ctc_kenlm_lexcoin":
        raise NotImplementedError
        from eteh.tools.decode.ctc_w2l_decoder import CTC_KenLM_Decoder
        import math
        decoder = CTC_KenLM_Decoder(
                beam_size=decode_config["beam"], beam_threshold=decode_config["beam_threshold"], 
                lexicon=decode_config["lexicon"], tokens_dict=decode_config["tokens_dict"], 
                kenlm_model=decode_config["kenlm_model"],
                sos='<eos>', blk='<blank>', unk='<unk>', sil=None,
                lm_weight=decode_config["lm_weight"], word_score=decode_config["word_score"], 
                unk_score=-math.inf, sil_score=decode_config["sil_score"],
                log_add=False,
        )
    else:
        pass

    decode_file = open(args.data_list, 'r')
    output_file = open(args.output_file, 'w', encoding='utf-8')
    kws_output_file = open(args.kws_output_file, 'w', encoding='utf-8')
    if args.ref_list is not None:
        decode_ref = open(args.ref_list, 'r', encoding='utf-8')
    else:
        decode_ref = None

    for line in decode_file:
        uid, path = line.strip().split(' ')
        feats = Read_KaldiFeat(path)
        print("id", uid) # , path, feats.shape)
        if decode_ref:
            print('ref', decode_ref.readline().strip())
        if decode_method == "ctc_att":
            hypo, kws_ans = ctc_att_decode_kws(model, decoder, char_list, device, feats, kwlist, tse_config)
        elif decode_method == "ctc_att_online":
            raise NotImplementedError
            hypo = ctc_att_decode_online(model, decoder, char_list, device, feats)
        else:
            raise NotImplementedError
            hypo = ctc_bs_decode(model, decoder, char_list, device, feats)
        print('hypo', hypo)
        output_file.write(hypo+" ({})\n".format(uid))
        kws_output_file.write("/{}\n".format(uid))
        for kw_occ in kws_ans:
            kws_output_file.write(kw_occ+'\n')
            # kws_output_file.write("{}\t{}\t{}\t{}\t{}\n".format(uid,*kw_occ))

    decode_file.close()
    output_file.close()
    kws_output_file.close()
    if decode_ref: decode_ref.close()


def ctc_att_decode_kws(model, decoder, char_list, device, feats, kwlist, tse_config):
    x = torch.from_numpy(feats)
    if device != 'cpu':
        x = x.cuda(device)
    xlen = torch.LongTensor([len(x)]).to(x.device)
    with torch.no_grad():
        ans, enc_tse_trsp = decoder.decode_feat_tse(x, xlen)
        kws_ans = decoder.kws_tse(ans, enc_tse_trsp, char_list, kwlist, **tse_config)
    ans = ans[0]['yseq']
    ans = [char_list[uid] for uid in ans][1:-1]
    kws_ans = ["{}\t00:00:{}\t00:00:{}\t{}".format(kw_occ[0], 0.04*kw_occ[1], 0.04*kw_occ[2], kw_occ[3]) for kw_occ in kws_ans]
    return str.join(' ', ans), kws_ans


def ctc_att_decode_online(model, decoder, char_list, device, feats):
    x = torch.from_numpy(feats)
    if device != 'cpu':
        x = x.cuda(device)
    xlen = torch.LongTensor([len(x)]).to(x.device)
    with torch.no_grad():
        ans = decoder.decode_feat_online(x, xlen)
    ans = ans[0]['yseq']
    ans = [char_list[uid] for uid in ans][1:-1]
    return str.join(' ', ans)


def ctc_bs_decode(model, decoder, char_list, device, feats):
    x = torch.from_numpy(feats).unsqueeze(0)
    if device != 'cpu':
        x = x.cuda(device)
    xlen = torch.LongTensor([x.shape[1]]).to(x.device)
    y = torch.LongTensor([len(char_list)-1]).cuda(device)
    ylen = [1]
    with torch.no_grad():
        model.eval()
        prob = model.get_ctc_prob(x, xlen)

    prob = torch.softmax(prob, -1)[0].detach().cpu().numpy()
    
    with torch.no_grad():
        ans_list = decoder.decode_problike(prob, True)
    ans = ans_list[0][0]
    ans = [char_list[uid] for uid in ans]
    return str.join(' ', ans).replace("<eos>", "")

        
if __name__ == '__main__':
    import sys
    print(' '.join(sys.argv))
    main()
