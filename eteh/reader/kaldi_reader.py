# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
from kaldi_io import read_mat


def Read_KaldiFeat(file_path):
    return read_mat(file_path)


def try_read_line(file_io):
    if file_io is not None:
        line = file_io.readline().strip()
        return line.split(' ')[0], " ".join(line.split(' ')[1:])
    else:
        return "Empty", "0"

def try_read_mat(mat_io, length_io=None, dim=40):
    mat_name, mat_path = try_read_line(mat_io)
    length_name, length_value = try_read_line(length_io)
    assert mat_name == length_name
    if len(mat_name) > 0:
        return mat_name, mat_path, int(length_value), dim
    else:
        return "", "", 0, dim

def try_read_wav(wav_io, length_io=None):
    wav_name, wav_path = try_read_line(wav_io)
    length_name, length_value = try_read_line(length_io)
    assert wav_name == length_name
    if len(wav_name) > 0:
        return wav_name, wav_path, float(length_value)
    else:
        return "", "", 0


def try_read_text(file_io, tokenizer=None):
    text_name, text_text = try_read_line(file_io)
    if tokenizer is None:
        return text_name, text_text
    else:
        return text_name, tokenizer.encode(text_text)

def try_read_list(file_io):
    list_name, list_id = try_read_line(file_io)
    list_id = list_id.split(' ')
    if len(list_name) > 0:
        list_id = [int(id) for id in list_id]
        return list_name, list_id
    else:
        return "", [0]

def try_read_align(file_io):
    if file_io is not None:
        line = file_io.readline().strip()
        if len(line) <= 1:
            return "Empty", [0], [0], [0]
        list_name, list_id, list_beg, list_end = line.split(" ")
        list_id, list_beg, list_end = [int(id) for id in list_id.split(',')], [int(id) for id in list_beg.split(',')], [int(id) for id in list_end.split(',')]
        return list_name, list_id, list_beg, list_end
    else:
        return "Empty", [0], [0], [0]

def Read_KaldiDict(data_dict={}, tokenizer=None):
    for k, v in data_dict.items():
        if k == "name":
            continue
        if v["type"] == "mat":
            v["mat_io"] = open(v["path"], 'r', encoding='utf-8')
            v["length_io"] = open(
                v["length"], 'r', encoding='utf-8') if "length" in v else None
            v["dim"] = int(v["dim"]) if "dim" in v else 1
        elif v["type"] == "wav":
            v["wav_io"] = open(v["path"], 'r', encoding='utf-8')
            v["length_io"] = open(
                v["length"], 'r', encoding='utf-8') if "length" in v else None
            v["dim"] = int(v["dim"]) if "dim" in v else 1   
            v["sr"] = int(v["sr"]) if "sr" in v else 8000         
        elif v["type"] == "label_text":
            v["text_io"] = open(v["path"], 'r', encoding='utf-8')
            v["dim"] = int(v["dim"]) if "dim" in v else 1
        elif v["type"] == "label_list":
            v["list_io"] = open(v["path"], 'r', encoding='utf-8')
            v["dim"] = int(v["dim"]) if "dim" in v else 1
        elif v["type"] == "label_align":
            v["align_io"] = open(v["path"], 'r', encoding='utf-8')
            v["dim"] = int(v["dim"]) if "dim" in v else 1
        else:
            raise RuntimeError("unknown data type" + v["type"])

    while True:
        kaldi_dict = {"name": ""}
        for k, v in data_dict.items():
            i = 0
            if k == "name":
                continue
            kaldi_dict[k] = {"type": v["type"]}
            if v["type"] == "mat":
                name, kaldi_dict[k]["mat_path"], kaldi_dict[k]["len"], kaldi_dict[k]["dim"] = try_read_mat(
                    v["mat_io"], v["length_io"], v["dim"])
            elif v["type"] == "wav":
                name, kaldi_dict[k]["wav_path"], kaldi_dict[k]["len"] = try_read_wav(
                    v["wav_io"], v["length_io"])                
                kaldi_dict[k]["sr"], kaldi_dict[k]["dim"] = v["sr"], v["dim"]
            elif v["type"] == "label_text":
                name, kaldi_dict[k]["text"] = try_read_text(v["text_io"], tokenizer)
                kaldi_dict[k]["len"], kaldi_dict[k]["dim"] =  len(kaldi_dict[k]["text"]), v["dim"]
            elif v["type"] == "label_list":
                name, kaldi_dict[k]["list"] = try_read_list(v["list_io"])
                kaldi_dict[k]["len"], kaldi_dict[k]["dim"] =  len(kaldi_dict[k]["list"]), v["dim"]
            elif v["type"] == "label_align":
                name, kaldi_dict[k]["list"], kaldi_dict[k]["beg"], kaldi_dict[k]["end"] = try_read_align(v["align_io"])                   
                kaldi_dict[k]["len"], kaldi_dict[k]["dim"] =  kaldi_dict[k]["end"][-1] + 1, v["dim"]  
            else:
                raise RuntimeError("unknown data type" + v["type"])
            #验证读的是同一条数据的信息
            if i > 0:
                assert kaldi_dict["name"] == name
            else:
                kaldi_dict["name"] = name
        if kaldi_dict["name"] == "" or kaldi_dict["name"] == "Empty":
            break
        else:
            yield kaldi_dict
    for k, v in data_dict.items():
        if k == "name":
            continue
        for _, fio in v.items():
            try:
                fio_close()
            except:
                pass

# def Read_KaldiDict(data_dict={}):
#     file_dict = {}
#     if "wav" in data_dict:
#         file_dict["wav"] = open(data_dict["wav"], 'r', encoding='utf-8')
#     if "feats" in data_dict:
#         file_dict["feats"] = open(data_dict["feats"], 'r', encoding='utf-8')
#     if "utt2spk" in data_dict:
#         file_dict["utt2spk"] = open(data_dict["utt2spk"], 'r', encoding='utf-8')
#     if "spk2utt" in data_dict:
#         file_dict["spk2utt"] = open(data_dict["spk2utt"], 'r', encoding='utf-8')
#     if "num_frs" in data_dict:
#         file_dict["num_frs"] = open(data_dict["num_frs"], 'r', encoding='utf-8')
#     if "text" in data_dict:
#         file_dict["text"] = open(data_dict["text"], 'r', encoding='utf-8')
#     while True:
#         read_dict = {"name":"", "wav":"","feats":"","utt2spk":"","spk2utt":"","num_frs":0,"text":""}
#         for k, v in file_dict.items():
#             i = 0
#             name, item = try_read_line(v)
#             if name == "Empty":
#                 break
#             read_dict[k] = item
#             if i > 0:
#                 assert read_dict["name"] == name
#             else:
#                 read_dict["name"] = name
#             i += 1
#         if read_dict["name"] == "":
#             break
#         else:
#             yield read_dict
#     for k in file_dict:
#         file_dict[k].close()

# from kaldi.util.table import SequentialMatrixReader
# import numpy as np

# def Read_KaldiFeat(file_path,file_type="scp",delta = False):
#     s1 = "%s:%s"%(file_type,file_path)
#     keys = []
#     datas = []
#     with SequentialMatrixReader(s1) as f:
#         for key,data in f:
#             keys.append(key)
#             data = data.numpy()
#             if delta:
#                 delta1 = np.diff(data,n=1,axis=0)
#                 delta2 = np.diff(data,n=2,axis=0)
#                 t = np.shape(delta2)[0]
#                 data = np.concatenate([data[:t],delta1[:t],delta2],axis=-1)
#             datas.append(data)
#     return keys,datas
