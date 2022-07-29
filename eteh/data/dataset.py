# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import torch.utils.data as data
import torch
import json
import numpy as np
import math
import random
import pickle
from .data import Data, Label, Dict, Kaldi_Data
from .datafrontend import GetFrontEnd, KaldiFrontend
from eteh.ops.data_ops import my_cat
from eteh.reader.kaldi_reader import Read_KaldiFeat, Read_KaldiDict
from eteh.reader.txtfile_reader import txt_reader
from eteh.utils.json_utils import make_batchset, batchfy_by_seq, batchfy_by_bin, batchfy_by_frame, CleanData
from eteh.utils.tokenizer import DictTokenizer
from eteh.utils.dynamic_import import dynamic_import

class BaseDataSet(data.Dataset):
    '''Basic DataSet inherit from torch.data.Dataset, self.train_set is a list to store Data, List[List[Tuple(Data,Label)]]
        length of the DataSet is the length of self.train_set. When use DataLoader to get Data, return a List of Tuple(Data,Lable)
    :param str name: the name the Dataset.
    '''

    def __init__(self, name='None'):
        self.name = name
        self.train_set = []
        self.distributed = False

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        item = self.train_set[index]
        return item

    def load_ark_file(self, source_paths, data_path=""):
        '''Read Data from file to story in the self.train_set. Overwrite this function to read different type of file.
            Must be called before used
        :param list source_paths: a list of the relative position of the data source files
        :param str data_path: the root path of the file.
        '''
        raise NotImplementedError()

    def to_Distributed(self, rank_size, rank_id):
        '''When used distributed dataset, Every DataSet on one process only need to maintain a part of data
        so drop the redundant data to reduce memory.for rank k in all N ranks, only train_set[k::N] will 
        be remain, to guarantee the number of data is equal between workers, the data in the center of 
        the dataset will be padded
        :param int rank_size: the total worker
        :param int rank_id: the id of this worker
        '''
        if self.distributed:
            return
        self.distributed = True
        dist_len = math.ceil(len(self.train_set)/rank_size)
        pad_num = rank_size*dist_len
        half = len(self.train_set)//2
        self.train_set = self.train_set[:half] + \
            self.train_set[half - pad_num:]
        self.train_set = self.train_set[rank_id::rank_size]
        assert len(self.train_set) == dist_len

    @staticmethod
    def SaveDataSet(dataset, path, model="wb"):
        with open(path, model) as f:
            pickle.dump(dataset, f)

    @staticmethod
    def LoadDataSet(path, model="rb"):
        with open(path, model) as f:
            dataset = pickle.load(f)
        return dataset

    def MergeBatch(self, batch, ignore_id=-1):
        return batch        


class JsonDataPacker(object):
    def __init__(self, frotend, pack_config={}):
        self.json_config = pack_config
        self.inname='feat'
        self.outname='tokenid'
        self.frotend = frotend
        self.ignore_id = -1

    def read_input(self, json_item):
        x = self.jsonInputData(json_item)
        return x

    def read_output(self, json_item):
        name = json_item[0]
        label = json_item[1]['output'][0][self.outname].split(' ')
        label = [int(c) for c in label]
        dim = json_item[1]['output'][0]['shape'][-1]
        y = Label(name, "idx", label, dim)
        return y

    # def read_output(self, json_item):
    #     name = json_item[0]
    #     y = []
    #     for i in range(len(json_item[1]['output'])):
    #         label = json_item[1]['output'][i][self.outname].split(' ')
    #         label = [int(c) for c in label]
    #         dim = json_item[1]['output'][i]['shape'][-1]
    #         y_u = Label(name, "idx", label, dim)
    #         y.append(y_u)
    #     return y

    def read_other(self, json_item):
        return 0

    def unpack_input(self, px):
        x, x_len = [], []
        data_channel = zip(*px)
        for data_u in data_channel:
            utt_ids, x_u, x_len_u = Data.MergeData(data_u)
            x_u, x_len_u = self.frotend.transfer_data(x_u, x_len_u)
            x.append(torch.from_numpy(x_u))
            x_len.append(torch.LongTensor(x_len_u))
        return utt_ids, x, x_len

    def unpack_output(self, py):
        y, y_len = Label.MergeLabel(py, self.ignore_id)
        y, y_len = self.frotend.transfer_label(y, y_len)
        y, y_len = torch.from_numpy(y), torch.LongTensor(y_len)
        return y, y_len

    # def unpack_output(self, py):
    #     y, y_len = [], []
    #     label_channel = zip(*py)
    #     for label_u in label_channel:
    #         y_u, y_len_u = Label.MergeLabel(label_u, self.ignore_id)
    #         y_u, y_len_u = self.frotend.transfer_label(y_u, y_len_u)
    #         # y_u, y_len_u = torch.from_numpy(y_u), torch.LongTensor(y_len_u)
    #         y.append(torch.from_numpy(y_u))
    #         y_len.append(torch.LongTensor(y_len_u))
    #     return y, y_len

    def unpack_other(self, pz):
        return 0

    def packData(self, json_item):
        input_x = self.read_input(json_item)
        output_y = self.read_output(json_item)
        other_z = self.read_other(json_item)
        return (input_x, output_y, other_z)

    def unpackData(self, batch):
        input_px, output_py, other_pz = zip(*batch)
        utt_ids, x, x_len = self.unpack_input(input_px)
        y, y_len = self.unpack_output(output_py)
        z = self.unpack_other(other_pz)
        data = {
            "utt_ids": utt_ids,
            "x": x,
            "y": y,
            "x_len": x_len,
            "y_len": y_len,
            "other": z, 
        }
        return data

    def jsonInputData(self, m, load=False):
        name = m[0]
        x = []
        for i in range(len(m[1]['input'])):
            channel_i = m[1]['input'][i]
            source, shape = channel_i[self.inname], channel_i['shape']
            data_type = channel_i['filetype'] if 'filetype' in channel_i else Data.KALDI_TYPE
            xu = Data(name, shape, source, load=load, file_type = data_type)
            x.append(xu)
        return x

class TimeAliJsonPacker(JsonDataPacker):
    def __init__(self, frotend, pack_config={}, begname='tokenstt', endname='tokenend'):
        super(TimeAliJsonPacker, self).__init__(frotend, pack_config)
        self.begname = begname
        self.endname = endname

    def read_other(self, json_item):
        label_beg = json_item[1]['output'][0][self.begname].split(' ')
        label_beg = [int(x) for x in label_beg]
        y_beg = Label("", "idx", label_beg, 1)
        label_end = json_item[1]['output'][0][self.endname].split(' ')
        label_end = [int(x) for x in label_end]
        y_end = Label("", "idx", label_end, 1)
        return (y_beg, y_end)

    def unpack_other(self, pz):
        label_beg, label_end = zip(*pz)
        y_beg, _ = Label.MergeLabel(label_beg, self.ignore_id)
        y_end, _ = Label.MergeLabel(label_end, self.ignore_id)
        y_beg, y_end = torch.LongTensor(y_beg), torch.LongTensor(y_end)
        return {"y_beg": y_beg, "y_end": y_end}

class JsonDataSet(BaseDataSet):
    '''DataSet generate from json file,should be rewrite

    '''
    def __init__(self, *jarg, name='Json', inname='feat', outname='tokenid', 
                    load=False, jconfig={}, front_type=None, trans_config={}, packer_type=None, packer_config={}):
        super(JsonDataSet, self).__init__(name=name)
        self.args = jarg
        self.kwarg = jconfig
        self.load = load
        self.inname = inname
        self.outname = outname
        self.front_type=front_type
        self.frotend = GetFrontEnd(trans_config=trans_config, frontend=self.front_type)
        self.packer = JsonDataSet.GetJsonPacker(self.frotend, packer_type, packer_config)

    @staticmethod
    def GetJsonPacker(frotend, packer_type=None, packer_config={}):
        if packer_type is None:
            return JsonDataPacker(frotend, packer_config)
        else:
            cls = dynamic_import(packer_type)
            return cls(frotend, packer_config)

    def load_ark_file(self, source_paths, data_path=""):
        dict_all = {}
        for source in source_paths:            
            if "json" in source:
                jfile = data_path+source["json"]
                print("Warning: json key will be disgarded in the future, use path instead")
            else:
                jfile = data_path+source["path"]
            '''
                jfile is a dict can be read from json file,egs:
                with open('data.json', 'rb') as f:
                    train_json = json.load(f)['utts']
                jtrams can convert json data to real data
            '''
            with open(jfile, 'r', encoding="utf-8") as f:
                train_json = json.load(f)['utts']
            dict_all.update(train_json)
        train_set = make_batchset(dict_all, *self.args, **self.kwarg)
        self.train_set += self.JsonItem2Data(train_set, self.load)

    def JsonItem2Data(self, jbanchs, load=False):
        data_list = []
        for jbanch in jbanchs:
            banch = []
            for m in jbanch:
                pk = self.packer.packData(m)
                banch.append(pk)
            data_list.append(banch)
        return data_list

    def MergeBatch(self, batch, ignore_id = -1):
        data = self.packer.unpackData(batch)
        return data

class DomainJsonDataSet(JsonDataSet):
    def __init__(self, *jarg, name='Json', inname='feat', outname='tokenid', domname='domain',
                    load=False, jconfig={}, front_type=None, trans_config={}):
        super(DomainJsonDataSet, self).__init__(*jarg, name=name, inname=inname, outname=outname,
                                                load=load, jconfig=jconfig, front_type=front_type, trans_config=trans_config)
        self.domname = domname
        self.domdim = self.domname + 'dim'

    def JsonItem2Data(self, jbanchs, load=False):
        data_list = []
        for jbanch in jbanchs:
            banch = []
            for m in jbanch:
                name = m[0]
                x = self.jsonInputData(m, load)
                label = m[1]['output'][0][self.outname].split(' ')
                label = [int(c) for c in label]
                dim = m[1]['output'][0]['shape'][-1]
                y = Label(name, "idx", label, dim)
                # parse domain tag
                domdim = m[1]['input'][0][self.domdim] + 1
                tag = [0] * domdim
                tag[int(m[1]['input'][0][self.domname])] = 1
                z = Label(name, "idx", tag, domdim)
                banch.append((x, y, z))
            data_list.append(banch)
        return data_list

    def MergeBatch(self, batch, ignore_id = -1):
        data_l, label_l, tag_l = zip(*batch)
        data_channel = zip(*data_l)
        x, x_len = [], []
        for data_u in data_channel:
            utt_ids, x_u, x_len_u = Data.MergeData(data_u)
            x_u, x_len_u = self.frotend.transfer_data(x_u, x_len_u)
            x.append(torch.from_numpy(x_u))
            x_len.append(torch.LongTensor(x_len_u))
        y, y_len = Label.MergeLabel(label_l, ignore_id)
        y, y_len = self.frotend.transfer_label(y, y_len)
        y, y_len = torch.from_numpy(y), torch.LongTensor(y_len)
        tag, _ = Label.MergeLabel(tag_l, 0)
        tag = torch.from_numpy(tag).squeeze(-1)
        data = {
            "utt_ids": utt_ids,
            "x": x,
            "y": y,
            "x_len": x_len,
            "y_len": y_len,
            "tag": tag
        }
        return data

class AlignJsonDataSet(JsonDataSet):
    def __init__(self, *jarg, name='Json', inname='feat', outname='tokenid', begname='tokenstt', endname='tokenend',
                    load=False, jconfig={}, front_type=None, trans_config={}):
        super(AlignJsonDataSet, self).__init__(*jarg, name=name, inname=inname, outname=outname,
                                                 load=load, jconfig=jconfig, front_type=front_type, trans_config=trans_config)
        self.begname = begname
        self.endname = endname

    def JsonItem2Data(self, jbanchs, load=False):
        data_list = []
        for jbanch in jbanchs:
            banch = []
            for m in jbanch:
                name = m[0]
                x = self.jsonInputData(m, load)
                label = m[1]['output'][0][self.outname].split(' ')
                label = [int(x) for x in label]
                dim = m[1]['output'][0]['shape'][-1]
                y = Label(name, "idx", label, dim)
                label_beg = m[1]['output'][0][self.begname].split(' ')
                label_beg = [int(x) for x in label_beg]
                y_beg = Label(name, "idx", label_beg, 1)
                label_end = m[1]['output'][0][self.endname].split(' ')
                label_end = [int(x) for x in label_end]
                y_end = Label(name, "idx", label_end, 1)
                banch.append((x, y, y_beg, y_end))
            data_list.append(banch)
        return data_list

    def jsonInputData(self, m, load):
        name = m[0]
        x = []
        for i in range(len(m[1]['input'])):
            channel_i = m[1]['input'][i]
            source, shape = channel_i[self.inname], channel_i['shape']
            data_type = channel_i['filetype'] if 'filetype' in channel_i else Data.KALDI_TYPE
            xu = Data(name, shape, source, load=load, file_type = data_type)
            x.append(xu)
        return x

    def MergeBatch(self, batch, ignore_id = -1):
        data_l, label_l, label_beg, label_end = zip(*batch)
        data_channel = zip(*data_l)
        x, x_len = [], []
        for data_u in data_channel:
            utt_ids, x_u, x_len_u = Data.MergeData(data_u)
            x_u, x_len_u = self.frotend.transfer_data(x_u, x_len_u)
            x.append(torch.from_numpy(x_u))
            x_len.append(torch.LongTensor(x_len_u))
        y, y_len = Label.MergeLabel(label_l, ignore_id)
        y, y_len = self.frotend.transfer_label(y, y_len)
        y, y_len = torch.from_numpy(y), torch.LongTensor(y_len)
        y_beg, _ = Label.MergeLabel(label_beg, ignore_id)
        y_end, _ = Label.MergeLabel(label_end, ignore_id)
        y_beg, y_end = torch.LongTensor(y_beg), torch.LongTensor(y_end)
        data = {
            "utt_ids": utt_ids,
            "x": x,
            "y": y,
            "x_len": x_len,
            "y_len": y_len,
            "y_beg": y_beg,
            "y_end": y_end
        }
        return data

class DomainAlignJsonDataSet(JsonDataSet):
    def __init__(self, *jarg, name='Json', inname='feat', outname='tokenid', begname='tokenstt', endname='tokenend', domname='domain',
                    load=False, jconfig={}, front_type=None, trans_config={}):
        super(DomainAlignJsonDataSet, self).__init__(*jarg, name=name, inname=inname, outname=outname,
                                                load=load, jconfig=jconfig, front_type=front_type, trans_config=trans_config)
        self.begname = begname
        self.endname = endname
        self.domname = domname
        self.domdim = self.domname + 'dim'

    def JsonItem2Data(self, jbanchs, load=False):
        data_list = []
        for jbanch in jbanchs:
            banch = []
            for m in jbanch:
                name = m[0]
                x = self.jsonInputData(m, load)
                label = m[1]['output'][0][self.outname].split(' ')
                label = [int(c) for c in label]
                dim = m[1]['output'][0]['shape'][-1]
                y = Label(name, "idx", label, dim)
                # parse align
                label_beg = m[1]['output'][0][self.begname].split(' ')
                label_beg = [int(x) for x in label_beg]
                y_beg = Label(name, "idx", label_beg, 1)
                label_end = m[1]['output'][0][self.endname].split(' ')
                label_end = [int(x) for x in label_end]
                y_end = Label(name, "idx", label_end, 1)
                # parse domain tag
                domdim = m[1]['input'][0][self.domdim] + 1
                tag = [0] * domdim
                tag[int(m[1]['input'][0][self.domname])] = 1
                z = Label(name, "idx", tag, domdim)
                banch.append((x, y, y_beg, y_end, z))
            data_list.append(banch)
        return data_list

    def MergeBatch(self, batch, ignore_id = -1):
        data_l, label_l, label_beg, label_end, tag_l = zip(*batch)
        data_channel = zip(*data_l)
        x, x_len = [], []
        for data_u in data_channel:
            utt_ids, x_u, x_len_u = Data.MergeData(data_u)
            x_u, x_len_u = self.frotend.transfer_data(x_u, x_len_u)
            x.append(torch.from_numpy(x_u))
            x_len.append(torch.LongTensor(x_len_u))
        y, y_len = Label.MergeLabel(label_l, ignore_id)
        y, y_len = self.frotend.transfer_label(y, y_len)
        y, y_len = torch.from_numpy(y), torch.LongTensor(y_len)
        y_beg, _ = Label.MergeLabel(label_beg, ignore_id)
        y_end, _ = Label.MergeLabel(label_end, ignore_id)
        y_beg, y_end = torch.LongTensor(y_beg), torch.LongTensor(y_end)
        tag, _ = Label.MergeLabel(tag_l, 0)
        tag = torch.from_numpy(tag).squeeze(-1)
        data = {
            "utt_ids": utt_ids,
            "x": x,
            "y": y,
            "x_len": x_len,
            "y_len": y_len,
            "y_beg": y_beg,
            "y_end": y_end,
            "tag": tag
        }
        return data

class LabelSeqDataSet(BaseDataSet):
    def __init__(self, name='LabelSeq', label_dict="", 
                    batch_size=32, batch_bin=0, max_len=150, min_len=1, max_size=1000, 
                    sort= False, unk='<unk>'
                ):
        super(LabelSeqDataSet, self).__init__(name=name)
        if isinstance(label_dict, dict):
            self.label_dict = Dict(name+'dict', source=label_dict, default_key=unk)
        elif isinstance(label_dict, str):
            self.label_dict = Dict(name+'dict', filepath=label_dict, default_key=unk)
        else:
            self.label_dict = None
        self.batch_size = batch_size
        self.batch_bin = batch_bin
        self.max_len = max_len
        self.min_len = min_len
        self.sort = sort
        self.max_size = max_size

    def load_ark_file(self, source_paths, data_path=""):
        label_lines = []
        for source in source_paths:
            if "label" in source:
                lfile = data_path+source["label"]
                print("Warning: label key will be disgarded in the future, use path instead")
            else:
                lfile = data_path+source["path"]
            label_l = txt_reader(lfile, id_dict=self.label_dict)
            label_lines.extend(label_l)
        self.train_set = self.make_batch(label_lines, self.batch_size, self.batch_bin, self.max_len, self.min_len)
        
    def make_batch(self, label_lines, batch_size, batch_bin, max_len, min_len):
        label_list = []
        if self.sort:
            label_lines = sorted(label_lines, key=len, reverse=False)
        for i in range(len(label_lines)):
            if len(label_lines[i]) > max_len or len(label_lines[i]) < min_len:
                continue
            y = Label(label=label_lines[i])
            label_list.append(y)
        if batch_bin<=0 and batch_size>0:
            batch_list = [label_list[i:batch_size+i] for i in range(0,len(label_list),batch_size)]
        elif batch_bin>0 and batch_size<=0:
            batch_list = []
            offset = 0
            while offset < len(label_list):
                bs = min(batch_bin // len(label_list[offset]) + 1, self.max_size)
                off_end = min(offset+bs, len(label_list))
                batch_list.append(label_list[offset:off_end])
                offset = off_end
        elif batch_bin<=0 and batch_size<=0:
            raise RuntimeError("batch_bin and batch_size should larger than 0")
        else:
            batch_list = [label_list[i:batch_size+i] for i in range(0,len(label_list),batch_size)]
            print("use batch size first")
        return batch_list

    @staticmethod
    def MergeBatch(batch, ignore_id = -1):
        label_l = batch
        y, y_len = Label.MergeLabel(label_l, ignore_id)
        data = {
            "y": torch.from_numpy(y),
            "y_len": torch.LongTensor(y_len)
        }
        return data

class KaldiDataSet(BaseDataSet):
    def __init__(self, name='Kaldi', label_dict=None, unk='<unk>', space='', eos_key='<eos>', jconfig={}, front_type=None, trans_config={}):
        super(KaldiDataSet, self).__init__(name=name)
        # self.label_dict = Dict(name+'dict', filepath=label_dict, default_key=unk)
        self.jconfig = jconfig
        self.tokenizer = DictTokenizer(label_dict, eos_key=eos_key, default_key=unk, sc=space)
        if front_type is None:
            self.frontend = KaldiFrontend(trans_config)
        else:
            self.frontend = GetFrontEnd(trans_config)

    def load_ark_file(self, source_paths, data_path=""):
        kaldi_datas = []
        for source in source_paths:
            for read_dict in Read_KaldiDict(source, self.tokenizer):
                kaldi_data = Kaldi_Data(name=read_dict["name"], tokenizer=self.tokenizer, kaldi_dict=read_dict)
                kaldi_datas.append(kaldi_data)
        self.train_set = self.make_batch(kaldi_datas, **self.jconfig)
    
    def make_batch(self, kaldi_datas, batch_size=32, max_length_in=float("inf"), max_length_out=float("inf"),
                num_batches=0, min_batch_size=1, shortest_first=False, batch_sort_key="feats", main_ikey="feats", main_okey="text", 
                swap_io=False, count="seq", batch_bins=0, batch_frames_in=0, batch_frames_out=0, batch_frames_inout=0,
                ilen_max=5000, ilen_min=17, olen_max=500, olen_min=1, down_sample=2, domain_id=None, clean_data=True):
        if batch_sort_key == "":
            pass
        elif batch_sort_key == "shuffle":
            import random
            random.shuffle(kaldi_datas)
        else:
            kaldi_datas.sort(key=lambda kaldi_data: kaldi_data.get_key_len_dim(batch_sort_key)[0], reverse=not shortest_first)
        if clean_data:
            kaldi_datas = CleanData(kaldi_datas, main_ikey, main_okey, ilen_max, ilen_min, olen_max, olen_min, down_sample, True)
        if count == "seq":
            batch_list = batchfy_by_seq(kaldi_datas, batch_size, max_length_in, max_length_out,
                                        min_batch_size=min_batch_size, shortest_first=shortest_first, kaldi_data=True,
                                        ikey=main_ikey, okey=main_okey)
        elif count == "bin":
            batch_list = batchfy_by_bin(kaldi_datas, batch_bins, num_batches=num_batches,
                                        min_batch_size=min_batch_size, shortest_first=shortest_first, kaldi_data=True,
                                        ikey=main_ikey, okey=main_okey)
        elif count == "frame":
            batch_list = batchfy_by_frame(kaldi_datas, batch_frames_in, batch_frames_out, batch_frames_inout,
                                        min_batch_size=min_batch_size, shortest_first=shortest_first, kaldi_data=True,
                                        ikey=main_ikey, okey=main_okey)
        else:
            batch_list = [kaldi_datas[i:batch_size+i] for i in range(0,len(kaldi_datas),batch_size)]
        return batch_list
    

    def MergeBatch(self, batch, ignore_id = -1):
        data = Kaldi_Data.MergeKaldi(batch, self.frontend)
        # data = self.frontend.transfer_kaldi(data)        
        return data
