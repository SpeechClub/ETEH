# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)


import numpy as np
from eteh.reader.txtfile_reader import dict_reader


class Data(object):
    '''Basic Data, use numpy to store the tensor of the data

    :param str name: the names of the data
    :param list shape:  the shape of the stored tensor
    :param str source:  the path of the real value of the data, use Kaldi style data format
                        eg. /data/deltafalse/feats.1.ark:48
    :param numpy.array value:   the real value of the data. If source is given and load is True, this param will be ignore
    :param bool load:   if true, the value will be loaded from the source when Data object is created, else the 
                        value will be loaded when getvalue function is called 
    '''
    KALDI_TYPE = 'kaldi'
    SOUND_HDF5_TYPE = 'sound.hdf5'
    WAV_TYPE = 'wave'

    def __init__(self, name="", shape=None, source=None, value=None, load=False, file_type=KALDI_TYPE):
        self.name = name
        self.shape = shape
        self.source = source
        self.file_type = file_type
        if load and source is not None:
            value, shape = self.tryread(source)
        self.value = value
        self.shape = shape

    def getvalue(self):
        '''Get the value of the Data. 
            If the value is None, it will be loaded from self.source
        '''
        if self.value is None:
            value, self.shape = self.tryread(self.source)
        else:
            value = self.value
        return value

    def setValue(self, value):
        self.value = value
        shape =  np.shape(self.value)
        if len(shape) == 1: #[T] extend to [T,1]
            self.value = np.expand_dims(value, axis=-1)
        self.shape = np.shape(self.value)

    def tryread(self, source):
        try:
            if self.file_type == Data.KALDI_TYPE:
                from eteh.reader.kaldi_reader import Read_KaldiFeat
                value = Read_KaldiFeat(self.source)
            elif self.file_type == Data.SOUND_HDF5_TYPE:
                from eteh.reader.h5py_reader import Read_SoundHDF5
                value = Read_SoundHDF5(self.source)
            elif self.file_type == Data.WAV_TYPE:
                from eteh.reader.wav_reader import wav_reader_soundfile
                value, _ = wav_reader_soundfile(self.source)
            else:
                raise RuntimeError(
                    "Unkown filetype %s, please check the file type in json" % self.file_type)
        except Exception as e:
            print("read file" + self.source + "Error:\n" + str(e))
            exit()
        shape = np.shape(value)
        if len(shape) == 1: #[T] extend to [T,1]
            value = np.expand_dims(value, axis=-1)
            shape = np.shape(value)
        return value, shape

    @staticmethod
    def MergeData(data_list, pad_value=0):
        '''Merge a list of data, return a tensor to represent the data list

        :param list data_list:  a list of Data. The Data must have the same shape at -1 and the number of the Data  axis is 2
        :param float pad_value: the value will be padded at the axis 1 
        :return: a tensor conatin all data in the list (N,T,D) N is the length of the data_list T is the max length(shape[0]) of the Data 
                 D is the dim(shape[1]) of the Data 
        :rtype: float ndarray
        '''
        if data_list[0] == 0:
            return "Empty", np.zeros(1), [0]
        num_frs = [dat.shape[0] for dat in data_list]
        max_t = max(num_frs)
        B, T= len(data_list), max_t 
        D = 1 if len(data_list[0].shape) == 1 else data_list[0].shape[-1]
        shape = (B, T, D)
        input_mat = pad_value * np.ones(shape, dtype=np.float32)
        for e, inp in enumerate(data_list):
            input_mat[e, :inp.shape[0], :] = inp.getvalue()
        utt_id = [dat.name for dat in data_list]
        return utt_id, input_mat, num_frs


class Label(object):
    '''Basic Label, use numpy to store the tensor of the label

    :param str name: the names of the label
    :param str label_type:  the type of the label. Choose from ['idx','one hot','value']
                            idx type will be store as a tensor with shape [1]
    :param ndarray label: the value of the Label
    :param int dim: the dim of the Label. Not be used when label_type is idx 
    '''

    def __init__(self, name="", label_type="idx", label=[0], dim=1):
        self.name = name
        assert label_type in ["idx", "one_hot", "value"]
        self.label_type = label_type
        self.label = label
        self.dim = dim
        if label_type == "idx":
            self.dim = 1

    def __len__(self):
        return len(self.label)

    @staticmethod
    def MergeLabel(label_list, ignore=-1):
        '''Merge a list of label, return a tensor to represent the label list

        :param list label_list:  a list of Label
        :param float pad_value: the value will be padded at the axis 1 
        :return: a tensor conatin all data in the list (N,T,D) N is the length of the data_list T is the max length(len(label)) of the Label 
                 D is the dim of the Label 
        :rtype: float ndarray
        '''
        if label_list[0] == 0:
            return np.zeros(1), [0]
        num_frs = [len(l) for l in label_list]
        max_t = max(num_frs)
        shape = (len(label_list), max_t, label_list[0].dim)
        input_mat = ignore*np.ones(shape, dtype=np.float32)
        for e, inp in enumerate(label_list):
            input_mat[e, :len(inp), :] = np.expand_dims(inp.label, axis=-1)
        return input_mat, num_frs


class Dict(dict):
    def __init__(self, name="", default_key='<unk>', eos_key='<eos>', source={}, filepath=None):
        dict.__init__(source)
        self.name = name
        self.default_key = default_key
        self.eos_key = eos_key
        if filepath is not None:
            self.load_file(filepath)

    def load_file(self, filepath):
        world_dict = dict_reader(filepath, eos=self.eos_key)
        self.update(world_dict)

    def __getitem__(self, i):
        if i in self:
            return self.get(i)
        else:
            return self.get(self.default_key)

class Kaldi_Data(object):
    def __init__(self, name="", tokenizer=None, kaldi_dict={}):
        self.name = name
        self.tokenizer = tokenizer

        self.kaldi_dict = kaldi_dict        

        self.i_len = 0
        self.i_dim = 1

        self.o_len = 0
        self.o_dim = 1
        # self.feat_data = None
        # self.wav_data = None
        # self.text_label = None

    def get_by_key(self, key, frontend=None):
        if key not in self.kaldi_dict:
            return 0   
        key_type = self.kaldi_dict[key]["type"]
        if key_type == "mat":
            return self.get_mat_data(key, frontend)
        if key_type == "wav":
            return self.get_wav_data(key, frontend)
        if key_type == "label_text":
            return self.get_text_label(key, frontend)
        if key_type == "label_list":
            return self.get_list_label(key, frontend)
        if key_type == "label_align":
            return self.get_align_label(key, frontend)
        else:
            return 0

    def get_mat_data(self, mat_key="feats", frontend=None):
        if mat_key not in self.kaldi_dict:
            return 0
        mat_data = self.kaldi_dict[mat_key]
        feat_data = Data(self.name, shape=[mat_data["len"], mat_data["dim"]], 
                            source=mat_data["mat_path"], load=True, file_type=Data.KALDI_TYPE)
        if frontend is not None:
            feat_data = frontend.transfer_Data_by_key(feat_data, mat_key)
        #这个修改只是为了方便之后的排序，实际并不会影响之后的操作
        mat_data["len"], mat_data["dim"] = feat_data.shape[0], feat_data.shape[1] 
        return feat_data

    def get_wav_data(self, wav_key="wav", frontend=None):
        if wav_key not in self.kaldi_dict:
            return 0
        wav_data = self.kaldi_dict[wav_key]
        wav_Data = Data(self.name, shape=[wav_data["len"], wav_data["dim"]],
                            source=wav_data["wav_path"], load=True, file_type=Data.WAV_TYPE)
        if frontend is not None:
            wav_Data = frontend.transfer_Data_by_key(wav_Data, wav_key)
        wav_data["len"], wav_data["dim"] = wav_Data.shape[0], wav_Data.shape[1]
        return wav_Data

    def get_text_label(self, text_key="text", frontend=None):
        if text_key not in self.kaldi_dict:
            return 0
        text_data=self.kaldi_dict[text_key]
        if isinstance(text_data["text"], str) and self.tokenizer is not None:
            ids = self.tokenizer.encode(text_data["text"])
        elif isinstance(text_data["text"], list):
            ids = text_data["text"]
        else:
            return 0
        label = Label(self.name, label=ids)
        if frontend is not None:
            label = frontend.transfer_Label_by_key(label, text_key)
        text_data["len"], text_data["dim"] = len(label), self.tokenizer.get_dictsize()
        return label

    def get_list_label(self, list_key="list", frontend=None):
        if list_key not in self.kaldi_dict:
            return 0
        list_data=self.kaldi_dict[list_key]
        ids = list_data["list"]
        label = Label(self.name, label=ids)
        if frontend is not None:
            label = frontend.transfer_Label_by_key(label, list_key)
        list_data["len"], list_data["dim"] = len(label), 1
        return label

    def get_align_label(self, list_key="align", frontend=None):
        if list_key not in self.kaldi_dict:
            return 0
        list_data=self.kaldi_dict[list_key]
        ids, begs, ends = list_data["list"], list_data["beg"], list_data["end"]
        # ex_ids = [0] * list_data["len"]
        ex_ids = np.zeros(list_data["len"])
        for i in range(len(ids)):
            ex_ids[begs[i]:ends[i]] = ids[i]
        label = Label(self.name, label=ex_ids)
        if frontend is not None:
            label = frontend.transfer_Label_by_key(label, list_key)
        list_data["len"], list_data["dim"] = len(label), 1
        return label
        
    def get_key_len_dim(self, main_key):
        if main_key not in self.kaldi_dict:
            return 0, 0
        else:
            return self.kaldi_dict[main_key]["len"], self.kaldi_dict[main_key]["dim"]

    @staticmethod
    def MergeKaldi(kalid_data_list, frontend=None, ignore=-1, pad_value=0):
        '''Merge a list of label, return a tensor to represent the label list

        :param list label_list:  a list of Label
        :param float pad_value: the value will be padded at the axis 1 
        :return: a tensor conatin all data in the list (N,T,D) N is the length of the data_list T is the max length(len(label)) of the Label 
                 D is the dim of the Label 
        :rtype: float ndarray
        '''
        output_dict = {"name_list":[]}
        example_dict = kalid_data_list[0].kaldi_dict
        type_list, key_list = [], []
        for k in example_dict:
            if k == "name":
                continue
            type_list.append(example_dict[k]["type"])
            key_list.append(k)
            output_dict[k] = {"type":example_dict[k]["type"]}
        pakle_list = []
        for kalid_data in kalid_data_list:
            name = kalid_data.name
            data_list = [kalid_data.get_by_key(k, frontend) for k in key_list]
            pakle_list.append(tuple([name]+data_list))
        zip_pakle = zip(*pakle_list)
        for i, pakle in enumerate(zip_pakle, -1):
            if i ==-1:
                 output_dict["name_list"] = pakle
            else:
                if output_dict[key_list[i]]["type"] in ["mat","wav"]:
                    _, output_dict[key_list[i]]["data"], output_dict[key_list[i]]["len"] = Data.MergeData(pakle, pad_value)
                elif output_dict[key_list[i]]["type"] in ["label_text", "label_list", "label_align"]:
                    output_dict[key_list[i]]["data"], output_dict[key_list[i]]["len"] = Label.MergeLabel(pakle, ignore)
                # elif output_dict[key_list[i]]["type"] == "label_list":
                #     output_dict[key_list[i]]["data"], output_dict[key_list[i]]["len"] = Label.MergeLabel(pakle, ignore)
                else:
                    raise RuntimeError("Unknown data type" + output_dict[key_list[i]]["type"])
        return output_dict
# '''
# 🤗
# '''
