# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

from numpy.lib.function_base import append
import torch
import yaml
import random
import threading
import numpy as np
from threading import Semaphore
from .dataset import *
from .dataloader import StandDataLoader
from eteh.component import EtehComponent

from eteh.data import dataset

class DataCenter(EtehComponent):
    '''DataCenter class is a generator that can generate DataSet or DataLoader for training,valid and recognize

    :param str data_config: path of the config yaml file for the datacenter. The yaml file contain a dictionary 
                            that consist of different data. 
                            >>>'clean_source' is a list of training data. 
                            >>>'valid_source' is a list of valid data
                            >>>other key is remain to extend
    :param bool distributed:Must be true if distributed training is used.
    :param int  rank_size:  Only needed if distributed is True. Must equal to the number of distributed worker
    :param int  rank_id:    Only needed if distributed is True. Represent the number of the worker
    :param int  rand_seed:  Random seed for the traning. Must be same for one training task
    '''

    def __init__(self, data_config, distributed=False, rank_size=0, rank_id=0, rand_seed=0):
        DataCenter.SOURCE_KEY = 'clean_source'
        DataCenter.VALID_KEY = 'valid_source'
        self.config = {}
        with open(data_config) as f:
            data = yaml.safe_load(f)
            self.source_paths = [j for i, j in data[DataCenter.SOURCE_KEY].items()]
            if DataCenter.VALID_KEY in data:
                self.valid_paths = [j for i, j in data[DataCenter.VALID_KEY].items()]
            else:
                self.valid_paths = None
        self.data_num = len(self.source_paths)
        self.pos = 0
        self.seq = list(range(self.data_num))
        self.buff = []
        self.Wmutex = Semaphore(1)
        self.Rmutex = Semaphore(0)
        self.distributed = distributed
        self.rank_id = rank_id
        self.rank_size = rank_size
        self.rand_seed = rand_seed
        self.epoch = 0
        self.refresh()

    def to_distributed(self, nodes, node_id, gpus, gpu):
        self.rank_size = nodes*gpus
        self.rank_id = node_id*gpus + gpu
        self.distributed = True

    def refresh(self, epoch=0):
        '''Refresh the datacenter to guarantee randomness between each training epoch

        :param int epoch: the epoch for the training
        '''
        self.buff.clear()
        self.pos = 0
        self.epoch = epoch
        random.seed(epoch + self.rand_seed)
        random.shuffle(self.seq)

    def GetDataLoaderIter(self, b_size=1, data_loader_threads=1, shuffle=True, data_config={}):
        '''Get the iterator of the training dataset.This function will split the training source by files instead of mix different 
            training files together. Examples
            >>>for dataloader in datacenter.GetDataLoaderIter(epoch,data_type,...):
            >>>    for data in dataloader:
            >>>        x, y = data['x'], data['y']

        :param str data_type: the type of the training data
        :param int b_size:  if json type is used. Would better to use json batch to control the batch size. So set this param to 1
        :param int data_loader_threads: the number of the thread of the dataloader
                                        #FIXME:data_loader_threads must not equal tp zero when use honvord
        :param bool shuffle: wheather shuffle the data in the dataset. Notice that the result is depend on the epoch
        :param *args,**kwargs: other parameter for DataSet
        '''
        t = threading.Thread(target=self.write_thread,
                             args=(data_config, ) )
        t.setDaemon(True)
        t.start()
        #print("loader generate begin")
        while self.pos < self.data_num:
            # waiting sub thread read data from ark file
            self.Rmutex.acquire()
            train_set = self.buff.pop()            
            self.pos = self.pos + 1
            self.Wmutex.release()
            if self.distributed and not train_set.distributed:
                train_set.to_Distributed(self.rank_size, self.rank_id)
            train_dataloader = StandDataLoader(self.epoch, train_set, b_size,
                                                   self.distributed, data_loader_threads,
                                                   shuffle=shuffle, randseed=self.rand_seed)
            if train_dataloader is not None:
                yield train_dataloader
        t.join()
        del t
        return

    def GetDataSetIter(self, data_config):
        '''Get the iterator of the training dataset.This function will split the training source by files instead of mix different 
            training files together. Examples
            >>>for dataset in datacenter.GetDataSetIter(epoch,data_type,...):
            >>>    for data in MakeLoaderFromSet(dataset):
            >>>        x, y = data['x'], data['y']

        :param str data_type: the type of the training data
        :param *args,**kwargs: other parameter for DataSet
        '''
        t = threading.Thread(target=self.write_thread,
                             args=(data_config, ))
        t.setDaemon(True)
        t.start()
        #print("loader generate begin")
        while self.pos < self.data_num:
            # waiting sub thread read data from ark file
            self.Rmutex.acquire()
            train_set = self.buff.pop()            
            self.pos = self.pos + 1
            self.Wmutex.release()
            if self.distributed:
                train_set.to_Distributed(self.rank_size, self.rank_id)
            yield train_set
        t.join()
        del t
        return

    def GetDataSet(self, source="train", data_config={}):
        '''Get one dataset for all source files

        :param str data_type: the type of the training data
        :param str source:  the source of the data."train" for training data and "valid" for valid data
        :param *args,**kwargs: other parameter for DataSet
        #NOTE: if the datacenter is distributed. The returned dataset is distributed too
        '''
        paths = self.choosepath(source)
        # data_set = self.get_dataset(**data_config)
        # data_set.load_ark_file(paths)
        data_set = self.get_dataset_and_load(paths, data_config, append_rank=False)
        if self.distributed and not data_set.distributed:
            data_set.to_Distributed(self.rank_size, self.rank_id)
        return data_set

    def GetUnDistributedDataLoader(self, b_size, data_loader_threads, source="train", shuffle=False, data_config={}):
        '''Get one dataset for all source files. The random seed i depend on the epoch when this function is called

        :param str data_type: the type of the training data
        :param int b_size:  if json type is used. Would better to use json batch to control the batch size. So set this param to 1
        :param int data_loader_threads: the number of the thread of the dataloader
                                        #FIXME:data_loader_threads must not equal tp zero when use honvord
        :param *args,**kwargs: other parameter for DataSet
        #NOTE: this function can only return a undistributed dataloader. 
        '''
        paths = self.choosepath(source)
        # data_set = self.get_dataset(data_type, *args, **kwargs)
        # data_set.load_ark_file(paths)
        dataset = self.get_dataset_and_load(paths, data_config, append_rank=False)
        dataloader = StandDataLoader(
                self.epoch, dataset, b_size, False, data_loader_threads, shuffle=shuffle, randseed=self.rand_seed)
        return dataloader

    def LoadDataLoaderBinFileIter(self, b_size=1, data_loader_threads=1, shuffle=True, source="train", append_rank=True):
        pos = 0
        paths = self.choosepath(source)
        while pos < self.data_num:
            if not append_rank:
                data_set = BaseDataSet.LoadDataSet(paths[self.seq[pos]]["path"])
            else:
                data_set = BaseDataSet.LoadDataSet(paths[self.seq[pos]]["path"] + ".{}".format(self.rank_id))
            train_dataloader = StandDataLoader(self.epoch, data_set, b_size,
                                        self.distributed, data_loader_threads,
                                        shuffle=shuffle, randseed=self.rand_seed)
            yield train_dataloader
            pos = pos + 1
        

    def MakeLoaderFromSet(self, dataset, batch_size=1, data_loader_threads=1, shuffle=True, num_samples=None):
        '''Make a DataLoader from a DataSet

        '''
        loader = StandDataLoader(self.epoch, dataset, batch_size,
                                    dataset.distributed, data_loader_threads,
                                    shuffle=shuffle, randseed=self.rand_seed, num_samples=num_samples)
        return loader

    def write_thread(self, data_config={}):
        #print("writer thread begin")
        while True:
            self.Wmutex.acquire()
            if self.pos >= self.data_num:
                #print("write thread finish")
                self.Wmutex.release()
                return
            path = self.source_paths[self.seq[self.pos]]
            data_set=self.get_dataset_and_load([path], data_config=data_config, append_rank=True)
            self.buff.append(data_set)
            self.Rmutex.release()

    def get_dataset_and_load(self, paths, data_config, append_rank=False):
        if 'type' in paths[0] and paths[0]['type']=="bin":
            assert len(paths) == 1
            if append_rank:
                data_set = BaseDataSet.LoadDataSet(paths[0]["path"] + ".{}".format(self.rank_id))
            else:
                data_set = BaseDataSet.LoadDataSet(paths[0]["path"])
        else:
            data_set = self.get_dataset(**data_config)
            if 'name' in paths[0]:
                data_set.name = paths[0]['name']
            data_set.load_ark_file(paths)
        return data_set

    def get_dataset(self, data_type, *args, **kwargs):
        if data_type == "json":
            data_set = JsonDataSet(*args, **kwargs)
        elif data_type == "text":
            data_set = LabelSeqDataSet(*args, **kwargs)
        elif data_type == "time_json":
            data_set = AlignJsonDataSet(*args, **kwargs)
        elif data_type == "domain_json":
            data_set = DomainJsonDataSet(*args, **kwargs)
        elif data_type == "dt_json":
            data_set = DomainAlignJsonDataSet(*args, **kwargs)
        elif data_type == "kaldi":
            data_set = KaldiDataSet(*args, **kwargs)
        else:
            raise ValueError("unexpected data type")
        return data_set

    def choosepath(self, source="train"):
        if source == "train":
            paths = self.source_paths
        elif source == "valid":
            paths = self.valid_paths
        else:
            paths = None
        return paths