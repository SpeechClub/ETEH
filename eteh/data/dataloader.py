# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import numpy as np
import operator
import torch
from .data import Data, Label
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import torch.distributed as dist
import math
import random


class StandDataLoader(DataLoader):
    '''StandDataLoader from torch.data.DataLoader. StandDataLoader can only accept BaseDataSet, in which getitem function
        will return a list of tuple. When use iter to get the data, collate_fn will be called 

    :param int epoch: the epoch of training, used it as the random seed
    :param BaseDataSet dataset: the dataset of this loader, getitem fnction must return a list of tuple(Data,Label). If distributed
    is True, the dataset must call to_distribued at first
    :param int batch_size: the number of list will be concat. So the number of batch_size*json_size item will be return
    :param bool distributed: wheather use disributed dataloader
    :param int num_workers: the number of the dataloader workers
    :param bool shuffle: wheather shuffle the data. Only the position between DataSet's item will be shuffle
    :param int ignore_id: same to torch 
    :param int timeout: same to torch 
    :param int randseed: the random seed offset
    '''
    def __init__(self, epoch, dataset, batch_size, distributed=False, num_workers=0, shuffle=True, ignore_id=-1, timeout=1000, randseed=0, num_samples=None):
        self.name = dataset.name
        self.dataset = dataset
        self.ignore_id = ignore_id
        if not distributed:
            if shuffle:
                if num_samples is None:
                    base_sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    base_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=num_samples)
            else:
                assert num_samples is None, "ERROR: SequentialSampler does not support sampling with replacement!"
                base_sampler = torch.utils.data.SequentialSampler(dataset)
            sampler = torch.utils.data.BatchSampler(
                base_sampler, batch_size, False)
            super(StandDataLoader, self).__init__(dataset,
                                                    batch_sampler=sampler,
                                                    num_workers=num_workers,
                                                    collate_fn=self.collate_fn)
        else:
            sampler = DistributedSampler(
                dataset, shuffle=shuffle, seed=randseed, num_samples=num_samples)
            sampler.set_epoch(epoch)
            super(StandDataLoader, self).__init__(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    num_workers=num_workers,
                                                    collate_fn=self.collate_fn,
                                                    drop_last=False,
                                                    timeout=timeout)
    def collate_fn(self, batch):
        #batch是一个list，大小为batchsize，每一项都是getitem的结果，getitem得到的也是list，大小为json里的batchsize
        m_bancth = []
        for b in batch:
            m_bancth += b
        return self.dataset.MergeBatch(m_bancth, self.ignore_id)



class DistributedSampler(Sampler):
    """This DistributedSampler override the DistributedSampler of torch. In torch.utils.data.DistributedSampler, all of 
        the data will be loader in the memory but only a part of the data will be used at each worker.This Sampler used 
        Distributed DataSet, so only the needed data will be loaded at each worker. This Sampler only guarantee the random 
        seed between the worker is same, so the result of the shuffle is same too 

    :param BaseDataSet dataset: the dataset of this loader, getitem fnction must return a list of tuple(Data,Label).The 
    dataset must call to_distribued at first
    :param bool shuffle: wheather shuffle the data. Only the position between DataSet's item will be shuffle 
    :param int seed: the random seed offset
    """

    def __init__(self, dataset, shuffle=True, seed=0, num_samples=None):

        self.dataset = dataset
        self.epoch = 0
        self.seed = seed
        self.num_samples = len(dataset) if num_samples is None else num_samples
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        if self.num_samples != len(self.dataset):
            if self.shuffle:
                weights = torch.tensor([1.0 for _ in range(len(self.dataset))])
                indices = torch.multinomial(weights, self.num_samples , replacement=True).tolist()
            else:
                indices = [indice % len(self.dataset) for indice in list(range(self.num_samples))]
        else:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
