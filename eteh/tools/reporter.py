# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)

import os
from eteh.utils.utils import AverageMeter, ProgressMeter
from eteh.utils import utils


class Reporter(object):
    '''Message Reporter

    :param list name_list: the names of the variable to show
    :param list mode_list: the statistics type of variable to show, you can show average(), sum[], the latest value or count
                           if the length of mode_list is shorter than the name_list, the last mode will be expended
                           more detail refer to utils.utils.py
    :param list fmt_list:  the format of the data to be shown. egs [':.4e',':.f']
                           if the length of fmt_list is shorter than the name_list, the fmt mode will be expended
    :param str log_file:   the file to save message whem used logging function
    '''

    def __init__(self, name_list=[], mode_list=[], fmt_list=[], log_file=None):
        assert len(name_list) >= len(mode_list)
        assert len(name_list) >= len(fmt_list)
        self.name_list = name_list
        self.mode_list = mode_list
        self.fmt_list = fmt_list
        self.file = None
        self.MeterDict = {}
        self.log_file = log_file
        if log_file is not None:
            self.file = open(log_file, 'a')
        self.meterlist = []
        for i in range(len(self.name_list)):
            name = self.name_list[i]
            mode = mode_list[min(i, len(mode_list)-1)]
            fmt = fmt_list[min(i, len(fmt_list)-1)]
            self.MeterDict[self.name_list[i]] = AverageMeter(name, fmt, mode)
            self.meterlist.append(self.MeterDict[self.name_list[i]])
        self.progress = ProgressMeter(0, *self.meterlist)

    def register(self, name_list, mode_list=[utils.AVERAGE_VAL_AVG], fmt_list=[':.4e']):
        self.name_list.extend(name_list)
        self.mode_list.extend(mode_list)
        self.fmt_list.extend(mode_list)
        for i in range(len(self.name_list)):
            name = self.name_list[i]
            mode = mode_list[min(i, len(mode_list)-1)]
            fmt = fmt_list[min(i, len(fmt_list)-1)]
            self.MeterDict[self.name_list[i]] = AverageMeter(name, fmt, mode)
            self.meterlist.append(self.MeterDict[self.name_list[i]])
        self.progress = ProgressMeter(0, *self.meterlist)

    def register_key(self, name, mode=utils.AVERAGE_VAL_AVG, fmt=':.4e'):
        self.name_list.append(name)
        self.mode_list.append(mode)
        self.fmt_list.append(fmt)
        self.MeterDict[name] = AverageMeter(name, fmt, mode)
        self.meterlist.append(self.MeterDict[name])
        self.progress = ProgressMeter(0, *self.meterlist)

    def re_open(self, log_file):
        if self.file is not None:
            self.file.close()
        self.log_file = log_file
        if log_file is not None:
            self.file = open(log_file, 'a')

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def updata(self, value_dict={}):
        '''update the variables registed in this Reporter

        :param dict value_dict: a dictonary to be update
                                the keys are str in the name_list
                                the value is the reported value
        '''
        for key in value_dict:
            if key not in self.MeterDict:
                print("Warning! %s is not registed in the reporter, regist new variable " % key)
                self.register_key(key)
            self.MeterDict[key].update(value_dict[key])
            # if len(value_dict[key]) > 1:
            #     self.MeterDict[key].update(
            #         value_dict[key][0], value_dict[key][1])
            # else:
            #     self.MeterDict[key].update(value_dict[key][0])

    def clear(self, name_list=None):
        '''clear the variables registed in this Reporter

        :param list name_list: the names of the variables to be cleared. None means all of the variables should be cleared 
        '''
        if name_list is None:
            name_list = self.name_list
        for name in name_list:
            self.MeterDict[name].reset()

    def print(self, pos, prefix=None, num_banches=None):
        '''print the message to the std output device(screen)

        :param int pos: the position of the message
        :param str prefix: the prefix of the message
        :param int num_banches: the number of message need to be showed
        format: prefix[pos/num_banches] V1 V2 ...
        '''
        s = self.tostring(pos, prefix, num_banches)
        print(s)

    def logging(self, pos, prefix=None, num_banches=None):
        '''logging the message to the log_file

        :param int pos: the position of the message
        :param str prefix: the prefix of the message
        :param int num_banches: the number of message need to be showed
        format: prefix[pos/num_banches] V1 V2 ...
        '''
        if self.file is None:
            print("Warning! log file is not exist")
            return
        s = self.tostring(pos, prefix, num_banches)
        self.file.write(s+'\n')
        self.file.flush()

    def logstring(self, str_msg):
        self.file.write(str_msg+'\n')

    def printstring(self, str_msg):
        print(str_msg)

    def tostring(self, pos, prefix=None, num_banches=None):
        '''get the string of the message to be shown

        :param int pos: the position of the message
        :param str prefix: the prefix of the message
        :param int num_banches: the number of message need to be showed
        format: prefix[pos/num_banches] V1 V2 ...
        '''
        self.progress.updata(num_batches=num_banches, prefix=prefix)
        return self.progress.getstr(pos)

    def getvalue(self, key, data_type="mode"):
        '''get the value of one reigisted variable

        :param str key: the name of the variable
        :param data_type key: the type of the variable. ["val","avg","sum","count"] can be choose
        '''
        if key not in self.MeterDict:
            print("Warning! %s is not registed in the reporter" % key)
            return 0
        if data_type == "val":
            return self.MeterDict[key].val
        elif data_type == "avg":
            return self.MeterDict[key].avg
        elif data_type == "sum":
            return self.MeterDict[key].sum
        elif data_type == "count":
            return self.MeterDict[key].count
        else:
            return self.MeterDict[key].getmodevalue()
