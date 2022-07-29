# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Han Zhu           zhuhan@hccl.ioa.ac.cn           (Institute of Acoustics, Chinese Academy of Science)

import inspect
import logging
from eteh.utils.dynamic_import import dynamic_import


class BaseConfig(object):
    def __init__(self, conf_dict={}):
        self.conf_dict = conf_dict
        self.conf_class = None
        if not self.config_initial():
            raise RuntimeError("config is wrong")

    def config_initial(self):
        return True

    def generateExample(self):
        raise NotImplementedError("BaseConfig is not implemented")

    def __getitem__(self, index):
        return self.conf_dict[index]

    def __setitem__(self, index, value):
        if index in self.conf_dict:
            self.conf_dict[index] = value
        else:
            raise RuntimeWarning(index + 'is not in this config')

    def get_conf_dict(self):
        return self.conf_dict


    @staticmethod
    def check_kwargs(cls, kwargs, name=None):
        def _default_params(cls):
            try:
                d = dict(inspect.signature(cls.__init__).parameters)
            except ValueError:
                d = dict()
            return {
                k: v.default for k, v in d.items() if v.default != inspect.Parameter.empty and k != "self"
            }
        def _required_params(cls):
            try:
                d = dict(inspect.signature(cls.__init__).parameters)
            except ValueError:
                d = dict()
            return {
                k: v.default for k, v in d.items() if v.default == inspect.Parameter.empty and k != "self"
            }
            
        try:
            params = inspect.signature(cls.__init__).parameters
        except ValueError:
            return
        if name is None:
            name = cls.__name__
        for k in kwargs.keys():
            if k not in params:
                raise ValueError("initialization of class '{0}' got an unexpected keyword argument '{1}', "
                                 "the standard config should be {2}".format(name, k, params))
        for k in _required_params(cls):
            if k not in kwargs:
                raise ValueError("initialization of class '{0}' require keyword argument '{1}', "
                                 "the standard config should be {2}".format(name, k, params))
        for k in _default_params(cls):
            if k not in kwargs.keys():
                logging.warning("initialization of class '{0}' require keyword argument '{1}', "
                              "set to default value {2}".format(name, k, _default_params(cls)[k]))


class ModelConfig(BaseConfig):
    def __init__(self, conf_dict={}):
        super(ModelConfig, self).__init__(conf_dict)

    def config_initial(self):
        if 'name' not in self.conf_dict:
            raise ValueError("model_config do not has name")
        self.conf_class = dynamic_import(self.conf_dict['name'])
        self.conf_dict.pop('name')
        self.check_kwargs(self.conf_class, self.conf_dict)
        
        return True

    def generateExample(self):
        return self.conf_class(**self.conf_dict)


class DataConfig(BaseConfig):
    def __init__(self, conf_dict={}):
        super(DataConfig, self).__init__(conf_dict)

class CriterionConfig(BaseConfig):
    def __init__(self, conf_dict={}):
        super(CriterionConfig, self).__init__(conf_dict)
    def config_initial(self):
        if 'name' not in self.conf_dict:
            raise ValueError("criterion_config do not has name")
        self.conf_class = dynamic_import(self.conf_dict['name'])
        self.conf_dict.pop('name')
        self.check_kwargs(self.conf_class, self.conf_dict)
        
        return True

    def generateExample(self):
        return self.conf_class(**self.conf_dict)

class OptimConfig(BaseConfig):
    def __init__(self, conf_dict={}):
        super(OptimConfig, self).__init__(conf_dict)
        
    def config_initial(self):
        if 'name' not in self.conf_dict:
            raise ValueError("optim_config do not has name")
        self.conf_class = dynamic_import(self.conf_dict['name'])
        self.conf_dict.pop('name')
        self.conf_dict['params'] = None
        self.check_kwargs(self.conf_class, self.conf_dict)
        self.conf_dict.pop('params')
        
        return True

    def generateExample(self, model):
        return self.conf_class(model.parameters(), **self.conf_dict)


class SetConfig(BaseConfig):
    def __init__(self, conf_dict={}):
        super(SetConfig, self).__init__(conf_dict)

class ValidConfig(BaseConfig):
    def __init__(self, conf_dict={}):
        super(ValidConfig, self).__init__(conf_dict)

