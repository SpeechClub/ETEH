import torch
from eteh.utils.data_utils import pad_list, to_device
from eteh.tools.interface.pytorch_backend.th_task import TH_Task


class LMTask(TH_Task):
    def __init__(self, name, cuda_id, exp_path, 
                    model_config, data_config, set_config, 
                    criterion_config, optim_config, train_config, valid_config, 
                    other_config, random_seed=0, use_npu=False):
        super(LMTask, self).__init__(name, cuda_id, exp_path, 
                    model_config, data_config, set_config, 
                    criterion_config, optim_config, train_config, valid_config,
                    other_config, random_seed, use_npu)
        

    def pack_data(self, data):
        y, ylen = data["y"].long().squeeze(-1), data["y_len"]
        y, ylen = to_device(self.model, y), to_device(self.model, ylen)
        ys_in, ys_out = self.add_sos_eos(y, sid=self.train_config['char_num']-1)
        return {
            "y_in": ys_in,
            "y_out": ys_out
        }

    # def pack_result(self, results):
    #     return {
    #         "Loss": results["loss_main"],
    #         "PPL": results["ppl"],
    #         "Lr": self.lr,
    #     }

    # def pack_valid_result(self, results):
    #     return {
    #         "Loss": results["loss_main"],
    #         "PPL": results["ppl"],
    #     }


    def add_sos_eos(self, ys_pad, sid, ignore_id=-1):
        eos = ys_pad.new([sid])
        sos = ys_pad.new([sid])
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        return pad_list(ys_in, sid), pad_list(ys_out, ignore_id)

