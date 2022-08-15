import torch
from eteh.utils.data_utils import pad_list, to_device
from eteh.tools.interface.pytorch_backend.th_task import TH_Task


class CtcAttTask(TH_Task):
    def __init__(self, name, cuda_id, exp_path, 
                    model_config, data_config, set_config, 
                    criterion_config, optim_config, train_config, valid_config, 
                    other_config, random_seed=0, use_npu=False):
        super(CtcAttTask, self).__init__(name, cuda_id, exp_path, 
                    model_config, data_config, set_config, 
                    criterion_config, optim_config, train_config, valid_config,
                    other_config, random_seed, use_npu)
        

    def pack_data(self, data):
        x, xlen, y, ylen = data["x"][0], data["x_len"][0], data["y"].long().squeeze(-1), data["y_len"]
        y_beg = data["y_beg"].squeeze(-1) if "y_beg" in data else None
        y_end = data["y_end"].squeeze(-1) if "y_end" in data else None
        x, xlen, y, ylen = to_device(self.model, x), to_device(self.model, xlen), to_device(self.model, y), to_device(self.model, ylen)
        ys_in, ys_out = self.add_sos_eos(y, sid=self.train_config['char_num']-1)
        return {
            "x": x, 
            "xlen": xlen, 
            "y": y, 
            "ylen": ylen,
            "ys_in": ys_in,
            "ys_out": ys_out,
            "y_beg": y_beg,
            "y_end": y_end,
        }

    # def pack_result(self, result):
    #     loss, att_loss, ctc_loss = result["loss_main"], result["att_loss"], result["ctc_loss"]
    #     cor = result["att_corr"]
    #     return {
    #             "Loss": loss,
    #             "Corr": cor,
    #             "Att-Loss": att_loss,
    #             "Ctc-Loss": ctc_loss,
    #             "Lr": self.lr
    #     }

    # def pack_valid_result(self, results):
    #     return {
    #             "Loss": results["loss_main"],
    #             "Att-Loss": results["att_loss"],
    #             "Ctc-Loss": results["ctc_loss"],
    #             "Att-Corr": results["att_corr"],
    #             "CTC-CER": results["ctc_corr"],
    #     }

    def add_sos_eos(self, ys_pad, sid, ignore_id=-1):
        eos = ys_pad.new([sid])
        sos = ys_pad.new([sid])
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        return pad_list(ys_in, sid), pad_list(ys_out, ignore_id)


class CtcAttTask_Kaldi(CtcAttTask):
    def pack_data(self, data):
        data = {
            "name": data["name_list"],
            "x": [torch.from_numpy(data["feats"]["data"])],
            "x_len": [torch.LongTensor(data["feats"]["len"])],
            "y": torch.LongTensor(data["text"]["data"]),
            "y_len": torch.LongTensor(data["text"]["len"]),
            "y_beg": torch.LongTensor(data["y_beg"]["data"]) if "y_beg" in data else None,
            "y_end": torch.LongTensor(data["y_end"]["data"]) if "y_end" in data else None
        }  
        x, xlen, y, ylen = data["x"][0], data["x_len"][0], data["y"].long().squeeze(-1), data["y_len"]
        x, xlen, y, ylen = to_device(self.model, x), to_device(self.model, xlen), to_device(self.model, y), to_device(self.model, ylen)
        ys_in, ys_out = self.add_sos_eos(y, sid=self.train_config['char_num']-1)
        y_beg = to_device(self.model, data["y_beg"].squeeze(-1)) if data["y_beg"] is not None else None
        y_end = to_device(self.model, data["y_end"].squeeze(-1)) if data["y_end"] is not None else None
        return {
            "x": x, 
            "xlen": xlen, 
            "y": y, 
            "ylen": ylen,
            "ys_in": ys_in,
            "ys_out": ys_out,
            "y_beg": y_beg,
            "y_end": y_end
        }