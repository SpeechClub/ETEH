# ETEH: Unified End-to-End Speech Processing Platform

- [Contents](#contents)
    - [Description](#description)
    - [Configuration](#configuration)
    - [Install Guideline](#install-guideline)
    - [Use Guideline](#use-guideline)
    - [Citation](#citation)
    - [Contact](#contact)
    - [Reference](#reference)

***
## [Description](#contents)

ETEH platform is a lightweight toolkit for speech deep learning tasks. You can easily train the end-to-end speech recognition, keyword search, text-and-speech alignments generation, universal speech embedding extraction, etc. with the ETEH. 
“ETE” stands for end-to-end modeling and optimizing, whereas “H” represents the hybrid gaussian mixture model and hidden Markov model (GMM-HMM). 

Compared to other toolkits, ETEH provides both training examples project and training APIs. You can directly run the examples project to realize high-performance ASR systems for industry or implement the training API to train your model for research.

***
## [Configuration](#contents)

ETEH is based on python and pytorch, but we recommend the below configuration for use.
- Python3.7+  
- PyTorch 1.4.1+
- numpy 1.9+
- Cuda 9.1+, 10.1+, 11.1+ (for the use of GPU)  
- Cudnn 6+ (for the use of GPU)  
- NCCL 2.0+ (for the use of multi-GPUs)
- editdistance (eval the ASR results, install with pip)
- kaldi-io (if you want to read kaldi style data, install with pip)
- soundfile (if you want to read raw wav file or flac file, install with pip)
- [torchaudio](https://pytorch.org/) (if you want to extract the speech feathers online)
- [apex](https://github.com/NVIDIA/apex) (if you want to use mixed precision training)
- [Kaldi](https://github.com/kaldi-asr/kaldi) (Kaldi is not necessary for the ASR training, but you can use Kaldi to prepare the training resource)

***
## [Install Guideline](#contents)

If all recommended configurations are satisfied, ETEH can be directly used by only adding it to the `PYTHONPATH` environment variable.
```bash
export PYTHONPATH=/path/to/etehfolder/:$PYTHONPATH
```


***
## [Use Guideline](#contents)

### Running examples project provided by us (Recommended for first use).

You can run the `run_asr_pipeline.sh` in `example/asr` folder, including offline E2E ASR training and decoding process.
Other examples such as online E2E ASR, E2E keyword search, and LM training (RNN LM) can also be found in the example folder.

We provide different ASR and LM models in the `eteh/models` folder. You can choose other models and loss functions in the config file.

Details can be found in the `example/asr` and `example/lm`.

### Train your own task with ETEH API.

ETEH support custom task training; you can train most speech task with the ETEH interface and API. In the `example/asr`, KWS training is an example of custom task training, and you can train your task by referring to the KWS example and the `eteh/lib eteh/models`. A simple guideline is also provided as follows:

#### Prepare your own data

#### Dictionary
ETEH need a dict file to decide the index of the output token, a dict example is:
```
<unk> 1
A 2
B 3 
...
Z 27
<space> 28
' 29
```
Index 0 will be used for ctc blank, and an extra `<eos>` token will be appended at last automatically.

#### Training resource
For ASR or other speech task, the ETEH can read 4 kinds of data, incuding wav file, kaldi ark file (scp,ark), text file and label sequence file. The format of each file is listed as follow:

```
#wav file (wav or flac file can be read by soundfile):
uttrance_id1 /path/to/wav/file1.wav
uttrance_id2 /path/to/wav/file1.wav

#kaldi ark file (use kaldi-io to read):
uttrance_id1 /path/to/ark/file1.ark:80
uttrance_id2 /path/to/ark/file1.ark:1200

#text file (use space to divide the token): 
uttrance_id1 H E L L O <space> W O R L D !
uttrance_id2 E T E H <space> T O O L K I T

#label sequence file (the label should be integer)
uttrance_id1 1 2 3 4 5
uttrance_id2 6 7 8 9 10 11
```

After preparing these data, you should write a `data_config.yaml` file then the ETEH can read these data according to the `data_config.yaml`. We also provide a shell file to generate the `data_config.yaml` automatically. You can see the details in `example/asr`.

For LM or other text tasks, the ETEH can read text files directly but guarantee the token is split by space. Details can be found in `example/lm/config/data`

#### Define your training task with TH_Task.

You can define a training task with `eteh.tools.interface.pytorch_backend.th_task.TH_Task` and overwirte `pack_data` function.

```python
class MyTask(TH_Task):
  def pack_data(self, data):
    x = data["feats"]["data"]    
    x_len = data["feats"]["len"]    
    y = data["text"]["data"]
    y_len = data["text"]["len"]
    return {
      "x": x
      "x_len": x_len,
      "y": y,
      "y_len": y_len
    }
```
The `data` parameter of the `MyTask.pack_data` is decided by the `data_config.yaml`.

You can also overwrite other functions in `TH_Task`.

#### Define your model with Model_Interface.

You can train your own torch model with the `eteh.models.model_interface.Model_Interface` by overwriting the `train_forward` and `valid_forward` functions. 

```python
class MyEtehModel(MyTorchModel, Model_Interface):
    def train_forward(self, input_dict):        
      h = self.forward(
            input_dict["x"],
            input_dict["x_len"] 
        )
      return {
        "h": h
      }

class MyEtehLoss(MyTorchLoss, Model_Interface):
    def train_forward(self, input_model_dict):
      loss = self.forward(
            input_model_dict["y"],
            input_model_dict["h"],
        )
      return {
        "loss_main": loss
      }
```

The `input_dict` of the `MyEtehModel.train_forward` is the return value of the `TH_Task.pack_data`. The `input_model_dict` of the `MyEtehLoss.train_forward` is the union of the `TH_Task.pack_data` and `MyEtehModel.train_forward` return values.

ETEH will optimize the system with the `"loss_main"` in the `MyEtehLoss.train_forward` return value, guarantee that the `MyEtehLoss.train_forward(input_model_dict)["loss_main"]` can be optimized.

#### Start training

After previous work, you can write a `train_config.yaml` file to appoint the model, criterion, and optimizer by the class name and the parameters of `__init__` function. We also provide a shell file to generate the `train_config.yaml` automatically.

You can also adjust the training parameters in the `train_config.yaml`. Details can be found in `example/asr/config/e2e`.

You can start training with the `bin/train.py` or `bin/train_dist.py` by providing the `train_config.yaml`, `data_config.yaml` and the path to the `MyTask`. Examples are shown in the `example/asr/run_basic.sh`. You can also directly define the `MyTask` in the python shell (see the `bin/train_dist.py` for more details).


***
## [Citation](#contents)


If you find this code useful in your research, please kindly consider citing our paper:

```
@article{cheng2022eteh,
  title={ETEH: Unified Attention-Based End-to-End ASR and KWS Architecture},
  author={Cheng, Gaofeng and Miao, Haoran and Yang, Runyan and Deng, Keqi and Yan, Yonghong},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={1360--1373},
  year={2022},
  publisher={IEEE}
}
```

***
## [Contact](#contents)

If you have any questions, please contact us. You could open an issue on github or email us.

| Authors       | Email                                                               |
|---------------|---------------------------------------------------------------------|
| Gaofeng Cheng | [chenggaofeng@hccl.ioa.ac.cn](mailto:chenggaofeng@hccl.ioa.ac.cn)   |
| Changfeng Gao | [gaochangfeng@hccl.ioa.ac.cn](mailto:gaochangfeng@hccl.ioa.ac.cn)   |
| Runyan Yang   | [yangrunyan@hccl.ioa.ac.cn](mailto:yangrunyan@hccl.ioa.ac.cn)       |


***
## [Reference](#contents)

[1] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and Desmaison, A., 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

[2] Povey, D., Ghoshal, A., Boulianne, G., Burget, L., Glembek, O., Goel, N., Hannemann, M., Motlicek, P., Qian, Y., Schwarz, P. and Silovsky, J., 2011. The Kaldi speech recognition toolkit. In IEEE 2011 workshop on automatic speech recognition and understanding (No. CONF). IEEE Signal Processing Society.

[3] Watanabe, S., Hori, T., Karita, S., Hayashi, T., Nishitoba, J., Unno, Y., Soplin, N.E.Y., Heymann, J., Wiesner, M., Chen, N. and Renduchintala, A., 2018. Espnet: End-to-end speech processing toolkit. arXiv preprint arXiv:1804.00015.