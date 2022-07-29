# An example of end-to-end ASR & KWS using ETEH

This dir incudes training recipes of the following ASR and KWS models:
- Standard CTC/attention E2E ASR
- Uniformer E2E ASR --- Unified online and offline ASR model. A concise model built by adopting the universal training strategy. [[Cheng, et al., 2022]](https://ieeexplore.ieee.org/document/9739972)
- TSE-based KWS --- Keyword search system based on E2E ASR and TimeStamp Estimator (TSE), which produces accurate keyword timestamps [[Cheng, et al., 2022]](https://ieeexplore.ieee.org/document/9739972)

Using Uniformer and TSE, you can build a unified E2E ASR and KWS architecture, which supports, in one model, both online and offline ASR decoding modes, and allows for precise and reliable KWS.

`run_asr_pipeline.sh` is the pipeline of building and testing a Standard CTC/attention E2E ASR system, including data & config preparation, training, decoding, and scoring. Please use this as a basic recipe example for building a ASR system using ETEH.

`run_uniformer.sh` is the pipeline of building and testing Uniformer E2E ASR systems, including MTA uniformer and TMTA uniformer [[Cheng, et al., 2022]](https://ieeexplore.ieee.org/document/9739972).

`run_tse_kws.sh` is the pipeline of building and testing a TSE-based KWS system, given a well-trained CTC/attention E2E ASR system.

We use the HKUST corpus as an example in these scripts. Here is the ASR performance of each systems:

||offline decoding WER(%)|online decoding WER(%)|
|:-:|:-:|:-:|
|eteh_baseline|20.8|\-|
|eteh_baseline_uniformer_mta|20.6||
|eteh_baseline_uniformer_tmta|||

## Run an example: run_asr_pipeline.sh

### Stage 0: Data preparation
Prepare training data using Kaldi.
Before running this stage, please modify `$KALDI_ROOT` in `path.sh`, as long as `$hkust_audio_path` and `$hkust_text_path` in `local/prepare_HKUST.sh` to your own paths.
```
KALDI_ROOT=your_own_path # /opt/kaldi
```
```
hkust_audio_path=your_own_path # /export/corpora/LDC/LDC2005S15/
hkust_text_path=your_own_path # /export/corpora/LDC/LDC2005T32/
```

By default, fbank-pitch features with global CMVN will be extracted and text will be segmented into characters. See generated files `data/HKUST/dev/feats_cmvn.scp` and `data/HKUST/dev/token` for an example.

We also provide an interface for loading Kaldi forced alignments.
If you want to use Kaldi forced alignments, please set `$prepare_ali` true and also modify `$kaldi_lang_dir`, `$train_set_ali_dir`, and `$train_dev_ali_dir` in `local/prepare_HKUST.sh`. See `data/HKUST/dev/{token,word}{stt,end}` for examples of generated alignment files.
```
kaldi_lang_dir=your_own_path # $KALDI_ROOT/egs/hkust/s5/data/lang
train_set_ali_dir=your_own_path # $KALDI_ROOT/egs/hkust/s5/exp/tri5a_ali_train_nodup_sp
train_dev_ali_dir=your_own_path # $KALDI_ROOT/egs/hkust/s5/exp/tri5a_ali_train_dev
```

### Stage 1: Config preparation
By default, we will generate an offline ASR training config in this stage, using config template `conf/e2e/eteh_baseline.yaml`.
To train other models, you can modify `$train_config_template` to other config files in `conf/e2e`.

A data config will also be generated automatically. If you want to use token-leval or word-level alignments during training, please set `$use_ali` to `token` or to `word`. For example, if you're going to train a TMTA Uniformer (`conf/e2e/eteh_baseline_uniformer_tmta.yaml`)[[Cheng, et al., 2022]](https://ieeexplore.ieee.org/document/9739972), which needs token-level alignments, you should set `$use_ali` to `token` in this stage.

You can also write your own config files and skip this stage.

### Stage 2: Training
The previous two stages, three key files have been generated in `data/HKUST/final_resource`:
- train_config.yaml
- data_config.yaml
- dict.txt - A dictionary file used for converting "text" files into label indices.

After training, checkpoints will be saved in `$exp_dir`.

### Stage 3: Decoding
In this stage, we average the best 10 checkpoints with regards to "att_corr" on valid set. We used this average model for decoding.

The language model used decoding is trained in `../lm`. 

By default, offline mode decoding will be performed. To perform online mode decoding please add `-online` flag to `${PLAT_ROOT}/bin/decode.py`.

Decoding results will be saved as `$output_file`.

### Stage 4: Scoring
Score the hypotheses `$output_file` using reference `$ref_file` 

## Citations
```
@ARTICLE{9739972,
  author={Cheng, Gaofeng and Miao, Haoran and Yang, Runyan and Deng, Keqi and Yan, Yonghong},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={ETEH: Unified Attention-Based End-to-End ASR and KWS Architecture}, 
  year={2022},
  volume={30},
  number={},
  pages={1360-1373},
  doi={10.1109/TASLP.2022.3161159}}

@ARTICLE{8068205,
  author={Watanabe, Shinji and Hori, Takaaki and Kim, Suyoun and Hershey, John R. and Hayashi, Tomoki},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Hybrid CTC/Attention Architecture for End-to-End Speech Recognition}, 
  year={2017},
  volume={11},
  number={8},
  pages={1240-1253},
  doi={10.1109/JSTSP.2017.2763455}}
```