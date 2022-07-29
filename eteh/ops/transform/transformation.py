from collections import OrderedDict
import copy
import io
import logging
import sys

import yaml
import numpy as np

from eteh.utils.dynamic_import import dynamic_import


PY2 = sys.version_info[0] == 2

if PY2:
    from collections import Sequence
    from funcsigs import signature
else:
    # The ABCs from 'collections' will stop working in 3.8
    from collections.abc import Sequence
    from inspect import signature


# TODO(karita): inherit TransformInterface
# TODO(karita): register cmd arguments in asr_train.py
import_alias = dict(
    identity="eteh.ops.transform.transform_interface:Identity",
    time_warp="eteh.ops.transform.spec_augment:TimeWarp",
    time_mask="eteh.ops.transform.spec_augment:TimeMask",
    freq_mask="eteh.ops.transform.spec_augment:FreqMask",
    spec_augment="eteh.ops.transform.spec_augment:SpecAugment",
    speed_perturbation="eteh.ops.transform.perturb:SpeedPerturbation",
    volume_perturbation="eteh.ops.transform.perturb:VolumePerturbation",
    noise_injection="eteh.ops.transform.perturb:NoiseInjection",
    bandpass_perturbation="eteh.ops.transform.perturb:BandpassPerturbation",
    rir_convolve="eteh.ops.transform.perturb:RIRConvolve",
    delta="eteh.ops.transform.add_deltas:AddDeltas",
    cmvn="eteh.ops.transform.cmvn:CMVN",
    utterance_cmvn="eteh.ops.transform.cmvn:UtteranceCMVN",
    fbank="eteh.ops.transform.spectrogram:LogMelSpectrogram",
    spectrogram="eteh.ops.transform.spectrogram:Spectrogram",
    stft="eteh.ops.transform.spectrogram:Stft",
    istft="eteh.ops.transform.spectrogram:IStft",
    stft2fbank="eteh.ops.transform.spectrogram:Stft2LogMelSpectrogram",
    wpe="eteh.ops.transform.wpe:WPE",
    channel_selector="eteh.ops.transform.channel_selector:ChannelSelector",
    kaldi_fbank="eteh.ops.transform.torch_audio_perturb:WavToKaldiFbank",
    time_stretch="eteh.ops.transform.torch_audio_perturb:TimeStretch",
    far_field="eteh.ops.transform.torch_audio_perturb:FarFieldSimu",
)


class Transformation(object):
    """Apply some functions to the mini-batch

    Examples:
        >>> kwargs = {"process": [{"type": "fbank",
        ...                        "n_mels": 80,
        ...                        "fs": 16000},
        ...                       {"type": "cmvn",
        ...                        "stats": "data/train/cmvn.ark",
        ...                        "norm_vars": True},
        ...                       {"type": "delta", "window": 2, "order": 2}]}
        >>> transform = Transformation(kwargs)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> xs = transform(xs)
    """

    def __init__(self, conffile=None):
        if conffile is not None:
            if isinstance(conffile, dict):
                self.conf = copy.deepcopy(conffile)
            else:
                with io.open(conffile, encoding="utf-8") as f:
                    self.conf = yaml.safe_load(f)
                    assert isinstance(self.conf, dict), type(self.conf)
        else:
            self.conf = {"mode": "sequential", "process": []}

        self.functions = OrderedDict()
        if self.conf.get("mode", "sequential") == "sequential":
            for idx, process in enumerate(self.conf["process"]):
                assert isinstance(process, dict), type(process)
                opts = dict(process)
                process_type = opts.pop("type")
                class_obj = dynamic_import(process_type, import_alias)
                # TODO(karita): assert issubclass(class_obj, TransformInterface)
                try:
                    self.functions[idx] = class_obj(**opts)
                except TypeError:
                    try:
                        signa = signature(class_obj)
                    except ValueError:
                        # Some function, e.g. built-in function, are failed
                        pass
                    else:
                        logging.error(
                            "Expected signature: {}({})".format(
                                class_obj.__name__, signa
                            )
                        )
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"])
            )

    def __repr__(self):
        rep = "\n" + "\n".join(
            "    {}: {}".format(k, v) for k, v in self.functions.items()
        )
        return "{}({})".format(self.__class__.__name__, rep)

    def __call__(self, xs, uttid_list=None, **kwargs):
        """Return new mini-batch

        :param Union[Sequence[np.ndarray], np.ndarray] xs:
        :param Union[Sequence[str], str] uttid_list:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        
        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(xs))]

        xs_return = xs
        if self.conf.get("mode", "sequential") == "sequential":
            for idx in range(len(self.conf["process"])):
                func = self.functions[idx]
                # TODO(karita): use TrainingTrans and UttTrans to check __call__ args
                # Derive only the args which the func has
                try:
                    param = signature(func).parameters
                except ValueError:
                    # Some function, e.g. built-in function, are failed
                    param = {}
                _kwargs = {k: v for k, v in kwargs.items() if k in param}
                try:
                    if uttid_list is not None and "uttid" in param:
                        for i in range(len(xs_return)):
                            xs_return[i] = func(xs_return[i], uttid_list[i], **_kwargs)
                    else:
                        for i in range(len(xs_return)):
                            xs_return[i] = func(xs_return[i], **_kwargs)
                except Exception:
                    logging.fatal(
                        "Catch a exception from {}th func: {}".format(idx, func)
                    )
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"])
            )

        return xs_return
