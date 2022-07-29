# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
# Yifan Guo         guoyifan@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import numpy 
import scipy, librosa
import soundfile
import torch
from .transform_interface import TransformInterface

# from espnet.utils.io_utils import SoundHDF5File


class NoiseInjection(TransformInterface):
    """Add isotropic noise"""

    def __init__(
        self,
        sr=16000,
        utt2noise=None,
        lower=-20,
        upper=-5,
        utt2ratio=None,
        filetype="list",
        dbunit=True,
        seed=None,
        dtype="float64",
    ):
        self.utt2noise_file = utt2noise
        self.utt2ratio_file = utt2ratio
        self.filetype = filetype
        self.dbunit = dbunit
        self.lower = lower
        self.upper = upper
        self.state = numpy.random.RandomState(seed)

        if utt2ratio is not None:
            # Use the scheduled ratio for each utterances
            self.utt2ratio = {}
            with open(utt2noise, "r") as f:
                for line in f:
                    utt, snr = line.rstrip().split(None, 1)
                    snr = float(snr)
                    self.utt2ratio[utt] = snr
        else:
            # The ratio is given on runtime randomly
            self.utt2ratio = None

        if utt2noise is not None:
            self.utt2noise = {}
            if filetype == "list":
                with open(utt2noise, "r") as f:
                    for line in f:
                        utt, filename = line.rstrip().split(None, 1)
                        signal, rate = soundfile.read(filename, dtype=dtype)
                        # Load all files in memory
                        self.utt2noise[utt] = signal

            # elif filetype == "sound.hdf5":
            #     self.utt2noise = SoundHDF5File(utt2noise, "r")
            else:
                raise ValueError(filetype)
            self.noise_list = list(self.utt2noise.values())
            self.num_noises = len(self.noise_list)
        else:
            self.utt2noise = None


        assert sr == rate, ""

        if utt2noise is not None and utt2ratio is not None:
            if set(self.utt2ratio) != set(self.utt2noise):
                raise RuntimeError(
                    "The uttids mismatch between {} and {}".format(utt2ratio, utt2noise)
                )

    def __repr__(self):
        if self.utt2ratio is None:
            return "{}(lower={}, upper={}, dbunit={})".format(
                self.__class__.__name__, self.lower, self.upper, self.dbunit
            )
        else:
            return '{}("{}", dbunit={})'.format(
                self.__class__.__name__, self.utt2ratio_file, self.dbunit
            )

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x
        x = x.astype(numpy.float32)

        # 1. Get ratio of noise to signal in sound pressure level
        if uttid is not None and self.utt2ratio is not None:
            ratio = self.utt2ratio[uttid]
        else:
            ratio = self.state.uniform(self.lower, self.upper)

        if self.dbunit:
            ratio = 10 ** (ratio / 20)
        scale = ratio * numpy.sqrt((x ** 2).mean())

        # 2. Get noise
        if self.utt2noise is not None:
            # Get noise from the external source
            if uttid is not None:
                noise = self.utt2noise[uttid]
            else:
                # Randomly select the noise source
                noise = self.noise_list[self.state.randint(self.num_noises)]
               

            # Normalize the level
            noise /= numpy.sqrt((noise ** 2).mean())

            # Adjust the noise length
            diff = abs(len(x) - len(noise))
            offset = self.state.randint(0, diff)
            if len(noise) > len(x):
                # Truncate noise
                noise = noise[offset : -(diff - offset)]
            else:
                noise = numpy.pad(noise, pad_width=[offset, diff - offset], mode="wrap")

        else:
            # Generate white noise
            noise = self.state.normal(0, 1, x.shape)

        # 3. Add noise to signal
        return x + noise * scale

class RIRConvolve(TransformInterface):
    """ 
        utt2rir文件需要保存成:
            1) .hdf5格式:   {key: (rir, sr)} or 
            2) rir(noise).scp:     uttid rir.wav
    """
    def __init__(self, utt2rir, sr=16000, filetype="list", seed=None, dtype="float64"):
        self.utt2rir_file = utt2rir
        self.filetype = filetype

        self.utt2rir = {}
        if filetype == "list":
            with open(utt2rir, "r") as f:
                for line in f:
                    utt, filename = line.rstrip().split(None, 1)
                    signal, rate = soundfile.read(filename, dtype=dtype)
                    self.utt2rir[utt] = signal
        

        # elif filetype == "sound.hdf5":
        #     self.utt2rir = SoundHDF5File(utt2rir, "r")
        else:
            raise NotImplementedError(filetype)
        
        self.state = numpy.random.RandomState(seed)
        self.rir_list = list(self.utt2rir.values())
        self.num_rir = len(self.rir_list)

        assert sr == rate, f"{sr} {rate}"

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.utt2rir_file)

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x

        x = x.astype(numpy.float32)

        if x.ndim != 1:
            # Must be single channel
            raise RuntimeError(
                "Input x must be one dimensional array, but got {}".format(x.shape)
            )

        if uttid is not None:
            rir = self.utt2rir[uttid]
        else:
            # Randomly select the rir source
            rir = self.rir_list[self.state.randint(self.num_rir)]
            

        if rir.ndim == 2:
            # FIXME(kamo): Use chainer.convolution_1d?
            # return [Time, Channel]
            return numpy.stack(
                [scipy.signal.convolve(x, rir) for r in rir], axis=-1
            )
        else:
            x = scipy.signal.convolve(x, rir)
            return x / x.max() * 0.3

class FarFieldSimu(TransformInterface):
    """ Generate Farfield simulated data. (Numpy input and output)

        Input:  np.int16
        Output: np.int16
        
        Args:
            sr: input sample rate

            rir_filetype: the filetype of utt2rir file "list" or "hdf5"
            use_rir_prob: the probability of reverberation simu

            lower, upper: the lower and upper bound of signal-noise ratio
            utt2ratio: FILELIST, specify the noise utt with a ratio,  egs: utt1 ratio1
            noise_filetype: refer to 'rir_filetype'
            dbunit: use dB as the unit of 'lower' and 'upper' or not
            noise_seed: Random number seed for selecting noise id and snr ratio
            use_noise_prop: refer to 'use_rir_prob'
            seed: Random number seed for whether use rir and noise
    """
    def __init__(
        self,
        sr=16000,
        # add reverberation
        utt2rir=None,
        rir_filetype="list",
        use_rir_prob=1.0,
        # add noise 
        utt2noise=None,
        lower=10,
        upper=30,
        utt2ratio=None,
        noise_filetype="list",
        dbunit=True,
        noise_seed=None,
        use_noise_prob=1.0,
        seed=None,
        ):

        assert use_noise_prob<=1.0 and use_noise_prob >=0, ""
        assert use_rir_prob<=1.0 and use_rir_prob >=0, ""
        self.rir_convolve= RIRConvolve(utt2rir, sr, rir_filetype)
        self.noise_injection = NoiseInjection(sr, utt2noise, -upper, -lower, utt2ratio, noise_filetype, dbunit, seed) 
        self.use_rir_prob = use_rir_prob
        self.use_noise_prob = use_noise_prob
        self.state = numpy.random.RandomState(seed)
        self.sr = sr


    def __repr__(self):
        return f"{self.__class__.__name__} ({self.rir_convolve} {self.noise_injection})"
    
    def __call__(self, audio, rir_uttid=None, noise_uttid=None, train=True):
        
        assert isinstance(audio, numpy.ndarray)
        assert audio.ndim == 2, ""

        audio = audio[:, 0]
        if audio.dtype == numpy.int16:
            audio = audio / 32768
        elif audio.dtype == numpy.int8:
            audio = audio / 128
        else:
            audio = audio
        if self.state.random() <= self.use_rir_prob:
            audio = self.rir_convolve(audio, rir_uttid, train)
        if self.state.random() <= self.use_noise_prob:
            audio = self.noise_injection(audio, noise_uttid, train)
        
        return numpy.expand_dims(audio, axis=-1)
        
class TimeStretch(TransformInterface):
    """ Speed Perturb 

        Input:
            audio: np.int16 (T1,)
        Output:
            audio: np.int16 (T2,) 

    """
    def __init__(
        self,
        lower=0.9,
        upper=1.5,
        sr=16000,
        seed=None,
        return_ratio=False,
        ):
        self.upper = upper
        self.lower = lower
        self.bound = numpy.round(numpy.arange(lower, upper+0.09, 0.1), 1)
        self.state = numpy.random.RandomState(seed)
        self.return_ratio = return_ratio
        self.sr = sr

    def __repr__(self):
        return f"{self.__class__.__name__}, upper: {self.upper}, lower: {self.lower}"
    
    def __call__(self, audio, *args, train=True, **kwargs):
        import torchaudio
        if not train:
            return audio
        
        if audio.dtype == numpy.int16:
            audio = audio / 32768
        elif audio.dtype == numpy.int8:
            audio = audio / 128
        else:
            audio = audio

        audio = torch.from_numpy(audio)        
        assert audio.ndim == 2, ""
        
        audio = audio.transpose(0,1).float()
        
        ratio = float(self.state.choice(self.bound))
        audio, sr = torchaudio.sox_effects.apply_effects_tensor(audio, self.sr, [
            ["tempo", f"{ratio}"],
            ["rate", f"{self.sr}"],
        ])

        if self.return_ratio:
            return audio.transpose(0,1).detach().numpy(), ratio

        return audio.transpose(0,1).detach().numpy()
        
        # return numpy.int16(32768 * audio.squeeze().numpy())


class WavToKaldiFbank(TransformInterface):
    """ Generate Kaldi Fbank with wavform

        Input  np.float (T, 1)
        Output np.float (T, F)

    """
    def __init__(
        self, 
        blackman_coeff: float = 0.42, 
        channel: int = -1, 
        dither: float = 0.0, 
        energy_floor: float = 1.0, 
        frame_length: float = 25.0, 
        frame_shift: float = 10.0, 
        high_freq: float = 0.0, 
        htk_compat: bool = False, 
        low_freq: float = 20.0, 
        min_duration: float = 0.0, 
        num_mel_bins: int = 80, 
        preemphasis_coefficient: float = 0.97, 
        raw_energy: bool = True, 
        remove_dc_offset: bool = True, 
        round_to_power_of_two: bool = True, 
        sample_frequency: float = 16000.0, 
        snip_edges: bool = True, 
        subtract_mean: bool = False, 
        use_energy: bool = False, 
        use_log_fbank: bool = True, 
        use_power: bool = True, 
        vtln_high: float = -500.0, 
        vtln_low: float = 100.0, 
        vtln_warp: float = 1.0, 
        window_type: str = 'povey',
        audio_bit: int = 16
        ):
        self.blackman_coeff = blackman_coeff
        self.channel = channel
        self.dither = dither
        self.energy_floor = energy_floor
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.high_freq = high_freq
        self.htk_compat = htk_compat
        self.low_freq = low_freq
        self.min_duration = min_duration
        self.num_mel_bins = num_mel_bins
        self.preemphasis_coefficient = preemphasis_coefficient
        self.raw_energy = raw_energy
        self.remove_dc_offset = remove_dc_offset
        self.round_to_power_of_two = round_to_power_of_two
        self.sample_frequency = sample_frequency
        self.snip_edges = snip_edges
        self.subtract_mean = subtract_mean
        self.use_energy = use_energy
        self.use_log_fbank = use_log_fbank
        self.use_power = use_power
        self.vtln_high = vtln_high
        self.vtln_low = vtln_low
        self.vtln_warp = vtln_warp
        self.window_type = window_type
        self.audio_bit = audio_bit

    def __call__(self, wavform, *args, **kwargs):
        import torchaudio
        wavform = torch.from_numpy(wavform).transpose(0,1).float()
        wavform = wavform * 2 ** (self.audio_bit - 1)
        fbank = torchaudio.compliance.kaldi.fbank(
            wavform, 
            blackman_coeff=self.blackman_coeff, 
            channel=self.channel, 
            dither=self.dither, 
            energy_floor=self.energy_floor, 
            frame_length=self.frame_length, 
            frame_shift=self.frame_shift, 
            high_freq=self.high_freq, 
            htk_compat=self.htk_compat, 
            low_freq=self.low_freq, 
            min_duration=self.min_duration, 
            num_mel_bins=self.num_mel_bins, 
            preemphasis_coefficient=self.preemphasis_coefficient, 
            raw_energy=self.raw_energy, 
            remove_dc_offset=self.remove_dc_offset, 
            round_to_power_of_two=self.round_to_power_of_two, 
            sample_frequency=self.sample_frequency, 
            snip_edges=self.snip_edges, 
            subtract_mean=self.subtract_mean, 
            use_energy=self.use_energy, 
            use_log_fbank=self.use_log_fbank, 
            use_power=self.use_power, 
            vtln_high=self.vtln_high, 
            vtln_low=self.vtln_low, 
            vtln_warp=self.vtln_warp, 
            window_type=self.window_type,
            )
        
        return fbank.detach().numpy()

    
        