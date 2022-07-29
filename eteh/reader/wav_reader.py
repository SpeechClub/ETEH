# AUTHORS
# Gaofeng Cheng     chenggaofeng@hccl.ioa.ac.cn     (Institute of Acoustics, Chinese Academy of Science)
# Changfeng Gao     gaochangfeng@hccl.ioa.ac.cn        (Institute of Acoustics, Chinese Academy of Science)
import numpy

def wav_reader_soundfile(wav_path):
    import soundfile
    waveform, sample_rate = soundfile.read(wav_path)
    return waveform, sample_rate

def wav_reader_torchaudio(wav_path):
    import torchaudio
    waveform, sample_rate = torchaudio.load(wav_path)
    return waveform.numpy()[0], sample_rate