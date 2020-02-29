import numpy as np
from numpy.lib import stride_tricks
import librosa
import scipy

def stft_default(signal, window_size, hop_length):
    window = scipy.signal.hann(window_size, sym=False)

    # return stft(signal, window_size, window, hop_length)
    return librosa.core.stft(signal, n_fft=window_size, hop_length=hop_length, window=window, center=True, pad_mode='constant').transpose(1, 0)
    #return scipy.signal.stft(
    #    signal,
    #    fs=8000,
    #    window=window, noverlap=window_size - hop_length,
    #    boundary='zeros',
    #    padded=False
    #)[2].transpose(1, 0) * 129


def istft_default(window_size, hop_length, spec1, spec2, phase1, phase2):
    window = scipy.signal.hann(window_size, sym=False)

    """ _, out_audio1 = scipy.signal.istft(
        spec1 * phase1,
        fs=8000,
        window=window, noverlap=window_size - hop_length,
        boundary=True,
        input_onesided=True,
        time_axis=0, freq_axis=1)
    _, out_audio2 = scipy.signal.istft(
        spec2 * phase2,
        fs=8000,
        window=window, noverlap=window_size - hop_length,
        boundary=True,
        input_onesided=True,
        time_axis=0, freq_axis=1) """

    out_audio1 = librosa.core.istft(
        np.transpose(spec1 * phase1),
        hop_length=hop_length,
        window=window,
        center=True
    )
    out_audio2 = librosa.core.istft(
        np.transpose(spec2 * phase2),
        hop_length=hop_length,
        window=window,
        center=True
    )

    return out_audio1.astype(np.float32), out_audio2.astype(np.float32)


def to_log_spec(wave, config):
    window_size = config.window_size
    hop_length = config.hop_length

    spec0 = stft_default(wave, window_size, hop_length)
    spec = np.abs(spec0)
    phase = spec0 / spec
    log_spec = np.maximum(spec, 1e-10)
    log_spec = 20. * np.log10(log_spec)
    return log_spec, phase, spec
