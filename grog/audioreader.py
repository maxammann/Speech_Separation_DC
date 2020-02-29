'''
Class audio_reader:
    read in a single audio sample for test
'''
import numpy as np
import librosa
from numpy.lib import stride_tricks
import os
from grog.fft import stft_default as stft, to_log_spec


class AudioReader(object):

    def __init__(self, speech_mix, config, log=True):
        '''Load in the audio file and transform the signal into
        the formats required by the model'''
        windows_per_sample = config.windows_per_sample
        ft_bins = config.window_size // 2 + 1

        self.windows_per_sample = config.windows_per_sample
        self.ft_bins = ft_bins
        self.ind = 0

        speech_mix_spec, speech_phase, speech_VAD = self.get_spectrum(speech_mix, config, log)

        len_spec = speech_mix_spec.shape[0]
        
        self.samples = []

        k = 0
        while(k + windows_per_sample < len_spec):
            from_index = k
            to_index = k + windows_per_sample
            k = to_index

            phase = speech_phase[from_index: to_index, :]
            sample_mix = speech_mix_spec[from_index:to_index, :]
            VAD = speech_VAD[from_index:to_index, :]
            self.samples.append({
                'Sample': sample_mix,
                'VAD': VAD,
                'Phase': phase
            })
            

        n_left = windows_per_sample - len_spec + k

        phase = np.concatenate((speech_phase[k:, :], np.zeros([n_left, ft_bins])))
        sample_mix = np.concatenate((speech_mix_spec[k:, :], np.zeros([n_left, ft_bins])))
        VAD = np.concatenate((speech_VAD[k:, :], np.zeros([n_left, ft_bins])))

        self.samples.append({
            'Sample': sample_mix,
            'VAD': VAD,
            'Phase': phase
        })
        self.tot_samp = len(self.samples)

    def get_spectrum(self, speech_mix, config, log):
        log_spec, speech_phase, linear_spec = to_log_spec(speech_mix, config)
        max_mag = np.max(log_spec)
        speech_VAD = (log_spec > (max_mag - config.threshold)).astype(int)

        if log:
            log_spec = (log_spec - config.dataset_mean) / config.dataset_std
            return log_spec, speech_phase, speech_VAD
        else:
            return linear_spec, speech_phase, speech_VAD

    def gen_next(self):
        begin = self.ind
        if begin >= self.tot_samp:
            return None
        self.ind += 1
        return [self.samples[begin]]

    def get_tf_next(self):
        data_batch = self.gen_next()

        if data_batch is None:
            return None

        in_data_np = np.concatenate(
            [np.reshape(item['Sample'], [1, self.windows_per_sample, self.ft_bins])
             for item in data_batch])
        in_phase_np = np.concatenate(
            [np.reshape(item['Phase'], [1, self.windows_per_sample, self.ft_bins])
             for item in data_batch])
        VAD_data_np = np.concatenate(
            [np.reshape(item['VAD'], [1, self.windows_per_sample, self.ft_bins])
             for item in data_batch])

        return in_data_np, in_phase_np, VAD_data_np
