'''
Class PackData
Script using PackData to generate .pkl format datasets
'''
import numpy as np
import librosa
import hickle
from numpy.lib import stride_tricks
import os
import argparse
import glob
from grog.fft import stft_default as stft, to_log_spec
import soundfile as sf

S_IN_HOUR = 60 * 60
SEED = 5587

class PackData(object):
    def __init__(self, data_dir, output, config):
        '''preprocess the training data'''
        self.config = config

        # get dirs for each speaker
        self.speakers_dir = [os.path.join(data_dir, i) for i in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, i))]
        self.n_speaker = len(self.speakers_dir)
        self.speaker_file = {}
        self.output = output

        # get the files in each speakers dir
        for i in range(self.n_speaker):
            wav_dir_i = [os.path.join(self.speakers_dir[i], file)
                         for file in os.listdir(self.speakers_dir[i]) if file[-3:] == "wav"]
            for j in wav_dir_i:
                if i not in self.speaker_file:
                    self.speaker_file[i] = []
                speech, sr = sf.read(j, dtype='float32')
                if not sr == config.sampling_rate:
                    raise Exception('Invalid sampling rate')
                self.speaker_file[i].append(speech)
        print("Finished loading dataset to memory. Finding statistics...")
        global_mean, global_std = self.find_global_statistics()
        self.global_mean = global_mean
        self.global_std = global_std
        print("Global mean %f" % global_mean)
        print("Global std %f" % global_std)

    def find_global_statistics(self):
        means = []
        stds = []

        for _, speaker_file in self.speaker_file.items():
            for utterance in speaker_file:
                speech_spec, _, _ = to_log_spec(utterance, self.config)
                means.append(np.mean(speech_spec))
                stds.append(np.std(speech_spec))
        
        return np.mean(means), np.mean(stds)

    def pack(self):
        '''Init the training data using the wav files'''
        np.random.seed(SEED)
        samples = []
        self.samples_counter = 0
        self.last_reported_progress = 0

        while True:
            try:
                samples.extend(self.pack_speakers())

                progress = int((self.samples_counter / self.config.sampling_rate) / S_IN_HOUR)

                if progress != self.last_reported_progress:
                    self.last_reported_progress = progress
                    print(progress)

                if (self.samples_counter >= (self.config.duration * self.config.sampling_rate)):
                    break
            except KeyboardInterrupt:
                break

        hickle.dump(samples, open(self.output, 'wb'), compression='gzip')

    def create_random_pair(self):
        i = np.random.randint(self.n_speaker)
        k = np.random.randint(self.n_speaker)
        while(i == k):
            k = np.random.randint(self.n_speaker)

        utterance_i = np.random.randint(len(self.speaker_file[i]))
        utterance_k = np.random.randint(len(self.speaker_file[k]))

        return self.speaker_file[i][utterance_i], self.speaker_file[k][utterance_k]

    def augment(self, speech):
        fac = np.random.uniform(self.config.augmentation_factor_min, self.config.augmentation_factor_max)
        return 10. ** (fac / 20) * speech

    def pack_speakers(self):
        config = self.config

        window_size = config.window_size
        windows_per_sample = config.windows_per_sample
        hop_length = config.hop_length
        global_std = self.global_std
        global_mean = self.global_mean
        threshold = config.threshold

        speech_1, speech_2 = self.create_random_pair()

        length = min(len(speech_1), len(speech_2))
        speech_1 = speech_1[:length]
        speech_2 = speech_2[:length]

        speech_1 = self.augment(speech_1)
        speech_2 = self.augment(speech_2)

        # mix
        speech_mix = speech_1 + speech_2

        speech_1_spec, _, _ = to_log_spec(speech_1, config)
        speech_2_spec, _, _ = to_log_spec(speech_2, config)
        speech_mix_spec, _, _ = to_log_spec(speech_mix, config)

        max_mag = np.max(speech_mix_spec)
        speech_VAD = (speech_mix_spec > (max_mag - threshold)).astype(int)

        # https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization
        speech_mix_spec = (speech_mix_spec - global_mean) / global_std

        samples = []
        len_spec = speech_1_spec.shape[0]

        k = 0
        while(k + windows_per_sample < len_spec):
            from_index = k
            to_index = k + windows_per_sample
            k = to_index

            sample_1 = speech_1_spec[from_index:to_index, :]
            sample_2 = speech_2_spec[from_index:to_index, :]

            sample_mix = speech_mix_spec[from_index:to_index, :].astype('float64')
            #mix_phase = speech_mix_phase[from_index:to_index, :]

            Y = np.array([sample_1 > sample_2, sample_1 < sample_2]).astype('bool')
            Y = np.transpose(Y, [1, 2, 0])  # Previously: dim(axis 0) = 2, dim(axis 1) = windows, dim(axis 2) = effective points

            VAD = speech_VAD[from_index:to_index, :].astype('bool')

            self.samples_counter += ((windows_per_sample - 1) * hop_length) + window_size
            samples.append({'Sample': sample_mix,
            #                'Phase': mix_phase,
                            'VAD': VAD,
                            'Target': Y})

        return samples
