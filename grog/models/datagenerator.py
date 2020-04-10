'''
Class DataGenerator:
    Read in the .pkl data sets generated using datagenerator.py
    and present the batch data for the model
'''
import numpy as np
import librosa
import hickle
from numpy.lib import stride_tricks
import os


class DataGenerator(object):
    def __init__(self, pkl_list, batch_size, windows_per_sample, ft_bins):
        '''pkl_list: .pkl files contaiing the data set'''
        self.ind = 0  # index of current reading position
        self.batch_size = batch_size
        self.samples = []
        self.epoch = 0
        self.windows_per_sample = windows_per_sample
        self.ft_bins = ft_bins

        # read in all the .pkl files
        for pkl in pkl_list:
            self.samples.extend(hickle.load(pkl))
        self.tot_samp = len(self.samples)
        print("Loaded %d samples into memory." % self.tot_samp)
        np.random.shuffle(self.samples)

    def gen_batch(self):
        # generate a batch of data
        n_begin = self.ind
        n_end = self.ind + self.batch_size
        # ipdb.set_trace()
        if n_end >= self.tot_samp:
            # rewire the index
            self.ind = 0
            n_begin = self.ind
            n_end = self.ind + self.batch_size
            self.epoch += 1
            np.random.shuffle(self.samples)
        self.ind += self.batch_size
        return self.samples[n_begin:n_end]

    def gen_tf_batch(self):
        data_batch = self.gen_batch()
        # concatenate the samples into batch data
        in_data_np = np.concatenate(
            [np.reshape(item['Sample'], [1, self.windows_per_sample, self.ft_bins])
             for item in data_batch]
        )
        VAD_data_np = np.concatenate(
            [np.reshape(item['VAD'], [1, self.windows_per_sample, self.ft_bins])
             for item in data_batch]
        )
        VAD_data_np = VAD_data_np.astype('int')
        Y_data_np = np.concatenate(
            [np.reshape(item['Target'], [1, self.windows_per_sample, self.ft_bins, 2]) # Target contains embeddings for "2" speakers
             for item in data_batch]
        )
        Y_data_np = Y_data_np.astype('int')
        return in_data_np, Y_data_np, VAD_data_np
