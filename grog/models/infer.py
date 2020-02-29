from datetime import datetime
import os.path
import time
import argparse
import ntpath

import tensorflow as tf
import numpy as np
import librosa

from sklearn.cluster import KMeans

from numpy.lib import stride_tricks

from grog.audioreader import AudioReader
from grog.models.model import Model
from grog.config import Config
from grog.fft import stft_default, istft_default
from grog.models.local_cluster import LocalCluster


class Inference(object):

    def __init__(self, config):
        self.config = config

        self.n_layers = config.n_layers
        self.n_hidden = config.n_hidden
        self.embedding_dimension = config.embedding_dimension
        self.windows_per_sample = config.windows_per_sample
        self.batch_size = config.batch_size

        self.window_size = config.window_size
        self.sampling_rate = config.sampling_rate
        self.hop_length = config.hop_length
        self.threshold = config.threshold

        self.ft_bins = config.window_size // 2 + 1

    def blind_source_separation(self, input_files, model_dir, output_dir):
        session, embedding_model, in_data, in_data, dropout_ff, dropout_rc = self.prepare_session(model_dir)

        for input_file in input_files:
            speech_mix, _ = librosa.load(input_file, sr=self.sampling_rate)

            embeddings, N_samples = self.estimate_embeddings(session, speech_mix, embedding_model, in_data, dropout_ff, dropout_rc)
            source1, source2 = self.estimate_sources(speech_mix, embeddings, N_samples)

            mix_dir = estimated_dir = output_dir + "/" + os.path.splitext(ntpath.basename(input_file))[0]

            print("Writing to %s" % mix_dir)
            estimated_dir = mix_dir + "/estimated"
            os.makedirs(estimated_dir, exist_ok=True)
            librosa.output.write_wav(estimated_dir + "/source1.wav", source1.astype(np.float32), self.sampling_rate)
            librosa.output.write_wav(estimated_dir + "/source2.wav", source2.astype(np.float32), self.sampling_rate)
        session.close()

    def prepare_session(self, model_dir, debug=False):
        windows_per_sample = self.windows_per_sample
        if debug:
            print("Using %d layers for inferring." % self.n_layers)

        batch_size = 1  # infer one audio

        with tf.Graph().as_default():
            dropout_ff = tf.placeholder(tf.float32, shape=None)
            dropout_rc = tf.placeholder(tf.float32, shape=None)

            # placeholder for model input
            in_data = tf.placeholder(tf.float32, shape=[batch_size, windows_per_sample, self.ft_bins])
            # init the model
            BiModel = Model(self.n_layers, self.embedding_dimension, windows_per_sample,
                            self.window_size, self.n_hidden, batch_size, dropout_ff, dropout_rc)
            # make inference of embedding
            embedding_model = BiModel.inference(in_data)
            saver = tf.train.Saver(tf.all_variables())
            session = tf.Session()
            # restore the model
            saver.restore(session, os.path.join(model_dir, "model.ckpt"))

            return session, embedding_model, in_data, in_data, dropout_ff, dropout_rc

    def estimate_embeddings(self, session, speech_mix, embedding_model, in_data, dropout_ff, dropout_rc):
        audio_reader = AudioReader(speech_mix, self.config)
        N_samples = audio_reader.tot_samp
        data_batch = audio_reader.get_tf_next()
        embeddings = []
        while data_batch is not None:
            in_data_np, _, _ = data_batch
            # get inferred embedding using trained model
            embedding, = session.run([embedding_model], feed_dict={in_data: in_data_np, dropout_ff: 0, dropout_rc: 0})
            embeddings.append((embedding, data_batch))
            data_batch = audio_reader.get_tf_next()

        return embeddings, N_samples

    def create_mask(self, embeddings, local=False):
        if local:
            clusterer = LocalCluster(self.config, 'k-means++')
            global_mask = []
            sample_step = 0

            for embedding, data_batch in embeddings:
                _, _, VAD_data_np = data_batch
                mask = clusterer.cluster2(embedding, sample_step, VAD_data_np)
                sample_step += 1

                global_mask.append(mask)
            return np.transpose(global_mask, (0, 1, 2, 3))  # samples, k, windows_per_sample, ft_bins
        else:
            global_embeddings = np.empty((len(embeddings) * self.windows_per_sample, self.ft_bins, 40))

            index = 0
            for embedding, data_batch in embeddings:
                global_embeddings[index:index+self.windows_per_sample] = embedding
                index += self.windows_per_sample

            global_embeddings = global_embeddings.reshape(-1, 40)

            k = 2
            eg = KMeans(k, random_state=0).fit_predict(global_embeddings)

            global_mask = np.zeros((k, eg.size))
            for i in range(k):
                global_mask[i, eg == i] = 1

            global_mask = global_mask.reshape(k, len(embeddings), self.windows_per_sample, self.ft_bins)
            global_mask = global_mask.transpose(1, 0, 2, 3)
            return global_mask  # samples, k, windows_per_sample, ft_bins

    def estimate_sources(self, speech_mix, embeddings, N_samples, ref):
        windows_per_sample = self.windows_per_sample
        window_size = self.window_size
        hop_length = self.hop_length

        global_mask = self.create_mask(embeddings, local=True)

        spec0 = stft_default(speech_mix, window_size, hop_length)
        spec = np.abs(spec0)
        phase = spec0 / spec
        spec_length = spec.shape[0]


        global_mask = global_mask.transpose(1, 0, 2, 3).reshape(2, len(embeddings) * windows_per_sample, self.ft_bins)
        global_mask = global_mask[:,:spec_length,:]

        #ref1_spec = np.abs(stft_default(ref[0], window_size, hop_length))
        #ref2_spec = np.abs(stft_default(ref[1], window_size, hop_length))
        #global_mask = np.array([ref1_spec > ref2_spec, ref1_spec < ref2_spec]).astype('bool')

        return istft_default(window_size, hop_length, spec * global_mask[0], spec * global_mask[1], phase, phase)


#import IPython.display as ipd
#display(ipd.Audio(ref[0], rate=8000))
#display(ipd.Audio(ref[1], rate=8000))