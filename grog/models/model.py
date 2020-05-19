'''
Class Model: model for the deep clustering speech seperation
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, RNN, LSTMCell

class Model(object):
    def __init__(self, n_layers, embedding_dimension, windows_per_sample, window_size, n_hidden, batch_size, dropout_ff, dropout_rc):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.dropout_ff = dropout_ff
        self.dropout_rc = dropout_rc
        self.windows_per_sample = windows_per_sample
        self.embedding_dimension = embedding_dimension
        self.ft_bins = window_size // 2 + 1

        # biases and weights for the last dense layer
        self.dense_weights = tf.Variable(tf.random_normal(
            [2 * n_hidden, embedding_dimension * self.ft_bins], mean=0.0, stddev=1.0))
        self.dense_biases = tf.Variable(tf.random_normal(
            [embedding_dimension * self.ft_bins], mean=0.0, stddev=1.0))

    def inference(self, x):
        '''The structure of the network'''

        # print("x.shape: %s" % x.shape)  # = (128, 100, 129)

        concated_outputs = x

        for i in range(self.n_layers):
            with tf.variable_scope('BLSTM%d' % (i + 1)) as scope:
                lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    self.n_hidden, layer_norm=False,
                    dropout_keep_prob=1 - self.dropout_rc)
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_fw_cell, input_keep_prob=1,
                    output_keep_prob=1 - self.dropout_ff)
                lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    self.n_hidden, layer_norm=False,
                    dropout_keep_prob=1 - self.dropout_rc)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_bw_cell, input_keep_prob=1,
                    output_keep_prob=1 - self.dropout_ff)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    lstm_fw_cell, lstm_bw_cell, concated_outputs,
                    sequence_length=[
                        self.windows_per_sample] * self.batch_size,
                    dtype=tf.float32)
                concated_outputs = tf.concat(outputs, 2)


                # Keras Way
                ''' mask = tf.expand_dims(tf.sequence_mask([self.windows_per_sample] * self.batch_size, dtype=tf.float32), axis=-1)
                cell = LSTMCell(self.n_hidden, activation='tanh', recurrent_activation='tanh', implementation=2, dropout=0.5, recurrent_dropout=0.5)
                bidi_layer = Bidirectional(RNN(cell, return_sequences=True, input_shape=[self.batch_size, self.windows_per_sample, self.ft_bins]), merge_mode="concat")
                concated_outputs = bidi_layer(concated_outputs, mask=mask)

                print("concated_outputs.shape: %s" % concated_outputs.shape) # = (128, 100, 1200) '''

        # one layer of embedding output with tanh activation function
        # = (12800 = 128 * 100, 1200)
        out_concate = tf.reshape(concated_outputs, [-1, self.n_hidden * 2])
        # (12800, 1200) x (1200, 40 * 129) = [40 * 129] / (12800, 40 * 129)
        emb_out = tf.matmul(out_concate, self.dense_weights) + self.dense_biases
        # print("self.dense_biases.shape: %s" %self.dense_biases.shape)  # = [40 * 129]
        emb_out = tf.nn.tanh(emb_out)
        reshaped_emb = tf.reshape(emb_out, [-1, self.ft_bins, self.embedding_dimension])  # = [12800, 129, 40]

        # normalization before output
        normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)  # = [12800, 129, 40]
        return normalized_emb

    def loss(self, embeddings, Y, VAD):
        '''Defining the loss function'''
        embeddings_rs = tf.reshape(
            embeddings, shape=[-1, self.embedding_dimension]) # Was: [-1, self.ft_bins, self.embedding_dimension]
        VAD_rs = tf.reshape(VAD, shape=[-1])
        # get the embeddings with active VAD
        embeddings_rsv = tf.transpose(tf.multiply(
            tf.transpose(embeddings_rs), VAD_rs))
        embeddings_v = tf.reshape(
            embeddings_rsv, [-1, self.windows_per_sample * self.ft_bins, self.embedding_dimension])
        # get the Y(speaker indicator function) with active VAD
        Y_rs = tf.reshape(Y, shape=[-1, 2])
        Y_rsv = tf.transpose(tf.multiply(tf.transpose(Y_rs), VAD_rs))
        Y_v = tf.reshape(
            Y_rsv, shape=[-1, self.windows_per_sample * self.ft_bins, 2])
        # fast computation format of the embedding loss function
        loss_batch = tf.nn.l2_loss(
            tf.matmul(tf.transpose(
                embeddings_v, [0, 2, 1]), embeddings_v)) - \
            2 * tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    embeddings_v, [0, 2, 1]), Y_v)) + \
            tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    Y_v, [0, 2, 1]), Y_v))
        loss_v = (loss_batch) / self.batch_size / self.windows_per_sample
        return loss_v

    def loss_attractor(self, in_data, in_ref1_data, in_ref2_data, embeddings, Y, VAD):
        embeddings_rs = tf.reshape(
            embeddings, shape=[-1, self.embedding_dimension]) # Was: [-1, self.ft_bins, self.embedding_dimension]
        VAD_rs = tf.reshape(VAD, shape=[-1])
        # get the embeddings with active VAD
        embeddings_rsv = tf.transpose(tf.multiply(
            tf.transpose(embeddings_rs), VAD_rs))
        embeddings_v = tf.reshape(
            embeddings_rsv, [-1, self.windows_per_sample * self.ft_bins, self.embedding_dimension])
        # get the Y(speaker indicator function) with active VAD
        Y_rs = tf.reshape(Y, shape=[-1, 2])
        Y_rsv = tf.transpose(tf.multiply(tf.transpose(Y_rs), VAD_rs))
        Y_v = tf.reshape(
            Y_rsv, shape=[-1, self.windows_per_sample * self.ft_bins, 2])

        mixed = tf.reshape(in_data, shape=[-1, self.windows_per_sample * self.ft_bins]) # X 
        S = mixed * tf.transpose(Y_v, [2, 0, 1]) # S 

        print("mixed: " + str(mixed.get_shape()))
        print("S: " + str(S.get_shape()))

        A = tf.matmul(tf.transpose(Y_v, [0, 2, 1]), embeddings_v) / tf.expand_dims((tf.reduce_sum(Y_v, axis=[1]) +10**-20), axis=2) # [128,2,45],/ [128, 2, 1].

        print("A: " + str(A.get_shape()))

        M = tf.nn.relu(tf.reduce_sum(tf.matmul(A, tf.transpose(embeddings_v, [0, 2, 1])), axis=1))

        print("M: " + str(M.get_shape()))

        ref1 = tf.reshape(in_ref1_data, shape=[-1, self.windows_per_sample * self.ft_bins])
        ref2 = tf.reshape(in_ref2_data, shape=[-1, self.windows_per_sample * self.ft_bins])
        loss = tf.reduce_mean(tf.square(ref1 - mixed * M) + tf.square(ref2 - (mixed - (mixed * M))), keepdims=True)
        return loss

    def train(self, loss, lr):
        '''Optimizer'''
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 200)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op
