'''
Script to train the model
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import re

import numpy as np
import tensorflow as tf
from grog.models.datagenerator import DataGenerator
from grog.models.model import Model
from grog.config import Config

def train(model_dir, sum_dir, train_pkl, val_pkl, config):
    learning_rate = config.learning_rate
    n_layers = config.n_layers
    n_hidden = config.n_hidden
    embedding_dimension = config.embedding_dimension
    windows_per_sample = config.windows_per_sample
    ft_bins = config.window_size // 2 + 1
    max_steps = config.max_steps
    batch_size = config.batch_size
    train_dropout_ff = config.train_dropout_ff
    train_dropout_rc = config.train_dropout_rc

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)

    train_loss_file = os.path.join(sum_dir, "train_loss.npy")
    val_loss_file = os.path.join(sum_dir, "val_loss.npy")

    with tf.Graph().as_default():
        # dropout probability
        dropout_ff = tf.placeholder(tf.float32, shape=None)
        dropout_rc = tf.placeholder(tf.float32, shape=None)
        # generator for training set and validation set
        data_generator = DataGenerator(train_pkl, batch_size, windows_per_sample, ft_bins)
        val_generator = DataGenerator(val_pkl, batch_size, windows_per_sample, ft_bins)
        # placeholder for input log spectrum, VAD info.,
        # and speaker indicator function
        in_data = tf.placeholder(tf.float32, shape=[batch_size, windows_per_sample, ft_bins])
        VAD_data = tf.placeholder(tf.float32, shape=[batch_size, windows_per_sample, ft_bins])
        Y_data = tf.placeholder(tf.float32, shape=[batch_size, windows_per_sample, ft_bins, 2])
        # init the model
        BiModel = Model(n_layers, embedding_dimension, windows_per_sample, config.window_size, n_hidden, batch_size, dropout_ff, dropout_rc)
        # build the net structure
        embedding = BiModel.inference(in_data)
        in_data_reshaped = tf.reshape(in_data, [-1, ft_bins])
        Y_data_reshaped = tf.reshape(Y_data, [-1, ft_bins, 2])
        VAD_data_reshaped = tf.reshape(VAD_data, [-1, ft_bins])
        # compute the loss
        loss = BiModel.loss_attractor(in_data_reshaped, embedding, Y_data_reshaped, VAD_data_reshaped)
        train_loss_summary_op = tf.summary.scalar('train_loss', loss)
        # get the train operation
        train_op = BiModel.train(loss, learning_rate)
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()

        # either train from scratch or a trained model
        seeds = [f for f in os.listdir(
            model_dir) if re.match(r'model\.ckpt.*', f)]
        if len(seeds) > 0:
            saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        init_step = 0
        if os.path.isfile(train_loss_file):
            train_loss = np.load(train_loss_file)
        else:
            train_loss = np.array([])
        if os.path.isfile(val_loss_file):
            val_loss = np.load(val_loss_file)
        else:
            val_loss = np.array([])

        summary_writer = tf.summary.FileWriter(sum_dir, sess.graph)
        last_epoch = data_generator.epoch

        for step in range(init_step, init_step + max_steps):
            start_time = time.time()
            in_data_np, Y_data_np, VAD_data_np = data_generator.gen_tf_batch()
            # train the model
            loss_value, _, summary_str = sess.run(
                [loss, train_op, train_loss_summary_op],
                feed_dict={in_data: in_data_np,
                           VAD_data: VAD_data_np,
                           Y_data: Y_data_np,
                           dropout_ff: train_dropout_ff,
                           dropout_rc: train_dropout_rc})

            assert not np.isnan(loss_value)

            summary_writer.add_summary(summary_str, step)
            sec_per_batch = time.time() - start_time

            if step % 10 == 0:
                train_loss = np.append(train_loss, loss_value.copy())

                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / sec_per_batch

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch, epoch %d)'
                )
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch,
                                    data_generator.epoch))
            if step % 50 == 0:
                # Periodically save model
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path)

                np.save(train_loss_file, train_loss)
                np.save(val_loss_file, val_loss)

            if last_epoch != data_generator.epoch:
                # doing validation every training epoch
                print('Doing validation')
                val_epoch = val_generator.epoch
                losses = []
                while(val_epoch == val_generator.epoch):
                    in_data_np, Y_data_np, VAD_data_np = val_generator.gen_tf_batch()
                    loss_value = sess.run(
                        loss,
                        feed_dict={in_data: in_data_np,
                                   VAD_data: VAD_data_np,
                                   Y_data: Y_data_np,
                                   dropout_ff: 0,
                                   dropout_rc: 0})
                    losses.append(loss_value)

                mean_loss = np.mean(losses)
                summary = tf.Summary()
                summary.value.add(tag="Validation loss", simple_value=mean_loss)
                summary_writer.add_summary(summary, step)
                val_loss = np.append(val_loss, mean_loss)
                print('validation loss: %.3f' % mean_loss)

            last_epoch = data_generator.epoch

        np.save(train_loss_file, train_loss)
        np.save(val_loss_file, val_loss)
