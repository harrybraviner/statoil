#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import argparse, os, time, json
from math import ceil
import dataset
import statNet1

class Trainer:

    def __init__(self, network, params, path_for_logging):

        self._net = network
        self._dropout_keep_prob = params['net_params']['dropout_keep_prob']

        self._training_dataset = dataset.StatoilTrainingDataset(params['dataset_params'])

        if path_for_logging is not None:
            if not os.path.exists(path_for_logging):
                os.mkdir(path_for_logging)
            f_params = open(os.path.join(path_for_logging, 'params'), 'w')
            json.dump(params, f_params, indent=4)
            f_params.close()
            self._train_stats_filename = os.path.join(path_for_logging, 'train.dat')
            self._validation_stats_filename = os.path.join(path_for_logging, 'validation.dat')
            # Write headers
            f_train = open(self._train_stats_filename, 'wt')
            f_train.write('#examples trained on\tcross entropy\n')
            f_train.close()
            f_validation = open(self._validation_stats_filename, 'wt')
            f_validation.write('#examples trained on\tcross entropy\taccuracy\n')
            f_validation.close()

            # Temporary hack - remove later
            self._validation_paths = [os.path.join(path_for_logging, 'val{}.dat'.format(i)) for i in range(1,6)]
            for x in self._validation_paths:
                f = open(x, 'wt')
                f.write('#logit for iceberg\n')
                f.close()
        else:
            self._train_stats_filename = None
            self._validation_stats_filename = None

        ## Setup placeholers and costs
        self._input_image = tf.placeholder(shape = [None, 75, 75, 2], dtype=tf.float32)
        self._keep_prob = tf.placeholder(shape = (), dtype=tf.float32)
        self._network_logit = self._net.connect(self._input_image, self._keep_prob)
        self._ship_logit = 1.0 - self._network_logit

        # Ordering must be this way around because label = 0 for ship, 1 for iceberg
        self._y_hat = tf.concat([self._ship_logit, self._network_logit], axis = 1)

        self._y_is_iceberg = tf.placeholder(shape=[None], dtype=tf.int32)
        self._y = tf.one_hot(self._y_is_iceberg, depth=2)

        self._cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = self._y, logits=self._y_hat)
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._y_hat, axis=1, output_type=tf.int32), self._y_is_iceberg), dtype=tf.float32))

        self._train_step = tf.train.AdamOptimizer().minimize(self._cross_entropy)

        ## Setup stats to track the training
        self._smoothing_decay = 0.95
        self._smoothed_cross_entropy = None
        self._num_examples_trained_on = 0
        self._training_seconds = 0.0
        self._validation_seconds = 0.0

        ## Initialize the session
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def update_stats(self, training_cross_entropy):
        if self._smoothed_cross_entropy is None:
            self._smoothed_cross_entropy = training_cross_entropy
        else:
            self._smoothed_cross_entropy = self._smoothed_cross_entropy * self._smoothing_decay \
                                           + training_cross_entropy * (1.0 - self._smoothing_decay)

    def print_training_stats(self):
        print('Trained on {} images ({} epochs)'.format(self._num_examples_trained_on,
                                                        self._num_examples_trained_on / self._training_dataset._N_train))
        print('Smoothed cross entropy: {}'.format(self._smoothed_cross_entropy))

        if self._train_stats_filename is not None:
            f_train = open(self._train_stats_filename, 'at')
            f_train.write('{}\t{}\n'.format(self._num_examples_trained_on, self._smoothed_cross_entropy))
            f_train.close()

    def train_batch(self, batch_size):

        start_time = time.time()

        image_batch, label_batch = self._training_dataset.get_next_training_batch(batch_size)
        _, ce = self._sess.run([self._train_step, self._cross_entropy],
                                feed_dict = {self._input_image : image_batch, self._y_is_iceberg : label_batch,
                                             self._keep_prob : self._dropout_keep_prob})

        self._num_examples_trained_on += batch_size
        self.update_stats(ce)

        end_time = time.time()
        self._training_seconds += (end_time - start_time)

    def get_and_print_validation_stats(self, batch_size):

        start_time = time.time()

        batches = self._training_dataset.get_validation_set(batch_size)

        n = 0
        acc = 0
        ce = 0

        batch_zero = True

        for (image_batch, label_batch) in batches:
            b_acc, b_ce = self._sess.run([self._accuracy, self._cross_entropy],
                                         feed_dict={self._input_image : image_batch, self._y_is_iceberg : label_batch,
                                                    self._keep_prob : 1.0})
            b_n = image_batch.shape[0]
            acc += b_n*b_acc
            ce += b_n*b_ce
            n += b_n

            if batch_zero:
                logits = self._sess.run([self._network_logit], feed_dict = {self._input_image : image_batch, self._y_is_iceberg : label_batch,
                                                                            self._keep_prob : 1.0})
                #print('logits: {}'.format(logits[0]))
                for (i, p) in zip(range(1, 6), self._validation_paths):
                    f = open(p, 'at')
                    f.write('{}\n'.format(logits[0][i-1, 0])) 
                    f.close()
                batch_zero = False

        acc /= n
        ce /= n

        print('Validation cross-entropy: {}'.format(ce))
        print('Validation accuracy: {}'.format(acc))

        if self._validation_stats_filename is not None:
            f_validation = open(self._validation_stats_filename, 'at')
            f_validation.write('{}\t{}\t{}\n'.format(self._num_examples_trained_on, ce, acc))
            f_validation.close()

        end_time = time.time()
        self._validation_seconds += (end_time - start_time)

    def print_times(self):
        print('Time spent training: {} seconds'.format(self._training_seconds))
        print('Time spent validating: {} seconds'.format(self._validation_seconds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, choices = ['statNet1'])
    parser.add_argument('--epochs', type=int, default = 10)
    parser.add_argument('--logdir', type=str, default = './logs')
    args = parser.parse_args()

    params = {
        'dataset_params' : {
            'flips' : False,
            'demean' : False,
            'range_normalize' : True,
            'exponentiate_base' : None,
            'add_noise' : False,
        },
        'net_params' : {
            'conv1_size' : 5,
            'conv1_channels' : 32,
            'conv2_size' : 5,
            'conv2_channels' : 32,
            'fc1_size' : 1024,
            'fc2_size' : 128,
            'dropout_keep_prob': 0.8
        }
    }

    if args.net == 'statNet1':
        net = statNet1.StatNet1(params['net_params'])
    else:
        raise ValueError('Net type {} is unknown.'.format(args.net))

    batch_size = 32
    trainer = Trainer(net, params, args.logdir)
    batches_per_epoch = trainer._training_dataset._N_train / batch_size

    print('Will train for {} batches.'.format(ceil(args.epochs * batches_per_epoch)))
    for i in range(ceil(args.epochs * batches_per_epoch)):
        trainer.train_batch(batch_size)
        
        if i%10 == 0:
            trainer.print_training_stats()

        if i%25 == 0:
            trainer.get_and_print_validation_stats(batch_size)

    trainer.print_times()
