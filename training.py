#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import argparse
import dataset
import statNet1

class Trainer:

    def __init__(self, network):

        self._net = network

        self._training_dataset = dataset.StatoilTrainingDataset()

        ## Setup placeholers and costs
        self._input_image = tf.placeholder(shape = [None, 75, 75, 2], dtype=tf.float32)
        self._network_logit = self._net.connect(self._input_image)
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

        ## Initialize the session
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def update_stats(self, training_cross_entropy, num_trained_on):
        if self._smoothed_cross_entropy is None:
            self._smoothed_cross_entropy = training_cross_entropy
        else:
            self._smoothed_cross_entropy = self._smoothed_cross_entropy * self._smoothing_decay \
                                           + training_cross_entropy * (1.0 - self._smoothing_decay)
        self._num_examples_trained_on += num_trained_on

    def print_training_stats(self):
        print('Trained on {} images ({} epochs)'.format(self._num_examples_trained_on,
                                                        self._num_examples_trained_on / self._training_dataset._N_train))
        print('Smoothed cross entropy: {}'.format(self._smoothed_cross_entropy))

    def train_batch(self, batch_size):

        image_batch, label_batch = self._training_dataset.get_next_training_batch(batch_size)
        _, ce = self._sess.run([self._train_step, self._cross_entropy],
                                feed_dict = {self._input_image : image_batch, self._y_is_iceberg : label_batch})
        self.update_stats(ce, batch_size)

    def get_and_print_validation_stats(self, batch_size):

        batches = self._training_dataset.get_validation_set(batch_size)

        n = 0
        acc = 0
        ce = 0
        for (image_batch, label_batch) in batches:
            b_acc, b_ce = self._sess.run([self._accuracy, self._cross_entropy],
                                         feed_dict={self._input_image : image_batch, self._y_is_iceberg : label_batch})
            b_n = image_batch.shape[0]
            acc += b_n*b_acc
            ce += b_n*b_ce
            n += b_n
        acc /= n
        ce /= n

        print('Validation cross-entropy: {}'.format(ce))
        print('Validation accuracy: {}'.format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, choices = ['statNet1'])
    parser.add_argument('--epochs', type=int, default = 10)
    args = parser.parse_args()

    if args.net == 'statNet1':
        net = statNet1.StatNet1()
    else:
        raise ValueError('Net type {} is unknown.'.format(args.net))

    batch_size = 32
    trainer = Trainer(net)

    for i in range(500):
        trainer.train_batch(batch_size)
        
        if i%10 == 0:
            trainer.print_training_stats()

        if i%100 == 0:
            trainer.get_and_print_validation_stats(batch_size)
