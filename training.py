#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import argparse, os, time, json
from math import ceil
import dataset
import dataset_new
import statNet1, statNet2
from collections import deque

class Trainer:

    def __init__(self, network, params, path_for_logging, no_validation_set):

        self._net = network
        self._dropout_keep_prob = params['net_params']['dropout_keep_prob']
        self._l2_penalty = params['net_params']['l2_penalty']
        decay_steps = params['net_params']['decay_steps']
        initial_learning_rate = params['net_params']['initial_learning_rate']

        self._params = params

        validation_fraction = 0.0 if no_validation_set else 0.25
        #self._training_dataset = dataset.StatoilTrainingDataset(params['dataset_params'], validation_fraction = validation_fraction)
        self._training_dataset = dataset_new.StatoilTrainingDataset(validation_fraction = validation_fraction,
                                                                    params = params['dataset_params'])

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
            f_train.write('#examples trained on\tcross entropy\ttime\n')
            f_train.close()
            f_validation = open(self._validation_stats_filename, 'wt')
            f_validation.write('#examples trained on\tvalidation cross entropy\tvalidation accuracy\tsmoothed training cross entropy\n')
            f_validation.close()

            self._test_output_filename = os.path.join(path_for_logging, 'submission.csv')

#            # Temporary hack - remove later
#            self._validation_paths = [os.path.join(path_for_logging, 'val{}.dat'.format(i)) for i in range(1,6)]
#            for x in self._validation_paths:
#                f = open(x, 'wt')
#                f.write('#logit for iceberg\n')
#                f.close()
        else:
            self._train_stats_filename = None
            self._validation_stats_filename = None

        ## Setup placeholers and costs
        self._input_image = tf.placeholder(shape = [None, 75, 75, 2], dtype=tf.float32)
        self._keep_prob = tf.placeholder(shape = (), dtype=tf.float32)
        self._inference = tf.placeholder(shape = (), dtype=tf.bool)
        self._moments = [[tf.placeholder(shape = shape, dtype = tf.float32, name = 'moment_ph_{}_{}'.format(i, j)) for (i, shape) in enumerate(layer)] for (j, layer) in enumerate(self._net.moment_shapes)]
        self._network_logit, self._moments_out = self._net.connect(self._input_image, self._keep_prob, self._inference, self._moments)
        self._ship_logit = 1.0 - self._network_logit

        # Ordering must be this way around because label = 0 for ship, 1 for iceberg
        self._y_hat = tf.concat([self._ship_logit, self._network_logit], axis = 1)

        self._y_is_iceberg = tf.placeholder(shape=[None], dtype=tf.int32)
        self._y = tf.one_hot(self._y_is_iceberg, depth=2)

        self._cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = self._y, logits=self._y_hat)
        self._l2_cost = self._l2_penalty * self._net.get_l2_weights()
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._y_hat, axis=1, output_type=tf.int32), self._y_is_iceberg), dtype=tf.float32))

        global_step = tf.Variable(0, trainable=False)
        self._learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                                         decay_steps = decay_steps, decay_rate = 0.5)
        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cross_entropy + self._l2_cost)

        ## Setup stats to track the training
        self._smoothing_decay = 0.95
        self._smoothed_cross_entropy = None
        self._num_examples_trained_on = 0
        self._training_seconds = 0.0
        self._validation_seconds = 0.0

        ## Setup the arrays we will use to keep aggregated means and variances
        self._num_batches_to_track = 1604 // 64
        #self._num_batches_to_track = 1
        # This will hold a record of the means and variances for the last N batches
        self._moment_values_history = [[np.zeros(shape = [self._num_batches_to_track] + shape) for shape in layer] for layer in self._net.moment_shapes]
        # Which batch in the cirular list to update next
        self._moment_history_cursor = 0
        # This is where we'll write the mean-of-means and mean-of-variances
        self._moment_values = [[np.zeros(shape = shape) for shape in layer] for layer in self._net.moment_shapes]

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
            time = self._training_seconds + self._validation_seconds
            f_train.write('{}\t{}\t{}\n'.format(self._num_examples_trained_on, self._smoothed_cross_entropy, time))
            f_train.close()
    
    def add_to_moment_history(self, moments):
        """ Update the appropriate member of the history array,
        and advance the cursor.
        This should be done after every training round. (?)
        """
        for (history, new) in zip(self._moment_values_history, moments):
            for (x, y) in zip(history, new):
                x[self._moment_history_cursor, :] = y
        self._moment_history_cursor = (self._moment_history_cursor + 1) % self._num_batches_to_track

    def update_moment_values(self):
        """ Average the means and variances in our array of moment
        histories. This should be done before inference.
        """
        for i in range(self._num_batches_to_track):
            image_batch, label_batch = self._training_dataset.get_next_training_batch(batch_size)
            feed_dict = {self._input_image : image_batch,
                         self._y_is_iceberg : label_batch,
                         self._inference : False,
                         self._keep_prob : 1.0}
            feed_dict.update(self.get_moments_dict())
            moments_out = self._sess.run(self._moments_out,
                                         feed_dict = feed_dict)
            self.add_to_moment_history(moments_out)

        for (av, history) in zip(self._moment_values, self._moment_values_history):
            for (x, y) in zip(av, history):
                x[:] = np.mean(y, axis = 0)

    def get_moments_dict(self):
        return { x : y for layer in zip(self._moments, self._moment_values)
                            for (x, y) in zip(layer[0], layer[1])}

    def train_batch(self, batch_size):

        start_time = time.time()

        image_batch, label_batch = self._training_dataset.get_next_training_batch(batch_size)
        feed_dict = {self._input_image : image_batch,
                     self._y_is_iceberg : label_batch,
                     self._inference : False,
                     self._keep_prob : self._dropout_keep_prob}
        feed_dict.update(self.get_moments_dict())
        _, ce, moments_out = self._sess.run([self._train_step, self._cross_entropy, self._moments_out],
                                             feed_dict = feed_dict)
        #self.add_to_moment_history(moments_out)

        self._num_examples_trained_on += batch_size
        self.update_stats(ce)

        end_time = time.time()
        self._training_seconds += (end_time - start_time)

    def get_and_print_validation_stats(self, batch_size):

        start_time = time.time()

        self.update_moment_values()

        batches = self._training_dataset.get_validation_set(batch_size)

        n = 0
        acc = 0
        ce = 0

        #batch_zero = True

        for val in self._moment_values:
            val = val[1] # Variance
            mean_var = np.mean(val, axis=None)
            print('Mean var: {}'.format(mean_var))

        for (image_batch, label_batch) in batches:
            feed_dict = {self._input_image : image_batch,
                         self._y_is_iceberg : label_batch,
                         self._inference : True,
                         self._keep_prob : 1.0}
            feed_dict.update(self.get_moments_dict())
            b_acc, b_ce = self._sess.run([self._accuracy, self._cross_entropy],
                                         feed_dict = feed_dict)
            b_n = image_batch.shape[0]
            acc += b_n*b_acc
            ce += b_n*b_ce
            n += b_n

#            if batch_zero:
#                logits = self._sess.run([self._network_logit], feed_dict = {self._input_image : image_batch, self._y_is_iceberg : label_batch,
#                                                                            self._keep_prob : 1.0})
#                #print('logits: {}'.format(logits[0]))
#                for (i, p) in zip(range(1, 6), self._validation_paths):
#                    f = open(p, 'at')
#                    f.write('{}\n'.format(logits[0][i-1, 0])) 
#                    f.close()
#                batch_zero = False

        acc /= n
        ce /= n

        print('Validation cross-entropy: {}'.format(ce))
        print('Validation accuracy: {}'.format(acc))

        if self._validation_stats_filename is not None:
            f_validation = open(self._validation_stats_filename, 'at')
            f_validation.write('{}\t{}\t{}\t{}\n'.format(self._num_examples_trained_on, ce, acc, self._smoothed_cross_entropy))
            f_validation.close()

        end_time = time.time()
        self._validation_seconds += (end_time - start_time)

    def print_times(self):
        print('Time spent training: {} seconds'.format(self._training_seconds))
        print('Time spent validating: {} seconds'.format(self._validation_seconds))

    def make_predictions(self, batch_size=32):
        """Make predictions using the test data predictors.
        Write the results to a file in the log directory.
        Currently this function will just overwrite any previous
        predictions made by the same Trainer.
        """

        print('Making predictions for the test set...')

        # May have to modify this if the test dataset is too large for memory
        test_dataset = dataset.StatoilTrainingDataset(self._params['dataset_params'], filename='./data/test.json',
                                                     validation_fraction = 0.0)
        batches = test_dataset.get_all_images_without_augmentation(batch_size)

        self.update_moment_values()

        f = open(self._test_output_filename, 'wt')
        f.write('id,is_iceberg\n')

        for (image_ids, images) in batches:
            feed_dict = {self._input_image : images,
                         self._keep_prob : 1.0,
                         inference : False}
            feed_dict.update(self.get_moments_dict())
            [batch_ice_logits, batch_ship_logits] = \
                    self._sess.run([self._network_logit, self._ship_logit],
                                   feed_dict = feed_dict)
            logits_combined = np.concatenate([batch_ice_logits, batch_ship_logits], axis=1)
            c_0 = np.max(logits_combined, axis=1)
            logits_safe = logits_combined - np.stack([c_0, c_0], axis=1)
            denom = np.sum(np.exp(logits_safe), 1)
            batch_p_iceberg = np.exp(logits_safe[:, 0]) / denom

            for (i, p) in zip(image_ids, batch_p_iceberg):
                f.write('{},{}\n'.format(i, p))
            
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, choices = ['statNet1', 'statNet2'])
    parser.add_argument('--epochs', type=int, default = 10)
    parser.add_argument('--logdir', type=str, default = './logs')
    parser.add_argument('--l2penalty', type=float, default = '2e-2')
    parser.add_argument('--with-predictions', action='store_true')
    parser.add_argument('--no-validation-set', action='store_true')
    parser.add_argument('--shuffle-dataset', dest='shuffle_dataset', action='store_true')
    parser.add_argument('--do-not-shuffle-dataset', dest='shuffle_dataset', action='store_false')
    parser.set_defaults(shuffle_dataset = True)
    args = parser.parse_args()

    params = {
        'dataset_params' : {
            'flips' : True,
            'demean' : False,
            'range_normalize' : True,
            'exponentiate_base' : None,
            'add_noise' : False,
            'shuffle' : args.shuffle_dataset
        },
        'net_params' : {
            'net' : args.net,
            'conv1_size' : 5,
            'conv1_channels' : 16,
            'conv2_size' : 5,
            'conv2_channels' : 16,
            'fc1_size' : 1024,
            'fc2_size' : 128,
            'dropout_keep_prob': 1.0,
            'initial_learning_rate' : 1e-4,
            'decay_steps' : 200,
            'l2_penalty' : args.l2penalty
        }
    }

    if args.net == 'statNet1':
        net = statNet1.StatNet1(params['net_params'])
    elif args.net == 'statNet2':
        net = statNet2.StatNet2(params['net_params'])
    else:
        raise ValueError('Net type {} is unknown.'.format(args.net))

    batch_size = 64
    trainer = Trainer(net, params, args.logdir, no_validation_set = args.no_validation_set)
    batches_per_epoch = trainer._training_dataset._N_train / batch_size

    print('Will train for {} batches.'.format(ceil(args.epochs * batches_per_epoch)))
    for i in range(ceil(args.epochs * batches_per_epoch)):
        trainer.train_batch(batch_size)
        
        if i%10 == 0:
            trainer.print_training_stats()

        if (i > batches_per_epoch) and i%25 == 0 and not args.no_validation_set:
            trainer.get_and_print_validation_stats(batch_size)

    if args.with_predictions:
        trainer.make_predictions(batch_size)

    trainer.print_times()
