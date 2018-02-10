#! /usr/bin/python3
import unittest
import tensorflow as tf
from math import ceil

def _make_weight(shape):
    init_val = tf.truncated_normal(shape = shape, mean=0.0, stddev=0.05)
    return tf.Variable(init_val, dtype=tf.float32)

def _make_bias(shape):
    return tf.Variable(tf.zeros(shape=shape, dtype=tf.float32))

class StatNet2:

    def __init__(self, params, image_shape = [75, 75, 2]):
        
        self._built = False
        self._input_shape = image_shape
        self._params = params.copy()

        self.build()

        self._epsilon = 1e-2    # Constant to regularize inverse variance

    def build(self):

        if not self._built:
            input_shape = self._input_shape

            conv_layer_1_size     = self._params['conv1_size']
            conv_layer_1_channels = self._params['conv1_channels']
            conv_layer_2_size     = self._params['conv2_size']
            conv_layer_2_channels = self._params['conv2_channels']

            fc_layer_1_size = self._params['fc1_size']
            fc_layer_2_size = self._params['fc2_size']

            self._W_conv_1 = _make_weight([conv_layer_1_size, conv_layer_1_size, input_shape[2], conv_layer_1_channels])
            self._gamma_conv_1 = tf.Variable(tf.ones([conv_layer_1_channels]))
            self._beta_conv_1 = tf.Variable(tf.zeros([conv_layer_1_channels]))

            self._W_conv_2 = _make_weight([conv_layer_2_size, conv_layer_2_size, conv_layer_1_channels, conv_layer_2_channels])
            self._gamma_conv_2 = tf.Variable(tf.ones([conv_layer_2_channels]))
            self._beta_conv_2 = tf.Variable(tf.zeros([conv_layer_2_channels]))

            self._pooled_flat_size = ceil(ceil(input_shape[0]/2)/2) * ceil(ceil(input_shape[1]/2)/2) * conv_layer_2_channels
            self._W_fc_1 = _make_weight([self._pooled_flat_size, fc_layer_1_size])
            self._gamma_fc_1 = tf.Variable(tf.ones([fc_layer_1_size]))
            self._beta_fc_1 = tf.Variable(tf.zeros([fc_layer_1_size]))

            self._W_fc_2 = _make_weight([fc_layer_1_size, fc_layer_2_size])
            self._gamma_fc_2 = tf.Variable(tf.ones([fc_layer_2_size]))
            self._beta_fc_2 = tf.Variable(tf.zeros([fc_layer_2_size]))

            self._W_fc_3 = _make_weight([fc_layer_2_size, 1])
            self._b_fc_3 = _make_bias([1])

            self._built = True

    @property
    def moment_shapes(self):
        return [[[self._params['conv1_channels']]]*2,
                [[self._params['conv2_channels']]]*2,
                [[self._params['fc1_size']]]*2, [[self._params['fc2_size']]]*2]

    def connect(self, x, keep_prob, inference, moments):

        if not self._built:
            self.build()

        def connect_bn_conv_layer(u, W, mean_in, var_in, beta, gamma, name):
            x = tf.nn.conv2d(u, W, strides=[1,1,1,1], padding = 'SAME', name = name + "_conv")
            mean_batch, var_batch = tf.nn.moments(x, axes=[0, 1, 2], name = name + "_mom")
            moment_output = [mean_batch, var_batch]
            mean_to_use = tf.cond(inference, true_fn = lambda: mean_in, false_fn = lambda: mean_batch)
            var_to_use  = tf.cond(inference, true_fn = lambda: var_in,  false_fn = lambda: var_batch )
            x_hat = tf.nn.batch_normalization(x, mean_to_use, var_to_use, beta, gamma,
                                              self._epsilon, name = name + "_bn")
            act = tf.nn.relu(x_hat, name = name + "_act")
            pool = tf.nn.max_pool(act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = name + "_act")
            return pool, moment_output

        def connect_fc_layer(u, W, mean_in, var_in, beta, gamma, name):
            x = tf.matmul(u, W, name = name + "_matmul")
            mean_batch, var_batch = tf.nn.moments(x, axes=[0], name = name + "_mom")
            moment_output = [mean_batch, var_batch]
            mean_to_use = tf.cond(inference, true_fn = lambda: mean_in, false_fn = lambda: mean_batch)
            var_to_use  = tf.cond(inference, true_fn = lambda: var_in,  false_fn = lambda: var_batch )
            x_hat = tf.nn.batch_normalization(x, mean_to_use, var_to_use, beta, gamma,
                                              self._epsilon, name = name + "_bn")
            act = tf.nn.relu(x_hat, name = name + "_act")
            dropped = tf.nn.dropout(act, keep_prob, name = name + "_drop")
            return dropped, moment_output

        mean_conv_1, var_conv_1 = moments[0]
        z_conv_1, moments_out_conv1 = connect_bn_conv_layer(x, self._W_conv_1, mean_conv_1, var_conv_1,
                                                            self._beta_conv_1, self._gamma_conv_1, "conv_1")

        mean_conv_2, var_conv_2 = moments[1]
        z_conv_2, moments_out_conv2 = connect_bn_conv_layer(z_conv_1, self._W_conv_2, mean_conv_2, var_conv_2,
                                                            self._beta_conv_2, self._gamma_conv_2, "conv_2")
        z_conv_2_flat = tf.reshape(z_conv_2, shape = [-1, self._pooled_flat_size], name = "conv_2_flat")

        mean_fc_1, var_fc_1 = moments[2]
        z_fc_1, moments_out_fc1 = connect_fc_layer(z_conv_2_flat, self._W_fc_1, mean_fc_1, var_fc_1,
                                                   self._beta_fc_1, self._gamma_fc_1, "fc_1")

        mean_fc_2, var_fc_2 = moments[3]
        z_fc_2, moments_out_fc2 = connect_fc_layer(z_fc_1, self._W_fc_2, mean_fc_2, var_fc_2,
                                                   self._beta_fc_2, self._gamma_fc_2, "fc_2")

        self._output_logit = tf.matmul(z_fc_2, self._W_fc_3) + self._b_fc_3

        moments_out = [moments_out_conv1, moments_out_conv2, moments_out_fc1, moments_out_fc2]

        return self._output_logit, moments_out

    def get_l2_weights(self):
        if not self._built:
            self.build()

        total_l2_loss =   tf.nn.l2_loss(self._W_conv_1) \
                        + tf.nn.l2_loss(self._W_conv_2) \
                        + tf.nn.l2_loss(self._W_fc_1) \
                        + tf.nn.l2_loss(self._W_fc_2) \
                        + tf.nn.l2_loss(self._W_fc_3)   + tf.nn.l2_loss(self._b_fc_3) \
                        + tf.nn.l2_loss(self._gamma_conv_1) + tf.nn.l2_loss(self._beta_conv_1) \
                        + tf.nn.l2_loss(self._gamma_conv_2) + tf.nn.l2_loss(self._beta_conv_2) \
                        + tf.nn.l2_loss(self._gamma_fc_1) + tf.nn.l2_loss(self._beta_fc_1) \
                        + tf.nn.l2_loss(self._gamma_fc_2) + tf.nn.l2_loss(self._beta_fc_2)

        return total_l2_loss

class StatNet2Tests(unittest.TestCase):

    @property
    def default_params(self):
        params = {
            'conv1_size': 4,
            'conv1_channels': 2,
            'conv2_size': 5,
            'conv2_channels': 7,
            'fc1_size': 13,
            'fc2_size': 23
        }

        return params

    def test_init_and_building(self):
        with tf.variable_scope('test_init_and_building'):
            s_net = StatNet2(self.default_params)

    def test_connect(self):
        with tf.variable_scope('test_connect'):
            s_net = StatNet2(self.default_params)

            x = tf.placeholder(shape=[13, 75, 75, 2], dtype=tf.float32)
            keep_prob = tf.placeholder(shape = (), dtype = tf.float32)
            inference = tf.placeholder(shape = (), dtype = tf.bool)
            moments = [[tf.placeholder(shape = [self.default_params['conv1_channels']], dtype = tf.float32),
                        tf.placeholder(shape = [self.default_params['conv1_channels']], dtype = tf.float32)],
                       [tf.placeholder(shape = [self.default_params['conv2_channels']], dtype = tf.float32),
                        tf.placeholder(shape = [self.default_params['conv2_channels']], dtype = tf.float32)],
                       [tf.placeholder(shape = [self.default_params['fc1_size']], dtype = tf.float32),
                        tf.placeholder(shape = [self.default_params['fc1_size']], dtype = tf.float32)],
                       [tf.placeholder(shape = [self.default_params['fc2_size']], dtype = tf.float32),
                        tf.placeholder(shape = [self.default_params['fc2_size']], dtype = tf.float32)]]
            y, moments_out = s_net.connect(x, keep_prob=keep_prob, inference=inference, moments = moments)

            self.assertEqual([yy.value for yy in y.shape], [13, 1])
            self.assertEqual(len(moments), len(moments_out))
            for (moment, moment_out) in zip(moments, moments_out):
                self.assertEqual(len(moment_out), 2)
                self.assertEqual(moment[0].shape, moment_out[0].shape)
                self.assertEqual(moment[1].shape, moment_out[1].shape)


