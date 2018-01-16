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
            self._b_conv_1 = _make_bias([conv_layer_1_channels])

            self._gamma_conv_1 = tf.Variable(tf.ones([conv_layer_1_channels]))
            self._beta_conv_1 = tf.Variable(tf.zeros([conv_layer_1_channels]))

            self._W_conv_2 = _make_weight([conv_layer_2_size, conv_layer_2_size, conv_layer_1_channels, conv_layer_2_channels])
            self._b_conv_2 = _make_bias([conv_layer_2_channels])

            self._gamma_conv_2 = tf.Variable(tf.ones([conv_layer_2_channels]))
            self._beta_conv_2 = tf.Variable(tf.zeros([conv_layer_2_channels]))

            self._pooled_flat_size = ceil(ceil(input_shape[0]/2)/2) * ceil(ceil(input_shape[1]/2)/2) * conv_layer_2_channels
            self._W_fc_1 = _make_weight([self._pooled_flat_size, fc_layer_1_size])
            self._b_fc_1 = _make_bias([fc_layer_1_size])

            self._gamma_fc_1 = tf.Variable(tf.ones([fc_layer_1_size]))
            self._beta_fc_1 = tf.Variable(tf.zeros([fc_layer_1_size]))

            self._W_fc_2 = _make_weight([fc_layer_1_size, fc_layer_2_size])
            self._b_fc_2 = _make_bias([fc_layer_2_size])

            self._gamma_fc_2 = tf.Variable(tf.ones([fc_layer_2_size]))
            self._beta_fc_2 = tf.Variable(tf.zeros([fc_layer_2_size]))

            self._W_fc_3 = _make_weight([fc_layer_2_size, 1])
            self._b_fc_3 = _make_bias([1])

            self._built = True

    def connect(self, x, keep_prob):

        if not self._built:
            self.build()

        self._h_conv_1 = tf.nn.conv2d(x, self._W_conv_1, strides=[1,1,1,1], padding = 'SAME') + self._b_conv_1
        self._mean_conv_1, self._var_conv_1 = tf.nn.moments(self._h_conv_1, axes=[0])
        self._bn_conv_1 = tf.nn.batch_normalization(self._h_conv_1, self._mean_conv_1, self._var_conv_1,
                                                    self._beta_conv_1, self._gamma_conv_1, self._epsilon)
        self._act_conv_1 = tf.nn.relu(self._bn_conv_1)
        self._h_pool_1 = tf.nn.max_pool(self._act_conv_1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
                
        self._h_conv_2 = tf.nn.conv2d(self._h_pool_1, self._W_conv_2, strides=[1,1,1,1], padding = 'SAME') + self._b_conv_2
        self._mean_conv_2, self._var_conv_2 = tf.nn.moments(self._h_conv_2, axes=[0])
        self._bn_conv_2 = tf.nn.batch_normalization(self._h_conv_2, self._mean_conv_2, self._var_conv_2,
                                                    self._beta_conv_2, self._gamma_conv_2, self._epsilon)
        self._act_conv_2 = tf.nn.relu(self._bn_conv_2)
        self._h_pool_2 = tf.nn.max_pool(self._act_conv_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
        self._h_pool_2_flat = tf.reshape(self._h_pool_2, shape = [-1, self._pooled_flat_size])

        self._h_fc_1 = tf.matmul(self._h_pool_2_flat, self._W_fc_1) + self._b_fc_1
        self._mean_fc_1, self._var_fc_1 = tf.nn.moments(self._h_fc_1, axes=[0])
        self._bn_fc_1 = tf.nn.batch_normalization(self._h_fc_1, self._mean_fc_1, self._var_fc_1,
                                                  self._beta_fc_1, self._gamma_fc_1, self._epsilon)
        self._act_fc_1 = tf.nn.relu(self._bn_fc_1)
        self._h_fc_1_dropped = tf.nn.dropout(self._act_fc_1, keep_prob)

        self._h_fc_2 = tf.matmul(self._h_fc_1_dropped, self._W_fc_2) + self._b_fc_2
        self._mean_fc_2, self._var_fc_2 = tf.nn.moments(self._h_fc_2, axes=[0])
        self._bn_fc_2 = tf.nn.batch_normalization(self._h_fc_2, self._mean_fc_2, self._var_fc_2,
                                                  self._beta_fc_2, self._gamma_fc_2, self._epsilon)
        self._act_fc_2 = tf.nn.relu(self._bn_fc_2)
        self._h_fc_2_dropped = tf.nn.dropout(self._act_fc_2, keep_prob)

        self._output_logit = tf.matmul(self._h_fc_2_dropped, self._W_fc_3) + self._b_fc_3

        return self._output_logit

    def get_l2_weights(self):
        if not self._built:
            self.build()

        total_l2_loss =   tf.nn.l2_loss(self._W_conv_1) + tf.nn.l2_loss(self._b_conv_1) \
                        + tf.nn.l2_loss(self._W_conv_2) + tf.nn.l2_loss(self._b_conv_2) \
                        + tf.nn.l2_loss(self._W_fc_1)   + tf.nn.l2_loss(self._b_fc_1) \
                        + tf.nn.l2_loss(self._W_fc_2)   + tf.nn.l2_loss(self._b_fc_2) \
                        + tf.nn.l2_loss(self._W_fc_3)   + tf.nn.l2_loss(self._b_fc_3)

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
            y = s_net.connect(x, keep_prob=0.5)

            self.assertEqual([yy.value for yy in y.shape], [13, 1])


