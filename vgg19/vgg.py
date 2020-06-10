import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]

def rgb2bgr(rgb):
    return rgb[:, :, :, ::-1]

def bgr2rgb(bgr):
    return bgr[:, :, :, ::-1]

def preprocess(image):
    return image - VGG_MEAN

def unpreprocess(image):
    return image + VGG_MEAN

class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

    def build(self, bgr, clear_data=True):
        """
            load variable from npy to build the VGG
        """
        net = {}

        net['conv1_1'] = self.conv_layer(bgr, "conv1_1")
        net['conv1_2'] = self.conv_layer(net['conv1_1'], "conv1_2")
        net['pool1'] = self.max_pool(net['conv1_2'], 'pool1')
        net['conv2_1'] = self.conv_layer(net['pool1'], "conv2_1")
        net['conv2_2'] = self.conv_layer(net['conv2_1'], "conv2_2")
        net['pool2'] = self.max_pool(net['conv2_2'], 'pool2')
        net['conv3_1'] = self.conv_layer(net['pool2'], "conv3_1")
        net['conv3_2'] = self.conv_layer(net['conv3_1'], "conv3_2")
        net['conv3_3'] = self.conv_layer(net['conv3_2'], "conv3_3")
        net['conv3_4'] = self.conv_layer(net['conv3_3'], "conv3_4")
        net['pool3'] = self.max_pool(net['conv3_4'], 'pool3')
        net['conv4_1'] = self.conv_layer(net['pool3'], "conv4_1")
        net['conv4_2'] = self.conv_layer(net['conv4_1'], "conv4_2")
        net['conv4_3'] = self.conv_layer(net['conv4_2'], "conv4_3")
        net['conv4_4'] = self.conv_layer(net['conv4_3'], "conv4_4")
        net['pool4'] = self.max_pool(net['conv4_4'], 'pool4')
        net['conv5_1'] = self.conv_layer(net['pool4'], "conv5_1")
        self.conv1_1 = net['conv1_1']
        self.conv1_2 = net['conv1_2']
        self.pool1 = net['pool1']
        self.conv2_1 = net['conv2_1']
        self.conv2_2 = net['conv2_2']
        self.pool2 = net['pool2']
        self.conv3_1 = net['conv3_1']
        self.conv3_2 = net['conv3_2']
        self.conv3_3 =net['conv3_3']
        self.conv3_4 = net['conv3_4']
        self.pool3 = net['pool3']
        self.conv4_1 = net['conv4_1']
        self.conv4_2 = net['conv4_2']
        self.conv4_3 = net['conv4_3']
        self.conv4_4 = net['conv4_4']
        self.pool4 = net['pool4']
        self.conv5_1 = net['conv5_1']
        #self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        #self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        #self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        #self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        self.net = net

    def get_fv_dict(self):
        net = {}
        net["conv1_1"] = self.conv1_1


    def get_all_layers(self):
        return [self.conv1_1, self.conv1_2, self.pool1,\
                self.conv2_1, self.conv2_2, self.pool2, \
                self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, self.pool3, \
                self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, self.pool4, \
                self.conv5_1]

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
