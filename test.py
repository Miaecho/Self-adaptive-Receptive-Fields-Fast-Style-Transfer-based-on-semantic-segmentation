import tensorflow as tf
import numpy as np
from scipy.misc import imresize
WEIGHTS_INIT_STDEV = .1
import tensorflow as tf
from PIL import Image
import sys
from scipy.misc import imresize
import functools
import vgg19.vgg as vgg
def load_image(path, shape=None, crop='center'):
    img = Image.open(path).convert("RGB")

    if isinstance(shape, (list, tuple)):
        # crop to obtain identical aspect ratio to shape
        width, height = img.size
        target_width, target_height = shape[0], shape[1]

        aspect_ratio = width / float(height)
        target_aspect = target_width / float(target_height)

        if aspect_ratio > target_aspect: # if wider than wanted, crop the width
            new_width = int(height * target_aspect)
            if crop == 'right':
                img = img.crop((width - new_width, 0, width, height))
            elif crop == 'left':
                img = img.crop((0, 0, new_width, height))
            else:
                img = img.crop(((width - new_width) / 2, 0, (width + new_width) / 2, height))
        else: # else crop the height
            new_height = int(width / target_aspect)
            if crop == 'top':
                img = img.crop((0, 0, width, new_height))
            elif crop == 'bottom':
                img = img.crop((0, height - new_height, width, height))
            else:
                img = img.crop((0, (height - new_height) / 2, width, (height + new_height) / 2))

        # resize to target now that we have the correct aspect ratio
        img = img.resize((target_width, target_height))
    elif isinstance(shape, (int, float)):
        width, height = img.size
        large = max(width, height)
        ratio = shape / float(large)
        width_n, height_n = ratio * width, ratio * height
        img = img.resize((int(width_n), int(height_n)))
    return img

def shortcut_interpolation(image, sc, factor):
    # Valid factor range [0.0, 2.0], detailed interpolation are showing as follow:
    #   (1) When factor is in [0.0, 1.0], the stroke are combined with (1 - factor) * 256 + factor * 512
    #   (2) When factor is in [1.0, 2.0], the stroke are combined with (2 - factor) * 512 + (factor - 1) * 768
    # As a consequence of this design, the stroke will grow from 256 to 768 when factor grows from 0 to 2
    alpha = tf.cond(sc[0],
        lambda: tf.cond(sc[1],
            lambda: tf.maximum(0.0, 1.0 - factor),
            lambda: tf.constant(1.0)
        ),
        lambda: tf.constant(0.0)
    )
    beta = tf.cond(sc[0],
        lambda: tf.cond(sc[1],
            lambda: 1.0 - tf.sign(factor - 1.0) * (factor - 1.0),
            lambda: tf.constant(0.0)
        ),
        lambda: tf.cond(sc[1],
            lambda: tf.constant(1.0),
            lambda: tf.constant(0.0)
        )
    )
    gamma = tf.cond(sc[0],
        lambda: tf.cond(sc[1],
            lambda: tf.maximum(factor - 1.0, 0.0),
            lambda: tf.constant(0.0)
        ),
        lambda: tf.cond(sc[1],
            lambda: tf.constant(0.0),
            lambda: tf.constant(1.0)
        )
    )
    conv1 = _conv_layer(image, 16, 3, 1)
    conv2 = _conv_layer(conv1, 32, 3, 2)
    conv3 = _conv_layer(conv2, 48, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid4_1 = _residual_block(resid4, 3)
    resid5 = alpha * resid3 + beta * resid4 + gamma * resid4_1
    conv_t1 = _conv_tranpose_layer(resid5, 32, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 16, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net

def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 48, filter_size, 1)
    return net + _conv_layer(tmp, 48, filter_size, 1, relu=False)

def _instance_norm(net, train=True):
    in_channels = net.get_shape().as_list()[3]
    var_shape = [in_channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu) / (sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size = net.get_shape().as_list()[0]
    channels = net.get_shape().as_list()[3]

    net_shape = tf.shape(net)
    rows, cols = net_shape[1], net_shape[2]

    new_rows, new_cols = rows * strides, cols * strides
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, new_shape, strides_shape, padding='SAME')
    net = tf.reshape(net, [batch_size, new_rows, new_cols, num_filters])
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    in_channels = net.get_shape().as_list()[3]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
def fill_feed_dict(self, content_pl, img_size=None):
    content_images = np.zeros((self.batch_size,) + img_size, dtype=np.float32)
    for i in xrange(self.batch_size):
        img = np.array(load_image(self.mscoco_fnames[self.perm[self.batch_idx * self.batch_size + i]], shape=img_size), dtype=np.float32)
        content_images[i] = img

    self.batch_idx += 1
    if self.batch_idx == self.nbatches:
        self.batch_idx = 0
        self.epochs += 1
        self.perm = np.random.permutation(self.train_size)

    return {content_pl: content_images}
VGG_MEAN = [103.939, 116.779, 123.68]
def rgb2bgr(rgb):
    return rgb[:, :, :, ::-1]
def preprocess(image):
    return image - VGG_MEAN
shortcut = tf.placeholder_with_default([False, False], shape=[2], name="shortcut")
content_input = tf.placeholder(tf.float32, shape=(1, 1156, 812, 3), name='content_input')
# img = np.array(load_image("he0.jpg", 1024), dtype=np.float32)
# border = np.ceil(np.shape(img)[0] / 20 / 4).astype(int) * 5
# container = [imresize(img, (np.shape(img)[0] + 2 * border, np.shape(img)[1] + 2 * border, 3))]
# container[0][border: np.shape(img)[0] + border, border: np.shape(img)[1] + border, :] = img
# container = np.repeat(container, 1, 0)
# print(container.shape)
preds = shortcut_interpolation(content_input / 255.,shortcut, 0.0)
print (preds)
preds_bgr = rgb2bgr(preds)
preds_pre = preprocess(preds_bgr)
print (preds_pre)
net = vgg.Vgg19()
net.build(preds_pre)
fv = net.net['conv1_1']
# interpolation_factor = tf.placeholder_with_default(0.0, shape=[], name="interpolation_factor")
# a=tf.cast(interpolation_factor, float32, name=None)
# print(a)
# shortcut = tf.placeholder_with_default([False, False], shape=[2], name="shortcut")
# print(shortcut)
# shortcut = tf.placeholder_with_default([False], shape=[1], name="shortcut")
# print(shortcut)
# content_width_1=tf.cast(content_width_1,dtype=tf.int8)
# if k_iter :
#     content_width_1, content_height_1=512,512
# else:
#     content_width_1, content_height_1 = 100,100
# content_input = tf.placeholder(tf.float32, shape=(1, 100, 120, 3), name='content_input')
# print (content_input.shape)
# k_iter = tf.placeholder_with_default([True], shape=[1], name="shortcut")
# with tf.Session() as sess:
#     content_width_1, content_height_1=tf.cond(k_iter[0],
#                                           # lambda:(tf.constant(tf.shape(content_input)[1]),tf.constant(tf.shape(content_input)[2])),
#                                           lambda:(tf.constant(100),tf.constant(100)),
#                                           lambda:(512,512)
#                                           )
#     content_width_1 = sess.run(content_width_1)
#     content_height_1 = sess.run(content_height_1)
# print(content_width_1)