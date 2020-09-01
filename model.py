# -*- coding: utf-8 -*-
from __future__ import division
import os
from os import listdir
from os.path import isfile, join
import time
import tensorflow as tf
from tensorflow import keras

import numpy as np
import sys
from scipy.misc import imresize
import functools

import vgg19.vgg as vgg
from utils import *
import netdef

from PIL import Image
from functools import partial
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import math
from functools import reduce
from operator import mul

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

STROKE_SHORTCUT_DICT = {"480": [False, False], "interp": [True, True]}
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
STYLE_LAYERS_2 = ('conv1_1_1', 'conv2_1_1', 'conv3_1_1', 'conv4_1_1', 'conv5_1_1')
CONTENT_LAYERS = ('conv4_2')

DEFAULT_RESOLUTIONS = (480, 480)


class DataLoader(object):
    def __init__(self, args):
        file_names = [join(args.train_path, f) for f in listdir(args.train_path) if
                      isfile(join(args.train_path, f)) and ".jpg" in f]
        self.mscoco_fnames = file_names
        self.train_size = len(file_names)
        self.batch_size = args.batch_size
        self.epochs = 0
        self.nbatches = int(self.train_size / args.batch_size)
        self.batch_idx = 0
        self.perm = np.random.permutation(self.train_size)

        print('[*] Training dataset size: {}'.format(self.train_size))
        print('[*] Batch size: {}'.format(self.batch_size))
        print('[*] {} #Batches per epoch'.format(self.nbatches))

    def fill_feed_dict(self, content_pl, img_size=None):
        content_images = np.zeros((self.batch_size,) + img_size, dtype=np.float32)
        for i in xrange(self.batch_size):
            img = np.array(
                load_image(self.mscoco_fnames[self.perm[self.batch_idx * self.batch_size + i]], shape=img_size),
                dtype=np.float32)
            content_images[i] = img

        self.batch_idx += 1
        if self.batch_idx == self.nbatches:
            self.batch_idx = 0
            self.epochs += 1
            self.perm = np.random.permutation(self.train_size)
        return {content_pl: content_images}


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self._build_model(args)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.data_loader = DataLoader(args)

    def _build_model(self, args):
        # ***************************************************************
        VGG_MEAN = [103.939, 116.779, 123.68]

        def gram_matrix(activations):
            height = tf.shape(activations)[1]
            width = tf.shape(activations)[2]
            num_channels = tf.shape(activations)[3]
            gram_matrix = tf.transpose(activations, [0, 3, 1, 2])
            gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
            gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
            return gram_matrix

        def rgb2bgr(rgb, vgg_mean=True):
            if vgg_mean:
                return rgb[:, :, ::-1] - VGG_MEAN
            else:
                return rgb[:, :, ::-1]

        def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
            color_codes = ['BLACK', 'WHITE', 'BLUE']

            def _extract_mask(seg, color_str):
                h, w, c = np.shape(seg)
                if color_str == "BLACK":
                    mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
                    mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
                    mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
                elif color_str == "WHITE":
                    mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
                    mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
                    mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
                elif color_str == "BLUE":
                    mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
                    mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
                    mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
                return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

            # PIL resize has different order of np.shape
            content_seg = np.array(
                Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR),
                dtype=np.float32) / 255.0
            style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR),
                                 dtype=np.float32) / 255.0

            color_content_masks = []
            color_style_masks = []
            for i in xrange(len(color_codes)):
                color_content_masks.append(
                    tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i])), 0), -1))
                color_style_masks.append(
                    tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0), -1))
            return color_content_masks, color_style_masks

        # *****************************************************
        # center-crop loading style image
        # change this the following two lines to load original style image
        style_highres_img = load_image(args.style_path, shape=DEFAULT_RESOLUTIONS)
        self.style_targets = [
            np.array(style_highres_img.resize(DEFAULT_RESOLUTIONS, resample=Image.BILINEAR), dtype=np.float32)]

        self.content_input = tf.placeholder(tf.float32, shape=(args.batch_size, None, None, 3), name='content_input')
        self.shortcut = tf.placeholder_with_default([False, False], shape=[2], name="shortcut")
        self.interpolation_factor = tf.placeholder_with_default(0.0, shape=[], name="interpolation_factor")
        self.k_iter = tf.placeholder_with_default([False], shape=[1], name="k_iter")
        self.style_features_pyramid = []
        with tf.name_scope("pre-style-features"), tf.Session() as sess:
            style_image = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='precompute_style')
            style_image_pre = vgg.preprocess(vgg.rgb2bgr(style_image))
            net = vgg.Vgg19()
            net.build(style_image_pre)

            for style_target in self.style_targets:
                style_target = np.expand_dims(style_target, 0)
                content_image_1 = np.array(Image.open(args.content_path).convert("RGB"), dtype=np.float32)
                content_width_1, content_height_1 = content_image_1.shape[1], content_image_1.shape[0]
                style_width_1, style_height_1 = style_target.shape[2], style_target.shape[1]

                content_masks, style_masks = load_seg(args.content_seg_path, args.style_seg_path,
                                                      [content_width_1, content_height_1],
                                                      [style_width_1, style_height_1])
                content_segs = content_masks
                style_segs = style_masks
                _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
                _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
                layer_index = 0
                style_features = {}

                for all_layer in [layer.name for layer in net.get_all_layers()]:
                    all_layer = all_layer[all_layer.find("/") + 1:]
                    if "pool" in all_layer:
                        content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(
                            math.ceil(content_seg_height / 2))
                        style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(
                            math.ceil(style_seg_height / 2))

                        for i in xrange(len(content_segs)):
                            content_segs[i] = tf.image.resize_bilinear(content_segs[i], tf.constant(
                                (content_seg_height, content_seg_width)))
                            style_segs[i] = tf.image.resize_bilinear(style_segs[i],
                                                                     tf.constant((style_seg_height, style_seg_width)))
                    elif "conv" in all_layer:
                        for i in xrange(len(content_segs)):
                            content_segs[i] = tf.nn.avg_pool(
                                tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                            style_segs[i] = tf.nn.avg_pool(
                                tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')

                    all_layer = all_layer[:all_layer.find("/")]
                    style_fvs = sess.run([net.conv1_1, net.conv2_1, net.conv3_1, net.conv4_1, net.conv5_1],
                                         feed_dict={style_image: style_target})
                    fv_const = [tf.constant(fv) for fv in style_fvs]

                    if all_layer == STYLE_LAYERS[layer_index]:
                        print('Setting up style layer (style_gram): <{}>'.format(all_layer))
                        seg_feature = {}
                        const_layer = fv_const[layer_index]
                        layer_index = layer_index + 1
                        seg_index = 0

                        for content_seg, style_seg in zip(content_segs, style_segs):
                            style_mask_mean = tf.reduce_mean(style_seg)
                            gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
                            gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                                        lambda: gram_matrix_const / (tf.to_float(
                                                            tf.size(const_layer)) * style_mask_mean),
                                                        lambda: gram_matrix_const
                                                        )
                            seg_feature[seg_index] = gram_matrix_const
                            seg_index = seg_index + 1
                        style_features[all_layer] = seg_feature

                    if not style_features:
                        self.style_features_pyramid = self.style_features_pyramid
                    else:
                        self.style_features_pyramid.append(style_features)

        # Content Loss and Style Loss
        content_bgr = vgg.rgb2bgr(self.content_input)
        content_pre = vgg.preprocess(content_bgr)
        content_net = vgg.Vgg19()
        content_net.build(content_pre)
        content_fv = content_net.net[CONTENT_LAYERS]

        self.preds = netdef.shortcut_interpolation(self.content_input / 255., self.shortcut, self.interpolation_factor)
        preds_bgr = vgg.rgb2bgr(self.preds)
        preds_pre = vgg.preprocess(preds_bgr)
        net = vgg.Vgg19()
        net.build(preds_pre)
        preds_content_fv = net.net[CONTENT_LAYERS]

        self.content_loss = args.content_weight * (2 * tf.nn.l2_loss(
            preds_content_fv - content_fv) / (tf.to_float(tf.size(content_fv)) * args.batch_size))
        self.style_loss_layer = []

        content_width_1, content_height_1 = 480, 480
        style_image_1 = rgb2bgr(np.array(Image.open(args.style_path).convert("RGB"), dtype=np.float32))
        style_width_1, style_height_1 = style_image_1.shape[1], style_image_1.shape[0]
        content_masks, style_masks = load_seg(args.content_seg_path, args.style_seg_path,
                                              [content_width_1, content_height_1], [style_width_1, style_height_1])
        content_segs = content_masks
        style_segs = style_masks
        _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
        _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
        layer_index = 0

        for all_layer in [layer.name for layer in net.get_all_layers()]:
            if "pool" in all_layer:
                content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(
                    math.ceil(content_seg_height / 2))
                style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(
                    math.ceil(style_seg_height / 2))
                for i in xrange(len(content_segs)):
                    content_segs[i] = tf.image.resize_bilinear(content_segs[i],
                                                               tf.constant((content_seg_height, content_seg_width)))
                    style_segs[i] = tf.image.resize_bilinear(style_segs[i],
                                                             tf.constant((style_seg_height, style_seg_width)))
            elif "conv" in all_layer:
                for i in xrange(len(content_segs)):
                    content_segs[i] = tf.nn.avg_pool(
                        tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                        ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                    style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"),
                                                   ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')

            all_layer = all_layer[:all_layer.find("/")]
            if all_layer == STYLE_LAYERS_2[layer_index]:
                print('Setting up style layer (gram): <{}>'.format(all_layer))
                style_layer = all_layer[:-2]
                fv = net.net[style_layer]
                var_layer = fv
                layer_index = layer_index + 1
                layer_style_loss = 0.0
                seg_index_ = 0

                for content_seg, style_seg in zip(content_segs, style_segs):
                    content_mask_mean = tf.reduce_mean(content_seg)
                    gram = gram_matrix(tf.multiply(var_layer, content_seg))
                    gram = tf.cond(tf.greater(content_mask_mean, 0.),
                                   lambda: gram / (tf.to_float(
                                       tf.size(var_layer)) * content_mask_mean),
                                   lambda: gram
                                   )
                    style_gram = tf.to_float(tf.cond(self.shortcut[0],
                                                     lambda: self.style_features_pyramid[2][style_layer][seg_index_],
                                                     lambda: tf.cond(self.shortcut[1],
                                                                     lambda:
                                                                     self.style_features_pyramid[1][style_layer][
                                                                         seg_index_],
                                                                     lambda:
                                                                     self.style_features_pyramid[0][style_layer][
                                                                         seg_index_]
                                                                     )
                                                     ))
                    seg_index_ = seg_index_ + 1
                    diff_seg_sum = args.style_weight * (2 * tf.nn.l2_loss(gram - style_gram) / tf.to_float(
                        tf.size(style_gram))) / args.batch_size
                    layer_style_loss += diff_seg_sum
                self.style_loss_layer.append(layer_style_loss)

        self.style_loss = 0.0
        for loss in self.style_loss_layer:
            self.style_loss += loss

        # Total Variational Loss
        tv_y_size = tf.to_float(tf.size(self.preds[:, 1:, :, :]))
        tv_x_size = tf.to_float(tf.size(self.preds[:, :, 1:, :]))
        y_tv = tf.nn.l2_loss(self.preds[:, 1:, :, :] - self.preds[:, :-1, :, :])
        x_tv = tf.nn.l2_loss(self.preds[:, :, 1:, :] - self.preds[:, :, :-1, :])
        self.tv_loss = 2 * args.tv_weight * (x_tv / tv_x_size + y_tv / tv_y_size) / args.batch_size
        self.loss = tf.add_n([self.content_loss, self.style_loss, self.tv_loss], name="loss")

        # def get_num_params():
        #     num_params = 0
        #     for variable in tf.global_variables():
        #         shape = variable.get_shape()
        #         num_params += reduce(mul, [dim.value for dim in shape], 1)
        #     return num_params
        #
        # num_params = get_num_params()
        # print num_params

    def train(self, args):
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        for iter_count in xrange(1, int(args.max_iter) + 1):
            feed_dict = self.data_loader.fill_feed_dict(
                self.content_input,
                img_size=DEFAULT_RESOLUTIONS + (3,)
            )
            feed_dict[self.shortcut] = [iter_count % 3 == 2, iter_count % 3 == 1]
            feed_dict[self.interpolation_factor] = 0.0
            feed_dict[self.k_iter] = [False]

            # start training
            _, content_loss, tv_loss, total_loss, style_losses_list = self.sess.run([
                self.optimizer,
                self.content_loss,
                self.tv_loss,
                self.loss,
                self.style_loss_layer,
            ], feed_dict=feed_dict)

            # per iterations display all type of losses (iter_print)
            if iter_count % args.iter_print == 0 and iter_count != 0:
                print('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, args.max_iter, content_loss))
                for idx, sloss in enumerate(style_losses_list):
                    print ('\tStyle {} loss: {}'.format(idx, sloss))
                print('\tTV loss: {}'.format(tv_loss))
                print('\tTotal loss: {}'.format(total_loss))

            # per iterations save stylized image and model (checkpoint_iter/cp_iter)
            if iter_count % args.cp_iter == 0 and iter_count != 0:
                self.save_sample_train(args, join(args.serial, "out_{}_480px.jpg".format(iter_count)),
                                       shortcut=STROKE_SHORTCUT_DICT["480"])
                self.save(args.checkpoint_dir, iter_count)
            # ...

    def finetune_model(self, args):
        self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        start_step = 0
        if self.load(args.checkpoint_dir):
            print '[*] Success load the checkpoint {}, continue to train.'.format(args.checkpoint_dir)
            checkpoint_names = [f for f in os.listdir(args.checkpoint_dir) if ".meta" in f]
            checkpoint_nums = [int(''.join(x for x in r if x.isdigit())) for r in checkpoint_names]
            start_step = max(checkpoint_nums) + 1
        else:
            print '[!] Error in loading checkpoint'
            return

        for iter_count in xrange(start_step, int(args.max_iter) + start_step + 1):
            feed_dict = self.data_loader.fill_feed_dict(
                self.content_input,
                img_size=DEFAULT_RESOLUTIONS + (3,)
            )

            feed_dict[self.shortcut] = [iter_count % 3 == 2, iter_count % 3 == 1]
            feed_dict[self.interpolation_factor] = 0.0

            _, content_loss, tv_loss, total_loss, style_losses_list = self.sess.run([
                self.optimizer,
                self.content_loss,
                self.tv_loss,
                self.loss,
                self.style_loss_layer
            ], feed_dict=feed_dict)

            if iter_count % args.iter_print == 0 and iter_count != 0:
                print('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, int(args.max_iter) + start_step,
                                                                     content_loss))
                for idx, sloss in enumerate(style_losses_list):
                    print('\tStyle {} loss: {}'.format(idx, sloss))
                print('\tTV loss: {}'.format(tv_loss))
                print('\tTotal loss: {}'.format(total_loss))

            if iter_count % args.cp_iter == 0 and iter_count != 0:
                self.save(args.checkpoint_dir, iter_count)
                self.save_sample_train(args, join(args.serial, "out_{}_480px.jpg".format(iter_count)),
                                       shortcut=STROKE_SHORTCUT_DICT["480"])

    def load(self, checkpoint_dir):
        print(' [*] Reading checkpoint...')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            try:
                self.saver.restore(self.sess, checkpoint_dir)
                return True
            except:
                return False

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model'), global_step=step)

    def save_sample_train(self, args, output_path, shortcut):
        img = np.array(load_image(args.content_path, 1024), dtype=np.float32)
        border = np.ceil(np.shape(img)[0] / 20 / 4).astype(int) * 5
        container = [imresize(img, (np.shape(img)[0] + 2 * border, np.shape(img)[1] + 2 * border, 3))]
        container[0][border: np.shape(img)[0] + border, border: np.shape(img)[1] + border, :] = img
        container = np.repeat(container, args.batch_size, 0)
        preds = self.sess.run(self.preds, feed_dict={self.content_input: container, self.shortcut: shortcut,
                                                     self.interpolation_factor: 0.0, self.k_iter: [True]})
        print '******************************************************'
        save_image(output_path,
                   np.squeeze(preds[0][border: np.shape(img)[0] + border, border: np.shape(img)[1] + border, :]))
        print("[*] Save to {}".format(output_path))
