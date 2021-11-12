# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.vision.image_classification.resnet import imagenet_preprocessing


ef compression(vector):
    epsilon = 1e-9
    vec_Compression_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_Compression_norm / (1 + vec_Compression_norm) / tf.sqrt(vec_Compression_norm + epsilon)
    vec_Compression = scalar_factor * vector
    return (vec_Compression)


def bias_variable(shape, namevalue):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=namevalue)


def weight_variable(shape, name):
    default_initializer = tf.contrib.layers.xavier_initializer()
    initial = tf.get_variable(name=name, shape=shape, initializer=default_initializer)
    return initial




layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
  return tf.keras.regularizers.L2(
      l2_weight_decay) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2c')(
          x)

  x = layers.add([x, input_tensor])
  x = layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = layers.Conv2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2a')(
          input_tensor)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2a')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2b')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2b')(
          x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '2c')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '2c')(
          x)

  shortcut = layers.Conv2D(
      filters3, (1, 1),
      strides=strides,
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name=conv_name_base + '1')(
          input_tensor)
  shortcut = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=bn_name_base + '1')(
          shortcut)

  x = layers.add([x, shortcut])
  x = layers.Activation('relu')(x)
  return x


def inner_product(data1, data2):  # 可同时计算多组向量的内积，保证最后一个维度是dim
    temp1 = tf.multiply(data1, data2)
    ans = tf.reduce_sum(temp1, axis=-1, keepdims=True)
    return ans


def comp_o(x0, x1, x2, epsilon=1e-9):
    # x0 = tf.Print(x0, [x0], message='[INFO]x==>', summarize=65)
    # x1 = tf.Print(x1, [x1], message='[INFO]x1==>', summarize=65)
    # x2 = tf.Print(x2, [x2], message='[INFO]x2==>', summarize=65)
    x1x2 = x2 - x1
    x1x0 = x0 - x1
    l = inner_product(x1x0, x1x2)
    x1x2_ = inner_product(x1x2, x1x2)
    l = tf.div(l, x1x2_ + epsilon)
    l = tf.clip_by_value(l, 0., 1.)
    # l = tf.Print(l, [tf.shape(l), l, tf.reduce_min(tf.squeeze(l, axis=-1), axis=-1), tf.reduce_max(tf.squeeze(l, axis=-1), axis=-1)], message='[INFO]lambda==>')
    o = tf.add(tf.multiply((1.-l), x1), tf.multiply(l, x2))
    # o = tf.Print(o, [tf.shape(o), o], message='[INFO]o==>', summarize=65)
    return o


def get_dist_line(x, x1, x2, epsilon=1e-9):
    o = comp_o(x, x1, x2)
    ox = x - o
    d = tf.sqrt(inner_product(ox, ox) + epsilon)
    d = tf.squeeze(d, axis=-1)
    # d = tf.Print(d, [tf.shape(d), d], message='[INFO]d==>')
    return d


def get_cos_dist_line(x, x1, x2, epsilon=1e-9):
    o = comp_o(x, x1, x2)
    d = inner_product(x, o)
    d = tf.divide(d, tf.multiply(inner_product(x, x), inner_product(o, o)) + epsilon)
    # ox = x - o
    # d = tf.sqrt(inner_product(ox, ox) + epsilon)
    d = tf.squeeze(d, axis=-1)
    # d = tf.Print(d, [tf.shape(d), d], message='[INFO]d==>')
    return d


def forward(data, x1, x2, cluster_num, dim):
    epsilon = 1e-9
    var = tf.abs(bias_variable([1, cluster_num], 'radius'))
    data_expand = tf.expand_dims(data, axis=1)
    x1_expand = tf.expand_dims(x1, axis=0)
    x2_expand = tf.expand_dims(x2, axis=0)
    d = get_cos_dist_line(data_expand, x1_expand, x2_expand)
    matrix = tf.exp(-tf.pow(tf.multiply(1 / (d + epsilon) - 1, 1 / (tf.sqrt(tf.cast(dim, tf.float32)) * (var + epsilon))), 2))
    # matrix = tf.Print(matrix, [tf.shape(matrix), matrix], message='[INFO]matrix==>')
    # matrix = tf.reshape(matrix, [-1, cluster_num])
    return tf.reshape(var, [-1, cluster_num]), tf.reshape(d, [-1, cluster_num]), \
           tf.reshape(tf.sqrt(inner_product(x1 - x2, x1 - x2)), [-1, cluster_num]), matrix


def Center_variable(shape, name):
    default_initializer = tf.contrib.layers.xavier_initializer()
    initial = tf.get_variable(name=name, shape=shape, initializer=default_initializer)
    return initial



def resnet50(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             rescale_inputs=False,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):
  """Instantiates the ResNet50 architecture.

  Args:
    num_classes: `int` number of classes for image classification.
    batch_size: Size of the batches for each step.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
      A Keras model instance.
  """
  input_shape = (224, 224, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  if rescale_inputs:
    # Hub image modules expect inputs in the range [0, 1]. This rescales these
    # inputs to the range expected by the trained model.
    x = layers.Lambda(
        lambda x: x * 255.0 - tf.keras.backend.constant(    # pylint: disable=g-long-lambda
            imagenet_preprocessing.CHANNEL_MEANS,
            shape=[1, 1, 3],
            dtype=x.dtype),
        name='rescale')(
            img_input)
  else:
    x = img_input

  if tf.keras.backend.image_data_format() == 'channels_first':
    x = layers.Permute((3, 1, 2))(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  block_config = dict(
      use_l2_regularizer=use_l2_regularizer,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon)
  x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
  x = layers.Conv2D(
      64, (7, 7),
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv1')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = conv_block(
      x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), **block_config)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', **block_config)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', **block_config)

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', **block_config)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', **block_config)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', **block_config)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', **block_config)

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', **block_config)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', **block_config)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', **block_config)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', **block_config)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', **block_config)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', **block_config)

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', **block_config)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', **block_config)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', **block_config)

  x = layers.GlobalAveragePooling2D()(x)

  dim = 2048

  pred = compression(features)
  # ------------------------------------------------------------------------------

  each_category = 100  # 每个类别用多少神经元刻画
  class_num = num_classes
  cluster_num = each_category * class_num  #   簇个数

  NeuronKernel1 = Center_variable([cluster_num, dim], name='W_kernal1')#[簇个数, 样本维度 ]
  NeuronKernel2 = Center_variable([cluster_num, dim], name='W_kernal2')


  var, dist, line, neuron_layer = forward(pred, NeuronKernel1, NeuronKernel2, cluster_num, dim)

  w2 = tf.Variable(tf.truncated_normal([cluster_num, class_num]), 'weight_w2')
  b2 = bias_variable([class_num], 'biases_b2')
  x = tf.matmul(neuron_layer, w2)+b2


  # x = layers.Dense(
  #     num_classes,
  #     kernel_initializer=tf.initializers.random_normal(stddev=0.01),
  #     kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
  #     bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
  #     name='fc1000')(
  #         x)


  # A softmax that is followed by the model loss must be done cannot be done
  # in float16 due to numeric issues. So we pass dtype=float32.
  x = layers.Activation('softmax', dtype='float32')(x)

  # Create model.
  return tf.keras.Model(img_input, x, name='resnet50')



