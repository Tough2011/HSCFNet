import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, Input, AveragePooling2D, Flatten, Dense
import keras
from keras.models import Model
from keras.regularizers import l2
import numpy as np
import tensorflow as tf
import time
import cifar10
import os
from sklearn.manifold import Isomap
import pandas as pd
from pandas import Series,DataFrame

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
batch_size = 128


def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


def compression(vector):
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


def conv2d(x, W, strides=1):
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet(input_shape, depth):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    outputs = Flatten()(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


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


# -------------------------------------------------------------------------------Feater
x = tf.placeholder(tf.float32, [None, 32, 32, 3])

resnet20 = resnet([32, 32, 3], 20)
features = resnet20(x)
#print(features)
dim = 64
# pred_feat = Dense(units=dim)(features)
# tf.nn.selu
pred = compression(features)
# ------------------------------------------------------------------------------

each_category = 5  # 每个类别用多少神经元刻画
class_num = 10
cluster_num = each_category * class_num  #   簇个数

NeuronKernel1 = Center_variable([cluster_num, dim], name='W_kernal1')#[簇个数, 样本维度 ]
NeuronKernel2 = Center_variable([cluster_num, dim], name='W_kernal2')
batchSize = 128
testbatchSize = 128
train_flag = tf.placeholder(tf.float32)
var, dist, line, neuron_layer = forward(pred, NeuronKernel1, NeuronKernel2, cluster_num, dim)

w2 = tf.Variable(tf.truncated_normal([cluster_num, class_num]), 'weight_w2')
b2 = bias_variable([class_num], 'biases_b2')
y = tf.matmul(neuron_layer, w2)+b2
y_input = tf.placeholder("float", [None, class_num])

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_input))

learning_rates = 1e-3
# global_step = tf.train.get_or_create_global_step()
# learning_rates = tf.train.piecewise_constant(
#     tf.Variable(0, trainable=False),
#     [10000, 30000],
#     [1e-3, 5e-4, 1e-4],
#     name='learning_rate')

# NeuronKernel1_grad = tf.gradients(l, NeuronKernel1)
optimizer = tf.train.AdamOptimizer(learning_rates).minimize(l)


def batch_augmentation(batch):
    batch = tf.image.random_flip_left_right(batch)
    batch = tf.image.random_flip_up_down(batch)
    batch = tf.pad(batch, tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]]))
    batch = tf.random_crop(batch, [batch_size, 32, 32, 3])
    return batch


train_data, train_labels, test_data, test_labels = cifar10.prepare_data()
train_data, test_data = cifar10.color_preprocessing(train_data, test_data)
input_queue_train = tf.train.slice_input_producer([tf.cast(train_data, tf.float32), tf.cast(train_labels, tf.float32)],
                                                  shuffle=True)
batch_train_x, batch_train_y = tf.train.shuffle_batch(input_queue_train, batch_size=batch_size,
                                                      capacity=100+3*batch_size, min_after_dequeue=1)
batch_train_x = batch_augmentation(batch_train_x)

# print(batch_train_x)
input_queue_test = tf.train.slice_input_producer([tf.cast(test_data, tf.float32), tf.cast(test_labels, tf.float32)],
                                                 shuffle=True)
batch_test_x, batch_test_y = tf.train.shuffle_batch(input_queue_test, batch_size=batch_size,
                                                    capacity=100+3*batch_size, min_after_dequeue=1)

color = ['red', 'green', 'blue', 'black', 'purple', 'orange', 'yellow', 'grey', 'brown', 'silver']
mark = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []
step = []
second = []
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print('start')
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    start0 = time.time()
    start = time.time()

    for i in range(500000):
        batch = sess.run([batch_train_x, batch_train_y])
        batch[0] = cifar10.data_augmentation(batch[0])
        if i % 10 != 0:
            sess.run([optimizer], feed_dict={x: batch[0], y_input: batch[1]})

        else:
            _, train_acc, lossv = \
                sess.run([optimizer, accuracy, l],
                                             feed_dict={x: batch[0], y_input: batch[1]})
            batch_val = sess.run([batch_test_x, batch_test_y])
            val_accuracy, vlossv = sess.run([accuracy, l],
                                            feed_dict={x: batch_val[0], y_input: batch_val[1]})
            if i % 1000 == 0:
                saver.save(sess, 'trained_models/cifar10_HSN_aug_%d/model' % each_category, global_step=i)

            # 绘制特征部分图像，若特征维度大于2，则降维
            # _y = np.argmax(batch[1], axis=-1)
            # _yv = np.argmax(batch_val[1], axis=-1)
            # if predv.shape[-1] > 2:
            #     embedding = Isomap(n_components=2)
            #     predv_transformed = embedding.fit_transform(predv)
            #     vpredv_transformed = embedding.fit_transform(vpredv)
            #     x1v_transformed = np.reshape(embedding.fit_transform(np.reshape(x1v, [class_num * each_category, -1])),
            #                                  [class_num, each_category, -1])
            #     x2v_transformed = np.reshape(embedding.fit_transform(np.reshape(x2v, [class_num * each_category, -1])),
            #                                  [class_num, each_category, -1])
            # else:
            #     predv_transformed = predv
            #     vpredv_transformed = vpredv
            #     x1v_transformed = x1v
            #     x2v_transformed = x2v
            # plt.switch_backend('agg')
            # fig = plt.figure()
            # for c in np.unique(_y, axis=0):
            #     # print(np.concatenate((x1v[c, :, 0], np.array([x2v[c, -1, 0]])), axis=0))
            #     # print(np.concatenate((x1v[c, :, 1], np.array([x2v[c, -1, 1]])), axis=0))
            #     plt.plot(np.concatenate((x1v_transformed[c, :, 0], np.array([x2v_transformed[c, -1, 0]])), axis=0),
            #              np.concatenate((x1v_transformed[c, :, 1], np.array([x2v_transformed[c, -1, 1]])), axis=0), c=color[c])
            #     plt.scatter(predv_transformed[_y == c, 0], predv_transformed[_y == c, 1],
            #                 alpha=0.8, c=color[c], s=30, label=c)
            #     plt.scatter(vpredv_transformed[_yv == c, 0], vpredv_transformed[_yv == c, 1],
            #                 alpha=0.8, c=color[c], marker='+', s=30, label=c)
            # plt.title('Hyper sausage')
            # plt.legend(loc=2)
            # plt.savefig('3d_features_%05d.png' % i)

            print('====================================================>>>>>')
            # print('var====>>' + str(varv))
            # print('dist====>>' + str(distv))
            # print('line====>>' + str(linev))
            # print('NeuronKernel1_grad====>>' + str(NeuronKernel1_gradv))
            print("step %d, train_accuracy %g" % (i, train_acc))
            print("step %d, val_accuracy %g" % (i, val_accuracy))
            print('tra_loss====>>' + str(lossv))
            print('val_loss====>>' + str(vlossv))
            train_accuracy.append(train_acc)
            test_accuracy.append(val_accuracy)
            train_loss.append(lossv)
            test_loss.append(vlossv)
            step.append(i)
            second.append(time.time() - start0)
            df = pd.DataFrame({'second': second,
                               'step': step,
                               'train_accuracy': train_accuracy,
                               'test_accuracy': test_accuracy,
                               'train_loss': train_loss,
                               'test_loss': test_loss})
            df.to_csv('log_hsn_aug_%d.csv' % each_category, index=False, sep=',')
            endtm = time.time() - start

            print('--------------------------------------------endtm    ' + str(endtm))
            start = time.time()


            # print('predv====>>\n' + str(predv))

            # print("step %d, train_accuracy %g" % (i, train_accuracy))
            # print("step %d, val_accuracy %g" % (i, val_accuracy))
            # print('tra_loss====>>' + str(lossv))
            # print('val_loss====>>' + str(vlossv))
            # # print('diffv====>>\n' + str(diffv))
            # # print('flagv====>>\n' + str(flagv))
            # # print('kernel_variablev====>>\n' + str(kernel_variablev))
            print('====================================================>>>>>')
    # 保存实验数据为csv
    df = pd.DataFrame({'second': second,
                       'step': step,
                       'train_accuracy': train_accuracy,
                       'test_accuracy': test_accuracy,
                       'train_loss': train_loss,
                       'test_loss': test_loss,
                       'variables_num': np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])})
    df.to_csv('log_cifar10_HSN_aug_%d.csv' % each_category, index=False, sep=',')
