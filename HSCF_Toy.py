import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import csv
from sklearn.manifold import Isomap
import pandas as pd
from pandas import Series,DataFrame
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

batch_size = 64

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


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inner_product(data1, data2):  # 可同时计算多组向量的内积，保证最后一个维度是dim
    temp1 = tf.multiply(data1, data2)
    ans = tf.reduce_sum(temp1, axis=-1, keepdims=True)
    return ans


def comp_o(x0, x1, x2, epsilon=1e-9):

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
 
    return d


def forward(data, x1, x2, cluster_num):
    epsilon = 1e-9
    var = tf.abs(bias_variable([1, cluster_num], 'radius'))
    data_expand = tf.expand_dims(data, axis=1)
    x1_expand = tf.expand_dims(x1, axis=0)
    x2_expand = tf.expand_dims(x2, axis=0)
    d = get_dist_line(data_expand, x1_expand, x2_expand)
    matrix = tf.exp(-tf.pow(tf.multiply(d, 1 / (tf.sqrt(2.0) * (var + epsilon))), 2))
    # matrix = tf.Print(matrix, [tf.shape(matrix), matrix], message='[INFO]matrix==>')
    # matrix = tf.reshape(matrix, [-1, cluster_num])
    return tf.reshape(var, [-1, cluster_num]), tf.reshape(d, [-1, cluster_num]), \
           tf.reshape(tf.sqrt(inner_product(x1 - x2, x1 - x2)), [-1, cluster_num]), matrix


def Center_variable(shape, name):
    default_initializer = tf.contrib.layers.xavier_initializer()
    initial = tf.get_variable(name=name, shape=shape, initializer=default_initializer)
    return initial

# -------------------------------------------------------------------------------Feater



data = pd.read_csv('data.csv')
data_shuf =  shuffle (data)

data = data_shuf.iloc[:,0:2]
label = data_shuf.iloc[:,2:3]
print(data.shape, label.shape)
label = to_categorical(label)

x = tf.placeholder(tf.float32, [None, 2])

dim = 100
pred = tf.layers.dense(x, units=dim, activation=None)
# tf.nn.selu
pred = compression(pred)
# ------------------------------------------------------------------------------

each_category = 30   # 每个类别用多少神经元刻画
class_num = 3
cluster_num = each_category * class_num  #   簇个数

NeuronKernel1 = Center_variable([cluster_num, dim], name='W_kernal1')#[簇个数, 样本维度 ]
NeuronKernel2 = Center_variable([cluster_num, dim], name='W_kernal2')

train_flag = tf.placeholder(tf.float32)
var, dist, line, neuron_layer = forward(pred, NeuronKernel1, NeuronKernel2, cluster_num)

w2 = tf.Variable(tf.truncated_normal([cluster_num, class_num]), 'weight_w2')
b2 = bias_variable([class_num], 'biases_b2')
y = tf.matmul(neuron_layer, w2)+b2
y_input = tf.placeholder("float", [None, class_num])

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_input))

learning_rates = 1e-3

dataset_size = data.shape[0]
NeuronKernel1_grad = tf.gradients(l, NeuronKernel1)
optimizer = tf.train.AdamOptimizer(learning_rates).minimize(l)
gpu_options = tf.GPUOptions(allow_growth=True)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print('start')
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    start0 = time.time()
    start = time.time()

    for i in range(10000):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        #batch = mnist.train.next_batch(batch_size)
        if i % 10 != 0:
            sess.run([optimizer], feed_dict={x: data.iloc[start:end, :].values, y_input: label[start:end]})
        else:
            _, train_acc, lossv = \
                sess.run([optimizer, accuracy, l],
                        feed_dict={x: data.values, y_input: label})
            
           
            print("step %d, train_accuracy %g" % (i, train_acc))
          
            print('tra_loss====>>' + str(lossv))
            
            endtm = time.time() - start

            print('--------------------------------------------endtm    ' + str(endtm))
            start = time.time()

