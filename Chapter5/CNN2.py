#Tensorflow实现进阶的卷积神经网络
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = 'H:/data/cifar-10-batches'

def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var


