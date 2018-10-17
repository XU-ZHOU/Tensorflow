#神经网络进一步优化

import tensorflow as tf
'''
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
learning_step = tf.train.GradientDescentOptimizer()
'''


#w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
#y = tf.matmul(x,w)
#loss = tf.reduce_mean(tf.sequare(y_-y))+tf.contrib.layers.12_regularizer(lambda)(w)

weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))

import tensorflow as tf

def get_weight(shape,lamb):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(lamb)(var))
    return var

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
batch_size = 8

layer_dimension = [2,10,10,10,1]
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1,n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))
tf.add_to_collection('losses',mse_loss)
loss = tf.add_n(tf.get_collection('losses'))


#滑动平均模型

import tensorflow as tf
v1 = tf.Variable(0,dtype=tf.float32)
step = tf.Variable(0,trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99,step)
maintain_average_op = ema.apply([v1])
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1,ema.average(v1)]))
    sess.run(tf.assign(v1,5))
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assige(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))
