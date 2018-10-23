'''
第4章 深层神经网络
  4.1 深度学习与深层神经网络
  1、深度学习两个重要特性是多层和非线性。
  2、分类问题中，交叉熵作为损失函数，衡量的是两个概率分布之间的距离。
  3、回归问题中，最常用的是损失函数是均方误差MSE。均方误差在分类中也经常使用。

'''

import tensorflow as tf
#tf.greater函数和tf.select函数的用法
v1 = tf.constant([1.0,2.0,3.0,4.0])
v2 = tf.constant([4.0,3.0,2.0,1.0])
sess = tf.InteractiveSession()
print(tf.greater(v1,v2).eval())
print(tf.where(tf.greater(v1,v2),v1,v2).eval())

#简单的神经网络讲解损失函数对模型训练结果的影响

import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),
                              (y - y_)*loss_more,
                              (y_ - y)*loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,
                 feed_dict={x:X[start:end],y_:Y[start:end]})
        print(sess.run(w1))

'''
4.3 神经网络优化算法
  1、梯度下降法主要用于单个参数的取值。
  2、反向传播算法在所有参数上使用梯度下降算法。
  3、梯度下降法存在的问题：1、参数的初始值很大程度上影响最后的结果，
     只有放损失函数是凸函数时，梯度下降法才能保证达到全局最优解；
     2、计算时间太长，因为要在全部训练数据上最小化损失。
  4、为了加快训练速度，随机梯度下降法，每一轮迭代中，随机优化某一条
    训练数据上的损失函数。 SGD的问题是，可能无法达到局部最优。
'''

'''
Tensorflow实现神经网络过程
batch_size = n
x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')
loss =
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    for i in range(STEPS):
        current_X,current_Y=
        sess.run(train_step,feed_dict={x:current_X,y_:current_Y})
'''

'''
4.4 神经网络进一步优化
  1、通过指数衰减方法设置梯度下降算法中的学习率，既可以在模型训练前期快速
  接近较优解，又可以保证模型在训练后期不会有太大波动。
  2、过拟合问题。
  3、滑动平均模型。滑动平均模型将每一轮得到的模型综合起来，从而得到最终的
  模型更加健壮。
'''

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    0.1,global_step,100,0.96,staircase=True)

'''
4.4.2 过拟合问题
  1、过拟合指的是当一个模型过于复杂之后，它可以很好的记忆每一个训练数据中
     随机噪音的部分而忘记要去学习训练数据中通用的趋势。
  2、避免过拟合问题，正则化，正则化的思想是在损失函数中加入刻画模型复杂程度的指标。
  3、L1正则化会让参数变得更加稀疏，即让更多参数变为0，打到类似特征选取的功能；
     L2正则化会让参数变得很小。
     L1正则化的计算公式不可导，L2正则化公式可导。
'''
#带L2正则化的损失函数定义
w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w)
#loss = tf.reduce_mean(tf.square(y_-y)) + tf.contrib.layers.l2_regularizer(lambda)(w)


weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))


#通过集合计算5层神经网络带L2正则化的损失函数的计算方式
import tensorflow as tf
'''
def get_weight(shape,lambda):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda)(var))
    return var
'''

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
batch_size = 8
layer_dimension = [2,10,10,10,1]
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(i,n_layers):
    out_dimension = layer_dimension[i]
    #weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses',mse_loss)
loss = tf.add_n(tf.get_collection('losses'))

#滑动平均模型
import tensorflow as tf
v1 = tf.Variable(0,dtype=tf.float32)
step = tf.Variable(0,trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99,step)
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run([v1,ema.average(v1)]))
    sess.run(tf.assign(v1,5))
    print(sess.run([v1,ema.average(v1)]))
    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

