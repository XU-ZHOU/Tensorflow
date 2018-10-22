'''
第3章 Tensorflow入门
  本章的前3个小节中，将分别介绍Tensorflow的计算模型、
  数据模型和运行模型，最后一张将介绍神经网络的主要计算流程。
'''

'''
3.1 Tensorflow计算模型-计算图
  1、计算图是Tensorflow中最基本的一个概念，Tensorflow中的所有计算都会转化为计算图上
  的一个节点。
  2、Tensor是张量，在Tensorflow 中可以被理解为多维数组。Flow是流，表达了张量之间通过
  计算相互转化的过程。
  3、Tensorflow是一个通过计算图的形式来表达计算的编程系统，Tensorflow中的每一个计算
  都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。
  4、Tensorflow程序一般分为两个阶段，第一个是定义计算图中所有的计算，第二个是定义一个
  计算得到他们的和。
  5、
'''

'''
3.1.2 计算图的使用
在Tensorflow中会自动维护一个默认的计算图，通过tf.get_default_graph函数可以获取
当前默认的计算图。
'''
import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = a + b
print(a.graph is tf.get_default_graph())

#Tensorflow支持通过tf.graph函数来生成新的计算图，不同计算图上的张量和运算都不会共享
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v",initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v",initializer=tf.ones_initializer(shape=[1]))

with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))


'''
3.2 Tensorflow数据模型-张量
  1、张量是Tensorflow管理数据的形式
  2、在Tensorflow中所有的数据都是通过张量的形式来表现的。
  3、张量可以被简单的理解为多维数组，其中零阶张量表示标量，也就是一个数；
  第一阶张量为向量，也就是一个一维数组；第n阶张量可以理解为一个n维数组。
  4、张量在Tensorflow中并不直接采用数组的形式，知识对运算结果的引用。
  5、张量中主要保存三个属性：名字、维度、类型。
  6、张量使用主要总结为两大类：第一类用途是对中间计算结果的引用；第二类是
  当计算图构造完成之后，张量可以用来获得计算结果，也就是获得真是的数字。
'''

#使用张量记录中间结果
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = a + b

#直接计算向量的和，这样可读性会比较差
result = tf.constant([1.0,2.0],name="a") + tf.constant([2.0,3.0],name="b")




'''
3.3 Tensorflow运行模型-会话
  1、会话（session）执行定义好的运算，有用并管理Tensorflow程序运行时的所有资源。
  2、计算完成以后需要关闭会话帮助回收资源，否则会发生资源泄露问题。
  3、Tensoflow使用会话的模式有两种。
'''

#第一种模式需要明确调用会话生成函数和关闭会话函数。
#然而当程序异常退出时，关闭会话的函数可能就不会被执行从而导致资源泄露。
sess = tf.Session()
sess.run()
sess.close()

#第二种，通过Python上下文管理器使用会话
#不需要调用sess.close()关闭会话，当上下文退出时会话关闭并且资源自动释放
with tf.Session() as sess:
    sess.run()





'''
3.4 Tensorflow实现神经网络
  使用神经网络解决分类问题主要分为以下4个步骤：
  1.提取问题中实体的特征向量作为神经网络的输入。
  2.定义神经网络的结构，并定义如何从神经网络的输入
  得到输出。这个过程就是神经网络的前向传播算法。
  3.通过训练数据来调整神经网络中参数的取值。
  4.使用训练好的神经网络来预测未知的数据。
  5.神经网络的结构指的是不同神经元之间的连接结构。
  6.Tensorflow提供了placeholder机制用于提供输入数据，placeholder相当于
  定义了一个位置，这个位置中的数据在程序运行时再指定。placeholder类型不可改变。
'''

#通过placeholder实现前向传播算法

import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1))

x = tf.placeholder(tf.float32,shape=(1,2),name="input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))



'''
3.4.5 完整神经网络样例程序
  神经网络过后才能分为3步：
  1、定义神经网络的结构和前向传播的输出结果
  2、定义损失函数以及选择反向传播优化的算法
  3、生成会话并在训练数据上反复运行反向传播优化算法。
'''
import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g" %
                  (i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
