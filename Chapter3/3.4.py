#Tensorflow实现神经网络
'''
训练神经网络的三个步骤：
1.定义神经网络的结构和前向传播的输出结果。
2.定义损失函数以及选择反向传播优化的算法。
3.生成会话，并且在训练数据上发怒放运行反向传播优化算法。
'''


import tensorflow as tf
weights = tf.Variable(tf.random_normal([2,3],stddev=2))

biases = tf.Variable(tf.zeros([3]))
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value()*0.2)

import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.constant([[0.7,0.9]])

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)
#sess.run(w1.initializer)
#sess.run(w2.initializer)
print(sess.run(y))
sess.close()


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


#3.4.5完整神经网络样例程序
import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_= tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 120
X = rdm.rand(dataset_size,2)

Y = [[int(x1+x2<1)] for (x1,x2) in X]

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
            print("After %d training step(s),crpss entropy on all data is %g"%(i,total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))


