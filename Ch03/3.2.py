'''
Tensorflow实现Softmax Regression识别手写数字
一、流程分成4个部分：
  1、定义算法公式，也就是神经网络forward时的计算
  2、定义loss，选定迭代器，并指定优化器优化loss
  3、迭代的对数据进行训练
  4、在测试集或者验证集上对准确率进行评测

二、我们定义的公式其实只是计算图，在执行代码是计算没有实际发生，只有调用run方法，并feed数据时计算才真正执行。
'''


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:/Users/zz/PycharmProjects/Action/Tensorflow/Ch03/mnist",one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w) + b)

y_ = tf.placeholder(tf.float32,[None,10])
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),axis=1))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))