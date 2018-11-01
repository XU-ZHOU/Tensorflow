'''
第5章 Tensorflow实现卷积神经网络
1、CNN最大的特点在于卷积的权值共享结构，可以大幅度减少神经网络的参数量，防止过拟合的同时又降低了神经网络模型的复杂度。
2、一般的卷积神经网络由多个卷积层构成，每个卷积层中通常会进行如下几个操作：
  （1）图像通过多个不同的卷积核滤波，并加偏置，提取出局部特征，每一个卷积核会映射出一个新的2D图像。
  （2）将前面卷积核的滤波输出结果进行非线性的激活函数处理。目前最常用的是ReLU。
  （3）对激活的结果再进行池化操作（即降采样，比如将2x2的图片降为1x1的图片），目前一般是使用最大池化，保留最显著的特征，
  并提升模型的畸变容忍能力。
3、卷积的好处是，不管图片尺寸如何，我们需要训练的权值数量只跟卷积核大小、卷积核数量有关，我们可以使用非常少的参数量处理
任意大小的图片。
4、卷积神经网络的要点：
 局部连接、权值共享、池化层中的降采样。
 其中，局部连接和权值共享降低了参数量，使训练复杂度大大下降，并减轻了过拟合。同时权值共享还赋予了卷积网络对平移的容忍性，
 而池化层降采样则进一步降低了输出参数量，并赋予模型对轻度形变的容忍性，提高了模型的泛化能力。
5、LeNet5的特性：
  （1）每个卷积层包含三个部分：卷积、池化和非线性激活函数。
  （2）使用卷积提取空间特性
  （3）降采样的平均池化层
  （4）双曲正切或S型的激活函数
  （5）MLP作为最后的分类器
  （6）层与层之间的稀疏连接减少计算复杂度
'''

#5.2 Tensorflow实现简单的卷积神经网络
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("C:/Users/zz/PycharmProjects/Action/Tensorflow/Ch03/mnist",one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = weight_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

