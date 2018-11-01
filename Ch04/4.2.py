'''
第四章 Tensorflow实现自编码器及多层感知机
1、自编码器
  自编码器，即可以使用自身的高阶特征编码自己，是无监督的，自编码器其实也是一种神经网络，他的输入和输出是一致的，
  它借助稀疏编码的思想，目标是使用稀疏的一些高阶特征重新组合重构自己。特点是：第一，希望输入/输出一致；第二：希望
  使用高阶特征来重构自己。
2、自编码器希望使用少量的高阶特征来重构输入，所以可以加入几种限制：
  （1）如果限制中间隐含层节点的数量，比如让中间隐含层节点的数量小于输入/输出节点的数量，就相当于一个降维的过程。
  （2）如果给数据加入噪声，那就是去燥自编码器，我们将从噪声中学习出数据的特征。唯有学习数据频繁出现的模式和结构，
  将无规律的噪声略去，才可以复原数据。
3、自编码器作为一种无监督学习方法，与其他无监督方法的主要不同之处是，它不是对数据进行聚类，而是提取其中最有用、最
频繁出现的高阶特征，根据这些高阶特征重构数据。
'''

#Tensorflow实现自编码器
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data


'''
参数初始化方法xavier initialization。
为什么需要参数初始化？
1、如果深度学习模型的初始权重初始化的太小，那么信号将在每层间传递时逐渐缩小而难以产生作用。
2、如果权重初始化的太大，那信号将在每层间传递时逐渐放大并导致发散和失效。
xevier初始化器的作用就是让权重被初始化的不大不小，正好合适。
'''
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                             minval= low,maxval= high,
                             dtype= tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.traning_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        #定义网络结构
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        #定义损失函数，使用平方误差作为cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))  #tf.subtract是减法
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    #用一个batch数据进行训练兵返回当前的损失
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict = {self.x:X,self.scale:self.traning_scale})
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict= {self.x:X,self.scale:self.traning_scale})

    #定义transform函数，返回自编码器隐含层的输出结果
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.traning_scale})

    #定义generate函数，将隐含层的输出作为输入，将高阶特征复原为原始数据的步骤
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    #定义reconstruct函数，通过高阶特征复原数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.traning_scale})

    #getWeights函数是获取隐含层权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    #getBiases函数获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

mnist = input_data.read_data_sets("C:/Users/zz/PycharmProjects/Action/Tensorflow/Ch03/mnist",one_hot=True)

def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

#创建AGN自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:",'%04d' %(epoch + 1),"cost=","{:.9f}".format(avg_cost))
    print("Total cost: "+ str(autoencoder.calc_total_cost(X_test)))
