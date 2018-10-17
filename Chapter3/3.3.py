#Tensorflow运行模型-会话

import tensorflow as tf
'''
sess = tf.Session()
sess.run()
sess.close()

#使用Python中的上下文管理器来管理会话
with tf.Session as sess:
    sess.run()
'''

a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = tf.add(a,b,name="add")
sess = tf.Session()
with sess.as_default():
    print(result.eval())

