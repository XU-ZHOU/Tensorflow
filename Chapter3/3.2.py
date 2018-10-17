#Tensorflow数据模型-张量

import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = tf.add(a,b,name="add")
print(result)  #只是结果的引用，并不是具体结果

