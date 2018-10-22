'''
知识点总结：
2.1.1 Protocol Buffer
  1、结构化数据指的是拥有多种属性的数据。
  2、序列化是将结构化的数据变成数据流的格式，简单的说就是变为一个字符串。
  3、如何将结构化的数据序列化，并从序列化之后的数据流中还原出原来的结构化数据，
     统称为处理结构化数据，这就是Protocol Buffer解决的主要问题。
  4、Protocol Buffer序列化之后得到的数据不是可读的字符串，而是二进制流。
  5、Protocol Buffer定义了属性是必须的、可选的或者可重复的。
  6、Tensorflow中的数据基本都是通过Protocol Buffer来组织的。

2.1.2 Bazel
  1、Tensorflow通过Bazel编译
  2、Bazel对Python支持的编译方式有三种：py_binary,py_library,py_test
     其中，py_binary将Python程序编译为可执行文件，py_test编译Python测试程序，
     py_library将Python程序编译成库函数供其他py_binary或py_test调用。
  3、
'''
import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = a + b
sess = tf.Session()
print(sess.run(result))