'''
7.3 多线程输入数据处理框架
  1、Tensorflow提供两种队列，FIFOQueue和RandomShuffleQueue。
  2、Tensorflow提供tf.Coordinator和tf.QueueRunner两个类完成多线程协同功能。

'''
import tensorflow as tf
q = tf.FIFOQueue(2,"int32")
init = q.enqueue_many(([0,10],))
x = q.dequeue()
y = x + 1
q_inc  = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v,_ = sess.run([x,q_inc])
        print(v)

