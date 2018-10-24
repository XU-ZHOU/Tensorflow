'''
第7章 图像数据处理
  1、Tensorflow提供TFRecord的格式统一存储数据。

'''
#TFRecord样例程序
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("H:/ML/Tensorflow/tensorflow-tutorial-master/Deep_Learning_with_TensorFlow/datasets",dtype=tf.uint8,one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = "H:/ML/Tensorflow/tensorflow-tutorial-master/to/output.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels':_int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()

#读取TFRecord文件中的数据
import tensorflow as tf
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(
    ['H:/ML/Tensorflow/tensorflow-tutorial-master/to/output.tfrecords'])
serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features = {
        '''
        Tensorflow提供两种不同的属性解析方法，一种是tf.FixedLenFeature,得到的
        解析结果为一个Tensor。另一种方法是tf.VarLenFeature,这种方法得到的解析结果
        是SparseTensor,用于处理稀疏数据。
        '''
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64),})
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(10):
    image,label,pixel = sess.run([images,labels,pixels])



