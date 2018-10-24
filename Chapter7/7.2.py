'''
7.2 图像数据处理

'''
# 使用Tensorflow对jpeg格式图像编码、解码函数
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("H:/ML/Tensorflow/datasets/cat.jpg", 'r').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

plt.imshow(img_data.eval())
plt.show()

img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
encoded_image = tf.image.encode_jpeg(img_data)
with tf.gfile.GFile("","wb") as f:
    f.write(encoded_image.eval())

