'''
第6章 图像识别与卷积神经网络
1、一个卷积神经网络由5中结构组成：输入层、卷积层、池化层、全连接层、softmax层。
2、卷积神经网络中，每一个卷积层中使用的过滤器中的参数都是一样的。
3、卷积层的参数个数之和卷积核的尺寸、深度以及当前层节点矩阵的深度有关。
4、池化层缩小矩阵的尺寸，减少最后全连接层中的参数，使用池化层既可以加快计算速度，也可以防止过拟合。
'''
import tensorflow as tf
filter_weight = tf.get_variable(
    'weights',[5,5,3,16],
    initializer=tf.truncated_normal_initializer(stddev=0.1))

biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input,filter_weight,strides=[1,1,1,1],padding='SAME')
bias = tf.nn.bias_add(conv,biases)
actived_conv = tf.nn.relu(bias)

pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')



'''
6.4 经典卷积网络模型
  1、LeNet-5：第一层，卷积层；第二层，池化层；第三层，卷积层；第四层，池化层；
  第五层，全连接层；第六层，全连接层；第七层，全连接层；
  2、dropout避免过拟合问题，一般只在全连接层使用。
  3、LeNet-5和Inception-v3模型的区别，LeNet-5中不同卷积层通过串联方式连接，
     Inception-v3中不同卷积层通过并联的方式连接。
'''

'''
6.5 卷积神经网络迁移学习
  1、迁移学习是将一个问题上训练好的模型通过简单的调整使其适用于一个新的问题。
  
'''
#利用数据集和已经训练好的模型完成迁移学习
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = 'H:/ML/Tensorflow/tensorflow-tutorial-master/Deep_Learning_with_TensorFlow/datasets/inception_dec_2015'
MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = 'H:/ML/Tensorflow/tensorflow-tutorial-master/tmp'
INPUT_DATA = 'H:/ML/Tensorflow/tensorflow-tutorial-master/Deep_Learning_with_TensorFlow/datasets/flower_photos'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100
def create_image_lists(testing_percentage,validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg','jpeg','JPG','JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:continue

        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images
        }
    return result

def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path

def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,
                          label_name,index,category) + '.txt'

def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                                {image_data_tensor:image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def get_or_create_bottleneck(
        sess,image_lists,label_name,index,
        category,jpeg_data_tensor,bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists,label_name,index,category)
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(
            image_lists,INPUT_DATA,label_name,index,category)
        image_data = gfile.FastGFile(image_path,'rb').read()
        bottleneck_values = run_bottleneck_on_image(
            sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def get_random_cached_bottlenecks(
        sess,n_classes,image_lists,how_many,category,
        jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index,label_name in enumerate(label_name_list):
        category = 'testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,index,category,
                                                  jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def main():
    image_lists = create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        grapg_def = tf.GraphDef()
        grapg_def.ParseFromString(f.read())
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(
        grapg_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

    bottleneck_input = tf.placeholder(
        tf.float32,[None,BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(
        tf.float32,[None,n_classes],name='GroundTruthInput')
    with tf.name_scope("final_training_ops"):
        weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights) + biases
        final_tensor = tf.nn.softmax(logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),
                                      tf.argmax(ground_truth_input,1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(STEPS):
            train_bottleneck,train_ground_truth = get_random_cached_bottlenecks(
                sess,n_classes,image_lists,BATCH,
                'training',jpeg_data_tensor,bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input:train_bottleneck,
                                ground_truth_input:train_ground_truth})
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks,validation_ground_truth = get_random_cached_bottlenecks(
                    sess,n_classes,image_lists,BATCH,
                    'validation',jpeg_data_tensor,bottleneck_tensor)
                validation_accuracy = sess.run(
                    evaluation_step,feed_dict={
                        bottleneck_input:validation_bottlenecks,
                        ground_truth_input:validation_ground_truth})
                print('Step %d :Validation accuracy on random sampled' '%d examples = %.1f%%' %(i,BATCH,validation_accuracy*100))

            test_bottlenecks,test_ground_truth = get_test_bottlenecks(
                sess,image_lists,n_classes,jpeg_data_tensor,
                bottleneck_tensor)
            test_accuracy = sess.run(evaluation_step,feed_dict={
                bottleneck_input:test_bottlenecks,
                ground_truth_input:test_ground_truth})
            print('Final test accuracy = %.1f%%'%(test_accuracy*100))

if __name__ == '__main__':
    tf.app.run()