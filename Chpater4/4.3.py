#神经网络训练大致过程
'''
batch_size = n

x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input'
y_ = tf.placeholder(tf.float32,shape=(batch_size,1,name='y-input')
loss = 
train_step = tf.train.AdamOptimizer(0.001).minimize(lose)
with tf.Session() as sess:
    for i in range(STEPS):
        current_X,current_Y=
        sess.run(train_step,feed_dict={x:current_X,y_:current_Y}
'''

