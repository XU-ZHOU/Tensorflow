'''
8.3 循环神经网络的变种
  1、循环神经网络一般只在不同层循环体结构之间使用dropout，而不再同一层的循环体中之间使用。

'''

'''
# Tensorflow实现带dropout的循环神经网络
lstm = rnn_cell.BasicLSTMCell(lstm_size)
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=0.5)
stacked_lstm = rnn_cell.MutiRNNCell([dropout_lstm]*number_of_layers)

'''
