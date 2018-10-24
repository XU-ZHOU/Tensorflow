'''
8.2 长短时记忆网络（LSTM）结构
  1、循环神经网络带来长期依赖问题。
  2、遗忘门和输入门是LSTM的核心。遗忘门的作用是让循环神经网络忘记之前没用的信息。

'''


'''
#实现LSTM结构的前向传播过程
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)
state = lstm.zero_state(batch_size,tf.float32)

loss = 0.0
for i in range(num_steps):
    if i > 0 : tf.get_variable_scope().reuse_variables()
    lstm_output,state = lstm(current_input,state)
    final_output = fully_connected(lstm_output)
    loss += calc_loss(final_output,expected_output)
'''

