'''
第8章 循环神经网络
  1、循环神经网络的主要用途是处理和预测序列数据。
  2、循环神经网络的来源是刻画一个序列当前的输出与之前信息之间的关系。
  3、循环神经网络的隐藏层之间的节点是有连接的，隐藏层的输入不仅包含输入层的输出，
  还包括上一时刻隐藏层的输出。
  4、循环神经网络可以被看做是同一神经网络结构在时间序列上被复制多次的结果，
  这个被复制多次的结构被称之为循环体。
  5、和卷积神经网络过滤器中参数是共享的类似，在循环神经网络中，循环体网络结构中
  的参数在不同时刻也是共享的。
  6、循环体中的神经网络的输入有两部分，一部分是上一时刻的状态，另一部分是当前时刻的输入样本。
  7、理论上循环神经网络可以支持任意长度的序列，但实际中，如果序列过长会导致优化时出现梯度消散问题。
  在训练中一般会规定一个最大的序列长度。
'''

#实现简单的循环神经网络前向传播
import numpy as np
X = [1,2]
state = [0.0,0.0]
w_cell_state = np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input = np.asarray([0.5,0.6])
b_cell = np.asarray([0.1,-0.1])

w_output = np.asarray([[1.0],[2.0]])
b_output = 0.1

for i in range(len(X)):
    before_activation = np.dot(state,w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state,w_output) + b_output
    print("before activation: ",before_activation)
    print("state: ",state)
    print("output: ",final_output)

