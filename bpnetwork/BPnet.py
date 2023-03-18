import random

import numpy as np

random.seed(0)


def set_weights(m, n):

    w = np.random.uniform(-1, 1, (n, m))
    return w


def set_bias(m):

    b = np.random.uniform(-1, 0, (m, 1))
    return b


def act_fun(x, mode=0):
    if mode == 0:
        return np.tanh(x)
    if mode == 1:
        return 1/(1+np.exp(-x))


def der_act_fun(x, mode=0):

    if mode == 0:
        return 1 - x*x
    else :
        return x*(1-x)


def softmax(a):
    # a = a - np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


class BPnet:
    def __init__(self):
        self.input_n = 0
        self.input_cells = []
        self.input_ws = []
        self.hidden_layers = []
        self.hidden_outputs = []
        self.hidden_bias = []
        self.hidden_ws = []
        self.output_n = 0
        self.output_cells = []
        self.output_ws = []
        self.output_bias = []
        self.r = 1
        self.mode = 0

    def set_net(self, input_n, output_n, hidden_layers, r, mode):
        self.input_n = input_n
        self.output_n = output_n
        self.r = r
        self.mode = mode

        self.hidden_layers = np.array(hidden_layers)
        self.hidden_outputs = [0.0]*len(hidden_layers)
        self.input_ws = set_weights(input_n, hidden_layers[0])

        self.hidden_ws = [0.0]*(len(hidden_layers)-1)
        for i in range(len(hidden_layers)-1):
            self.hidden_ws[i] = set_weights(hidden_layers[i], hidden_layers[i+1])

        self.hidden_bias = [0.0]*len(hidden_layers)
        for i in range(len(hidden_layers)):
            self.hidden_bias[i] = set_bias(hidden_layers[i])

        self.output_ws = set_weights(hidden_layers[len(hidden_layers)-1], output_n)
        self.output_bias = set_bias(output_n)

    def forward(self, input):

        self.input_cells = input
        '''
        输入到隐藏层
        '''
        sum = np.dot(self.input_ws, self.input_cells)

        self.hidden_outputs[0] = act_fun(sum+self.hidden_bias[0], self.mode)
        '''
        隐藏层传递

        '''
        for i in range(len(self.hidden_layers)-1):
            sum = np.dot(self.hidden_ws[i], self.hidden_outputs[i])
            self.hidden_outputs[i+1] = act_fun(sum+self.hidden_bias[i+1], self.mode)

        '''
        隐藏层到输出层
        for i in range(self.output_n):
            sum = 0.0
            for j in range(self.hidden_layers[len(self.hidden_layers)-1]):
                sum += self.output_ws[j][i]*self.hidden_outputs[len(self.hidden_layers)-1][j]
            self.output_cells[i] = act_fun(self.output_bias[i]+sum, self.mode)
            if self.mode == 1:
                self.output_cells = softmax(self.output_cells)
        return self.output_cells[:]
        '''
        sum = np.dot(self.output_ws, self.hidden_outputs[len(self.hidden_layers)-1])
        if self.mode == 0:
            self.output_cells = act_fun(self.output_bias+sum, self.mode)
        elif self.mode == 1:
            self.output_cells = softmax(self.output_bias+sum)

        return np.array(self.output_cells)

    def back_adjust(self, target):
        # error = [0.0]*self.output_n
        # output_delta_ws = [0.0]*self.output_n
        # for i in range(len(target)):
        #     error[i] = target[i]-self.output_cells[i]
        #     output_delta_ws[i] = error[i]*der_act_fun(self.output_cells[i],self.mode)
        # 计算输出层delta weights
        error = target - self.output_cells
        output_delta_ws = error * der_act_fun(self.output_cells, self.mode)
        # pre_delta_ws = output_delta_ws
        # pre_ws = self.output_ws
        # hidden_delta_ws = [0.0]*(len(self.hidden_layers))
        # for i in range(len(self.hidden_layers)-1, -1, -1):
        #     hidden_delta_ws[i] = [0.0]*(self.hidden_layers[i])
        #     for j in range(self.hidden_layers[i]):
        #         for k in range(len(pre_delta_ws)):
        #             hidden_delta_ws[i][j] += pre_delta_ws[k]*pre_ws[j][k]
        #         hidden_delta_ws[i][j] *= der_act_fun(self.hidden_outputs[i][j],self.mode)
        #     if i > 0:
        #         pre_delta_ws = hidden_delta_ws[i]
        #         pre_ws = self.hidden_ws[i-1]
        # 计算隐藏层delta weights
        pre_delta_ws = output_delta_ws
        pre_ws = self.output_ws
        hidden_delta_ws = [0.0]*(len(self.hidden_layers))
        for i in range(len(self.hidden_layers)-1, -1, -1):
            hidden_delta_ws[i] = np.dot(np.transpose(pre_ws), pre_delta_ws)*der_act_fun(self.hidden_outputs[i])

            if i > 0:
                pre_delta_ws = hidden_delta_ws[i]
                pre_ws = self.hidden_ws[i-1]

        # for i in range(self.input_n):
        #     for j in range(len(self.input_ws[i])):
        #         self.input_ws[i][j] += self.r*hidden_delta_ws[0][j]*self.input_cells[i]
        # 更新输入层weights
        self.input_ws += self.r*np.dot(hidden_delta_ws[0], np.transpose(self.input_cells))

        # for i in range(len(self.hidden_layers)-1):
        #     for j in range(self.hidden_layers[i]):
        #         for k in range(self.hidden_layers[i+1]):
        #             self.hidden_ws[i][j][k] += self.r*hidden_delta_ws[i+1][k]*self.hidden_outputs[i][j]
        # 更新隐藏层weights
        for i in range(len(self.hidden_layers)-1):
            self.hidden_ws[i] += self.r*np.dot(hidden_delta_ws[i+1], np.transpose(self.hidden_outputs[i]))
        # 更新输出层weights
        # for i in range(len(self.output_ws)):
        #     for j in range(len(self.output_ws[i])):
        #         self.output_ws[i][j] += self.r*output_delta_ws[j]*self.hidden_outputs[len(self.hidden_layers)-1][i]
        self.output_ws += self.r*np.dot(output_delta_ws, np.transpose(self.hidden_outputs[len(self.hidden_layers)-1]))

        # for i in range(len(self.output_bias)):
        #     self.output_bias[i] += self.r*output_delta_ws[i]
        # 更新输出层bias
        self.output_bias += self.r*output_delta_ws

        # for i in range(len(self.hidden_layers)):
        #     for j in range(self.hidden_layers[i]):
        #         self.hidden_bias[i][j] += self.r*hidden_delta_ws[i][j]
        # 更新隐藏层bias
        for i in range(len(self.hidden_layers)):
            self.hidden_bias[i] += self.r*hidden_delta_ws[i]

    def get_error(self, target):
        error = 0.0
        for i in range(len(target)):
            error += 0.5*(target[i]-self.output_cells[i])**2
        return error

    def avg_error(self, inputs, targets):
        error = 0.0
        predicts = []
        for i in range(len(inputs)):
            predicts.append(self.forward(np.array([inputs[i]]))[0][0])
            for j in range(len(targets[i])):
                error += 0.5*(targets[i][j]-self.output_cells[j][0])**2
        error /= len(inputs)
        return error, predicts

    def correct_rate(self, inputs, targets):
        count = 0
        for i in range(len(inputs)):
            self.forward(inputs[i])
            index = np.argmax(self.output_cells)
            if targets[index]:
                count += 1
        return count*1.0/len(inputs)

    def train_sin(self, inputs, target, times):
        for i in range(times):
            for j in range(len(inputs)):
                self.forward(np.array([inputs[j]]))
                self.back_adjust(np.array([target[j]]))

    def train_classify(self, train_inputs, train_targets, test_inputs, test_outputs, times):
        train_correct_rates = []
        test_correct_rates = []
        for i in range(times):
            count = 0
            a = np.arange(12)
            np.random.shuffle(a)
            for j in a:
                b = np.arange(len(train_inputs[j]))
                np.random.shuffle(b)
                for k in b:
                    self.forward(np.array([train_inputs[j][k]]).T)
                    index = np.argmax(self.output_cells)
                    if train_targets[j][k][index] > 0:
                        count += 1
                    self.back_adjust(np.array([train_targets[j][k]]).T)
            rate1 = count*1.0/(12*len(train_inputs[0]))
            train_correct_rates.append(rate1)

            count = 0
            for j in range(12):
                for k in range(len(test_inputs[i])):
                    self.forward(np.array([test_inputs[i][j]]).T)
                    index = np.argmax(self.output_cells)
                    if test_outputs[i][j][index] > 0:
                        count += 1
            rate2 = count * 1.0 / (12 * len(train_inputs[0]))
            test_correct_rates.append(rate2)
            print('第', i+1, '次:')
            print('训练正确率:', rate1)
            print('测试正确率', rate2)
        return train_correct_rates, test_correct_rates



