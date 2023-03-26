import random

import numpy as np

random.seed(0)


def set_weights(m, n):

    w = np.random.uniform(-0.5, 0.5, (n, m))
    return w


def set_bias(m):

    b = np.random.uniform(-1, 0, (m, 1))
    return b


def act_fun(x, mode=0):
    if mode == 0:
        return np.tanh(x)
    if mode == 1:
        indices_pos = np.nonzero(x >= 0)
        indices_neg = np.nonzero(x < 0)

        y = np.zeros_like(x)
        y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
        y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
        return y


def der_act_fun(x, mode=0):

    if mode == 0:
        return 1 - x*x
    else:
        return x*(1-x)


def softmax(a):
    a = a-a.max()
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def reciprocal(x):
    t = np.zeros([len(x), 1])
    for i in range(len(x)):
        t[i][0] = 1/x[i][0]
    return t


class BPnet:
    def __init__(self):
        self.input_n = 0
        self.input_units = []
        self.input_ws = []
        self.hidden_layers = []
        self.hidden_outputs = []
        self.hidden_bias = []
        self.hidden_ws = []
        self.output_n = 0
        self.output_units = []
        self.output_ws = []
        self.output_bias = []
        self.r = 1
        self.mode = 0
        self.soft_outputs = []

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

        self.input_units = input
        '''
        输入到隐藏层
        '''
        sum = np.dot(self.input_ws, self.input_units)

        self.hidden_outputs[0] = act_fun(sum+self.hidden_bias[0], self.mode)
        '''
        隐藏层传递

        '''
        for i in range(len(self.hidden_layers)-1):
            sum = np.dot(self.hidden_ws[i], self.hidden_outputs[i])
            self.hidden_outputs[i+1] = act_fun(sum+self.hidden_bias[i+1], self.mode)

        '''
        隐藏层到输出层
    
        '''
        sum = np.dot(self.output_ws, self.hidden_outputs[len(self.hidden_layers)-1])
        if self.mode == 0:
            self.output_units = act_fun(self.output_bias+sum, self.mode)
        elif self.mode == 1:
            self.output_units = softmax(self.output_bias+sum)
        # self.output_units = act_fun(self.output_bias+sum, self.mode)
        # if self.mode == 1:
        #     self.soft_outputs = softmax(self.output_units)
        # if self.mode == 0:
        #     return np.array(self.output_units)
        # return np.array(self.soft_outputs)
        return np.array(self.output_units)

    def back_adjust(self, target):

        # 计算输出层delta weights
        if self.mode == 0:
            output_delta_ws = (target - self.output_units) * der_act_fun(self.output_units, self.mode)
        else:
            output_delta_ws = target - self.output_units
            #output_delta_ws = (target - self.soft_outputs) * der_act_fun(self.output_units, self.mode)

        pre_delta_ws = output_delta_ws
        pre_ws = self.output_ws
        hidden_delta_ws = [0.0]*(len(self.hidden_layers))
        for i in range(len(self.hidden_layers)-1, -1, -1):
            hidden_delta_ws[i] = np.dot(np.transpose(pre_ws), pre_delta_ws)*der_act_fun(self.hidden_outputs[i])

            if i > 0:
                pre_delta_ws = hidden_delta_ws[i]
                pre_ws = self.hidden_ws[i-1]


        # 更新输入层weights
        self.input_ws += self.r*np.dot(hidden_delta_ws[0], np.transpose(self.input_units))


        # 更新隐藏层weights

        for i in range(len(self.hidden_layers)-1):
            self.hidden_ws[i] += self.r*np.dot(hidden_delta_ws[i+1], np.transpose(self.hidden_outputs[i]))
        # 更新输出层weights

        self.output_ws += self.r*np.dot(output_delta_ws, np.transpose(self.hidden_outputs[len(self.hidden_layers)-1]))


        # 更新输出层bias
        self.output_bias += self.r*output_delta_ws


        # 更新隐藏层bias
        for i in range(len(self.hidden_layers)):
            self.hidden_bias[i] += self.r*hidden_delta_ws[i]

    def avg_error(self, inputs, targets):
        error = 0.0
        predicts = []
        for i in range(len(inputs)):
            predicts.append(self.forward(np.array([inputs[i]]))[0][0])
            for j in range(len(targets[i])):
                error += 0.5*(targets[i][j]-self.output_units[j][0])**2
        error /= len(inputs)
        return error, predicts

    def correct_rate(self, inputs, targets):
        count = 0
        for i in range(len(inputs)):
            self.forward(inputs[i])
            index = np.argmax(self.output_units)
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
            train_loss = 0
            test_loss = 0
            count = 0
            a = np.arange(12)
            np.random.shuffle(a)
            for j in a:
                b = np.arange(len(train_inputs[j]))
                np.random.shuffle(b)
                for k in b:
                    self.forward(np.array([train_inputs[j][k]]).T)
                    train_loss += -np.sum(np.log(self.output_units)*np.array([train_targets[j][k]]).T)
                    index = np.argmax(self.output_units)
                    if train_targets[j][k][index] > 0:
                        count += 1
                    self.back_adjust(np.array([train_targets[j][k]]).T)
            rate1 = count*1.0/(12*len(train_inputs[0]))
            train_correct_rates.append(rate1)

            count = 0
            for j in range(12):
                for k in range(len(test_inputs[j])):
                    self.forward(np.array([test_inputs[j][k]]).T)
                    test_loss += -np.sum(np.log(self.output_units) * np.array([test_outputs[j][k]]).T)
                    index = np.argmax(self.output_units)
                    if test_outputs[j][k][index] > 0:
                        count += 1
            rate2 = count * 1.0 / (12 * len(test_inputs[0]))
            test_correct_rates.append(rate2)
            print('第', i+1, '次:')
            print('训练正确率:', rate1)
            print('数据集平均loss:', train_loss/(len(train_inputs)*len(train_inputs[0])))
            print('测试正确率', rate2)
            print('测试集平均loss:', test_loss/(len(test_inputs)*len(test_inputs[0])))
        return train_correct_rates, test_correct_rates



