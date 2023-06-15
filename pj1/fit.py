import numpy as np
import math
import BPnet
import random
from matplotlib import pylab
bpnet = BPnet.BPnet()


def train_datas():
    train_x = []
    train_y = []
    for i in range(-20, 20, 1):
        train_x.append([i * math.pi / 20])
        train_y.append(np.sin([i * math.pi / 20]))
    return train_x, train_y


def test_datas():
    datas = []
    test_x = []
    for i in range(200):
        t = random.uniform(-math.pi, math.pi)
        datas.append(t)
    datas.sort()
    for i in range(len(datas)):
        test_x.append([datas[i]])
    test_y = np.sin(test_x)
    return test_x, test_y




if __name__ == '__main__':
    bpnet.set_net(1, 1, [10,5], 0.02, 0)
    times = 1000
    train_x, train_y = train_datas()
    bpnet.train_sin(train_x, train_y, times)

    test_x, test_y = test_datas()
    error = 0.0
    predicts = []
    error, predicts = bpnet.avg_error(test_x, test_y)
    print("平均误差为：", error)
    pylab.plt.scatter(train_x, train_y, marker='x', color='k', label='train set')

    x = np.arange(-1 * np.pi, np.pi, 0.01)
    y = np.sin(x)
    pylab.plot(x, y, label='standard sinx')

    pylab.plot(test_x, predicts, label='predicate sinx', color='r')
    pylab.plt.xlabel('x')
    pylab.plt.ylabel('y')
    pylab.plt.legend()
    pylab.plt.show()


