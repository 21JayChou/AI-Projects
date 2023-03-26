import os
import BPnet
import cv2
import numpy as np
from matplotlib import pylab
np.set_printoptions(threshold=np.inf)
bpnet = BPnet.BPnet()


def get_image_bits(path):
    image_bits = [0.0]*12
    for i in range(12):
        images = os.listdir(os.path.join(path, str(i+1)))
        image_bits[i] = [0.0]*len(images)
        for j in range(len(images)):
            image = cv2.imread(os.path.join(path, str(i+1), images[j]), cv2.IMREAD_GRAYSCALE)
            image_bits[i][j] = np.ndarray.flatten(image/255)
    return image_bits


def get_datas(path):
    inputs = get_image_bits(path)
    train_inputs = [0.0]*12
    train_outputs = [0.0]*12
    test_inputs = [0.0]*12
    test_outputs = [0.0]*12

    for i in range(12):
        t = 12*[0.0]
        t[i] = 1.0
        train_amount = int(len(inputs[i])*9/10)
        test_amount = int(len(inputs[i])/10)
        train_inputs[i] = [0.0]*train_amount
        train_outputs[i] = [0.0]*train_amount
        for j in range(train_amount):
            train_inputs[i][j] = inputs[i][j]
            train_outputs[i][j] = t
        test_inputs[i] = [0.0]*test_amount
        test_outputs[i] = [0.0]*test_amount
        for j in range(train_amount, len(inputs[i])):
            test_inputs[i][j-train_amount] = inputs[i][j]
            test_outputs[i][j-train_amount] = t

    return train_inputs, train_outputs, test_inputs, test_outputs


def train(inputs, targets, times):
    for i in range(times):
        for j in range(12):
            for k in range(len(inputs[j])):
                bpnet.forward(inputs[j][k])
                bpnet.back_adjust(targets[j][k])


if __name__ == '__main__':
    bpnet.set_net(784, 12, [1000,300], 0.01, 1)
    times = 50
    train_inputs, train_outputs, test_inputs, test_outputs = get_datas("train")

    train_correct_rates, test_correct_rates = bpnet.train_classify(train_inputs, train_outputs, test_inputs, test_outputs, times)
    count = 0
    '''
    x = np.arange(1, times+1)
    pylab.plot(x, train_correct_rates, label='train_correct_rates')
    pylab.plot(x, test_correct_rates, label='test_correct_rates')
    pylab.plt.xlabel('times')
    pylab.plt.ylabel('correct_rate')
    pylab.plt.show()
    '''









