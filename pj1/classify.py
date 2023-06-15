import os
import cv2
import numpy as np
import BPnet
import pickle
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
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    for i in range(12):
        t = 12*[0.0]
        t[i] = 1
        train_amount = int(len(inputs[i])*0.9)
        test_amount = int(len(inputs[i])*0.1)
        for j in range(train_amount):
            train_inputs.append(inputs[i][j])
            train_outputs.append(t)
        for j in range(train_amount, len(inputs[i])):
            test_inputs.append(inputs[i][j])
            test_outputs.append(t)

    return train_inputs, train_outputs, test_inputs, test_outputs





if __name__ == '__main__':
    bpnet.set_net(784, 12, [200,100], 0.005, 1)
    #print(bpnet.output_ws)
    times = 200
    train_inputs, train_outputs, test_inputs, test_outputs = get_datas("train")
    train_correct_rates, test_correct_rates = bpnet.train_classify(train_inputs, train_outputs, test_inputs, test_outputs, times)
    # file = open("data.txt", "wb")
    # pickle.dump(bpnet, file)
    # file.close()
    x = np.arange(1, times+1)
    pylab.plot(x, train_correct_rates, label='train_correct_rates')
    pylab.plot(x, test_correct_rates, label='test_correct_rates')
    pylab.plt.xlabel('times')
    pylab.plt.ylabel('correct_rate')
    pylab.plt.show()










