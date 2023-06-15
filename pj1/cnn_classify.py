import Img_Data
import CNN
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from matplotlib import pylab

BatchSize = 10
transformer = transforms.Compose([
    transforms.ToTensor()  # 将数据转化为张量，并且归一化
    ]
)
train_data = Img_Data.ImageData('train', transformer)
test_data = Img_Data.ImageData('test', transformer)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BatchSize,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=BatchSize,
    shuffle=True
)

cnn = CNN.CNN()
cnn.int_para()

optimizer = optim.SGD(
    params=cnn.parameters(),
    lr=0.004
)

losses = nn.CrossEntropyLoss()

if __name__ == '__main__':
    times = 60
    train_acc = []
    test_acc = []
    for step in range(times):
        total = 0
        corrects = 0
        cnn.train()

        for data in train_loader:
            img, label = data
            total += BatchSize
            output = cnn.forward(img)
            res = torch.max(output.data, 1)[1]

            target = torch.max(label.data, 1)[1]
            corrects += (res == target).sum().item()
            loss = losses(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_acc.append(corrects/total)
        print(f"第{step+1}次训练正确率为:{corrects/total}")
        total = 0
        corrects = 0
        for data in test_loader:
            img, label = data
            total += BatchSize
            output = cnn.forward(img)
            res = torch.max(output.data, 1)[1]
            target = torch.max(label.data, 1)[1]
            corrects += (res == target).sum().item()
        test_acc.append(corrects / total)
        print(f'第{step+1}次测试正确率为:{corrects / total}')

    # torch.save(cnn, 'data1.txt')
    x = [i+1 for i in range(times)]
    pylab.plot(x, train_acc, label='train_acc')
    pylab.plot(x, test_acc, label='test_acc')
    pylab.xlabel('times')
    pylab.ylabel('acc')
    pylab.plt.legend()
    pylab.plt.show()


