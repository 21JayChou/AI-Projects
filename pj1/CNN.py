import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #  灰度图只有一个图层，设置参数使第一层卷积后生成三个图层，
        #  卷积核大小设为5*5，步长设为1，卷积后一个图层大小变为24*24
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(5, 5),
            stride=(1, 1)
        )
        # 卷积并激活后，做尺寸为2*2的2步长池化，数据尺寸变为12*12*3
        # 第二层卷积设置输出图层数为10，卷积核尺寸及步长不变，卷积后数据变为8*8*10
        self.conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=(5, 5),
            stride=(1, 1)
        )
        # 同样经过池化后，数据尺寸变为4*4*10
        # 因此第一层全连接层输入神经元为4*4*10，设置输出个数为100
        self.lin1 = nn.Linear(
            in_features=4 * 4 * 10,
            out_features=100
        )
        # 第二层全连接层为最后一层，因此输出神经元个数与图片类别数相同(12)
        self.lin2 = nn.Linear(
            in_features=100,
            out_features=12
        )
        # self.lin3 = nn.Linear(
        #     in_features=50,
        #     out_features=12
        # )

    def forward(self, _input):
        output = self.conv1(_input)  # 第一层卷积
        output = F.relu(output)      # 激活
        output = F.max_pool2d(output, (2, 2), 2)  # 最大值池化
        output = self.conv2(output)  # 第二层卷积
        output = F.relu(output)      # 池化
        output = F.max_pool2d(output, (2, 2), 2)  # 最大值池化
        output = output.view(output.size(0), -1)  # 将同一个batch的数据压缩为1列
        output = self.lin1(output)
        output = F.relu(output)
        output = self.lin2(output)
        # output = F.relu(output)
        # output = self.lin3(output)
        # output = F.softmax(output, dim=1)

        return output

    def int_para(self):  # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)  # 卷积层权重用kaiming初始化
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)  # 全连接层权重用标准方差初始化
                m.bias.data.zero_()