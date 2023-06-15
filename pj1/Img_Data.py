import torch
from PIL import Image
import os
from torch.utils.data import Dataset


class ImageData(Dataset):
    def __init__(self, dir, transformer=None):
        self.dataInfo = []
        for i in range(12):
            label = i
            for img in os.listdir(os.path.join(dir, str(i+1))):  # 得到图片读取路径
                imgpath = os.path.join(dir, str(i+1), img)
                self.dataInfo.append((imgpath, label))

        self.transformer = transformer

    def __getitem__(self, index):
        imgpath, label = self.dataInfo[index]
        img = Image.open(imgpath).convert('L')  # 读取图片并转化为灰度图
        if self.transformer is not None:
            img = self.transformer(img)
        l = [0]*12
        l[label] = 1
        return img, torch.FloatTensor(l)

    def __len__(self):
        return len(self.dataInfo)
