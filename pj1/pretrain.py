import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()]
)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=10, shuffle=False)
model = torchvision.models.resnet18(pretrained=True)

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
inchannels = model.fc.in_features
model.fc = nn.Linear(inchannels, 10)
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)
losses = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    losses = losses.cuda()


for epoch in range(30):
    model.train()
    total = 0
    corrects = 0
    for data in trainloader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        model.zero_grad()
        outputs = model(inputs)
        prediects = torch.max(outputs.data, 1)[1]
        loss = losses(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(model, 'data2.txt')
total = 0
corrects = 0
for data in testloader:
    inputs, labels = data
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    total += labels.size(0)
    outputs = model.forward(inputs)
    prediects = torch.max(outputs.data, 1)[1]
    corrects += (prediects == labels).sum().item()
print(f'测试正确率为:{corrects / total}')


