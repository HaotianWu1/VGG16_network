from torchvision import datasets as ds
from torch.utils.data import DataLoader
from torchvision import transforms as ts
import torchvision as tv
import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
from torch import optim
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn

transform = ts.Compose(
    [
        ts.ToTensor(),
        ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = ds.CIFAR10(root='./data',
                       train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

testset = tv.datasets.CIFAR10(root='./data',
                              train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=64, shuffle=True, num_workers=0)

test_data_size = len(testset)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Conv1
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),  # Conv2
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(32, 64, 3, padding=1),  # Conv3
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  # Conv4
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(64, 128, 3, padding=1),  # Conv5
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv6
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv7
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool3
            nn.Conv2d(128, 256, 3, padding=1),  # Conv8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv9
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool4
            nn.Conv2d(256, 256, 3, padding=1),  # Conv11
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv12
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv13
            nn.ReLU(True),
            # nn.MaxPool2d(2, 2)  # Pool5 是否还需要此池化层，每个通道的数据已经被降为1*1
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

total_train_step = 0

total_test_step = 0


m = VGGNet(10)
m.cuda()
lr = 1e-3
momentum = 0.9

num_epoch = 50

loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(m.parameters(), lr=lr, momentum=momentum)

print('Training with learning rate = %f, momentum = %f ' % (lr, momentum))

for i in range(num_epoch):
    print("epoch {}:".format(i))
    m.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = m(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1

        if (total_train_step % 50 == 0):
            print("train {}, loss {}".format(total_train_step, loss.item()))

    m.eval()
    total_test_loss = 0
    total_accuracy = 0
    with(torch.no_grad()):
        for data in testloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = m(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("total test loss {}".format(total_test_loss))
    print("total accuracy {}".format(total_accuracy / test_data_size))
    total_test_step = total_test_step + 1
