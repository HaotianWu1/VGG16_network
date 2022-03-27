import torch
import torch.nn as nn
import torch.nn.functional as F


lay_info = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class vgg(nn.Module):
    def __init__(self, in_ch, num_classes=10):
        super(vgg, self).__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.layers = self.make_layers(lay_info)
        self.classifer = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(4096, self.num_classes))


    def make_layers(self, lay_inf):
        lays = []
        pre_ch = self.in_ch
        for i in lay_inf:
            if i == 'M':
                lays.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                lays.extend([nn.Conv2d(pre_ch, i, kernel_size=3, padding=1), nn.BatchNorm2d(i), nn.ReLU(inplace=True)])
                pre_ch = i


        return nn.Sequential(*lays)


    def forward(self,x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out



if __name__ == '__main__':


    model = vgg(3, 10)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.size())









