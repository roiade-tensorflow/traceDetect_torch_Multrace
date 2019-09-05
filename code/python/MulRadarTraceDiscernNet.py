'''
用于多航迹识别
'''
import torch
import torch.nn as nn
class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet,self).__init__()

        self.conv3d_1= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16,
                      kernel_size=(3,3,3), stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU())
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU())



        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU())

        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU())

        self.conv3d_5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(1),
            nn.ReLU())

        self.maxpool_1 = nn.Sequential(
            nn.MaxPool3d((3, 3, 1), stride=(2, 2, 1),padding=(1,1,0))
           )
#linear | bilinear | bicubic | trilinear
        self.upSample=nn.Sequential(
        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

    def forward(self, x):
        out = self.conv3d_1(x)
        print(out.size())
        out=self.maxpool_1(out)
        out=self.conv3d_2(out)
        out=self.maxpool_1(out)
        out=self.conv3d_3(out)
        out=self.maxpool_1(out)

        out = self.upSample(out)
        out=self.conv3d_4(out)

        out = self.upSample(out)
        out = self.conv3d_4(out)

        out = self.upSample(out)
        out = self.conv3d_4(out)

        out = self.conv3d_5(out)

        return out

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, x, y):  # 定义前向的函数运算即可
        x=torch.exp(-x)
        x=1/(1+x)
        y=torch.exp(-y)
        y=1/(1+y)
        d = -(y * torch.log(x) + (1 - y) * torch.log(1 - x))
        return torch.sum(d)
        # return torch.sum(100*torch.pow((x - y)*y, 2)+torch.pow((x - y), 2))
        # return( nn.CrossEntropyLoss(x,y) )

    def myNomal(a):
        b = torch.exp(-a)
        c = 1 / (1 + b)
        # d = -(expect * torch.log(c) + (1 - expect) * torch.log(1 - c))
        return c

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, pre, label):  # 定义前向的函数运算即可
        num = 1
        for i in range(len(torch.tensor(pre.size()))):
            num *= pre.size()[i]
        c = torch.pow(label - pre, 2)
        d = torch.pow((40) * (label - pre) * label, 2)
        f = torch.sum(c +d)
        g=torch.sum(torch.pow((label - pre) * label, 2))
        return f,g,torch.sum(c)