from data.myData import *
from code.python.MulRadarTraceDiscernNet import *
from torch.autograd import Variable
import  torch
import sys


if __name__ == '__main__':
    net=CnnNet()

    opt_Adam = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_func=MyLoss()

    data_=myDaDetection(sizeFig=256,
                        _time=5,figNum=10000,noseNum=20,
                        traceNum=8,traceRandom=True,
                        randomrange=0.5)
    data_loader = data.DataLoader(data_, 1,)


    a=iter(data_loader)
    for i in range(len(a)):
        im,an=next(a)

        im=im.float()
        an=an.float()

        im=im.permute([0,4,1,2,3]).float()
        targets=an
        if torch.cuda.is_available():

            im=im.cuda()
            net=net.cuda()
            targets=an.cuda()

        im=Variable(im)
        targets=Variable(targets)


        out=net(im)
        print('out size',out.size())
        loss,dif,diff2=loss_func(out,targets)

        opt_Adam.zero_grad()
        loss.backward()
        opt_Adam.step()

        print(loss,dif,diff2)
