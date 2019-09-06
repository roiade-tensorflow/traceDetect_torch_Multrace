from data.myData import *
from code.python.MulRadarTraceDiscernNet import *
from torch.autograd import Variable
import  torch
import sys


if __name__ == '__main__':
    net=CnnNet()
    net.load_state_dict(torch.load('net_save.pth'))

    opt_Adam = torch.optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.99))
    loss_func=My_loss()

    data_=myDaDetection(sizeFig=256,
                        _time=5,figNum=100000,noseNum=200,
                        traceNum=2,traceRandom=False,
                        randomrange=0.5)
    data_loader = data.DataLoader(data_,batch_size=1)


    a=iter(data_loader)

    lossAll=[]
    lossmin = 10000
    net_save = net
    for iteration in range(len(a)):
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
        # print('out size',out.size())
        loss=loss_func(out,targets)
        opt_Adam.zero_grad()
        loss.backward()
        opt_Adam.step()
        if loss<lossmin:
            lossmin = loss
            net_save = net
            print('saveing ...  loss:', loss)
            print('saving model ......')
            torch.save(net_save.state_dict(), 'net_save.pth')

        print(loss)

