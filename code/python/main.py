from data.myData import *
from code.python.MulRadarTraceDiscernNet import *
from torch.autograd import Variable
import  torch
import sys


if __name__ == '__main__':
    net=CnnNet()
    net.load_state_dict(torch.load('../../nnModel/weight/net_save_200nose.pth'))

    opt_Adam = torch.optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.99))
    loss_func=MyLoss()

    data_=myDaDetection(sizeFig=256,
                        _time=5,figNum=100000,noseNum=200,
                        traceNum=8,traceRandom=True,
                        randomrange=0.5)
    data_loader = data.DataLoader(data_, 5,)


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
        loss,dif,diff2=loss_func(out,targets)

        opt_Adam.zero_grad()
        loss.backward()
        opt_Adam.step()
        if iteration%20 ==0 :
            print('第%d次,总损失:%f,航迹损失:%f, 噪声损失d%f',iteration,loss,dif,diff2)

        lossAll.append(loss)
        if iteration % 10 == 0:
            print('loss:', loss, 'diffrence:', dif, 'diff2', diff2)
        if loss < lossmin:
            lossmin = loss
            net_save = net
        if iteration % 100 == 0:
            print('saving model ......')
            torch.save(net_save.state_dict(), 'net_save.pth')
