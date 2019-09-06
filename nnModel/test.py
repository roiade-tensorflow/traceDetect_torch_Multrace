from data.myData import *
from code.python.MulRadarTraceDiscernNet import *
from torch.autograd import Variable
import  torch
import sys
from code.Include.plotFunc import *

if __name__ == '__main__':
    net=CnnNet()

    opt_Adam = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))
    loss_func=MyLoss()

    data_=myDaDetection(sizeFig=256,
                        _time=5,figNum=10000,noseNum=200,
                        traceNum=8,traceRandom=True,
                        randomrange=0.5)
    data_loader = data.DataLoader(data_, 1,)
    a=iter(data_loader)
    lossAll=[]
    lossmin = 10000
    net.load_state_dict(torch.load('./weight/net_save_200nose.pth'))
    for k in range(6):
        im, an = next(a)

        im = im.float()
        an = an.float()

        im = im.permute([0, 4, 1, 2, 3]).float()
        targets = an
        if torch.cuda.is_available():
            im = im.cuda()
            net = net.cuda()
            targets = an.cuda()

        im = Variable(im)
        targets = Variable(targets)

        yy=net(im)

        yy=yy.cpu().detach().numpy()

        dirTa = np.where(yy > 0.85)
        targetsFromNet=np.zeros_like(yy)
        targetsFromNet[dirTa[0],dirTa[1],dirTa[2],dirTa[3],dirTa[4]]=1
        imagesForDectet = im.cpu().detach().numpy()
        detectedResult=np.logical_and(imagesForDectet ,targetsFromNet )
        print(targets.size())

        targets=targets.cpu().detach().numpy()

        imageShow2(imagesForDectet[0,0, :, :, :],detectedResult[0,0,:,:,:],targets[0,  :, :, :])
        # imageShow(detectedResult[0,0,:,:,:])
        # imageShow(targets[0,  :, :, :])
        # imageShow(imagesForDectet[0,0, :, :, :])

        # print(detectedResult.sum())
        # print('len(dirTa)',len(dirTa))
        # print(dirTa)


