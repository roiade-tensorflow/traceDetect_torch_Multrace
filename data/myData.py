import  os.path as osp
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from code.python.traceProduce import traceImageProduce
from torch.autograd import Variable
from code.Include.VectorOperation import *
from  code.Include.vecOpFun import *
from code.Include.plotFunc import *
import numpy as np
class myDaDetection(data.Dataset):
    def __init__(self,sizeFig=1024,_time=9,figNum=100,noseNum=10,traceNum = 5,traceRandom=False,randomrange=0.8):
        self.traceRandom=traceRandom
        self.figNum=figNum
        self._width=sizeFig
        self._height=sizeFig
        self._time=_time
        self.randomrange=randomrange
        self.data=traceImageProduce(1,sizeFig=sizeFig,time=_time,
                            trainSavePath='../../data/Images/',
                            labelSavePaht='../../data/Annatations/trace/',
                            dataNameSavePath='../../data/Main/')

        self.ids=list()
        for i in range(figNum):
            self.ids.append(i)
        self.noseNum=noseNum
        self.traceNum=traceNum
    def __getitem__(self,index):
        tracenum=self.traceNum
        if(self.traceRandom):

            low_random=int((1-self.randomrange)*self.traceNum ) if int((1-self.randomrange)*self.traceNum)>0  else 1
            tracenum=np.random.uniform(low=low_random,high=int((1+self.randomrange)*self.traceNum))

            tracenum=int(tracenum)
        print(tracenum)
        nose=self.data.noseProdece_mul(noseNumber=self.noseNum)
        trace=self.data.traceProduce_mul(traceNum=tracenum)
        im=torch.cat((trace,nose),dim=0)

        im[:,0:2]=PolarToRec(im[:,0:2])
        trace[:,0:2]=PolarToRec(trace[:,0:2])

        ana=trace
        image=torch.zeros(size=(self._width,self._height,self._time,1),dtype=im.dtype)
        im[:,0:2]=torch.clamp(im[:,0:2],0,1)
        im[:,0:1]=self._width*im[:,0:1]
        im[:,1:2]=self._height*im[:,1:2]
        im=im.long()
        image[im[:,0],im[:,1],im[:,2],0]=1
        # return image
        annatition = torch.zeros(size=(self._width, self._height,self._time), dtype=im.dtype)
        ana[:, 0:2] = torch.clamp(ana[:, 0:2], 0, 1)
        ana[:, 0:1] = self._width * ana[:, 0:1]
        ana[:, 1:2] = self._height * ana[:, 1:2]
        ana = ana.long()
        annatition[ana[:, 0],ana[:, 1], ana[:, 2] ] = 1
        return image , annatition


    def __len__(self):
        return (self.figNum)
if __name__ =="__main__":
    data_=myDaDetection(sizeFig=256,_time=5,figNum=10000,noseNum=20,traceNum=5,traceRandom=True)
    data_loader = data.DataLoader(data_, 5,)
    a=iter(data_loader)
    im,an=next(a)
    imageShow(im[-1,:,:],scale=2)
    print(data_.__len__())






