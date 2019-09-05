from __future__ import division
import sys
sys.path.append('../../Include/')
from code.Include.VectorOperation import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from code.Include import vecOpFun
import random
import time
import warnings
warnings.filterwarnings("ignore")

class traceImageProduce():
    def __init__(self,x,sizeFig=1024,time=9,
                 trainSavePath='../../data/Images/',
                 labelSavePaht='../../data/Annatations/trace/',
                 dataNameSavePath='../../data/Main/'):
        self.parameterInit(sizeFig,time=time)
        self.traceNum = x
        self.vecOp=vecOpFun.vecOpration()
        self.faultDtype=torch.float64
        self.detype=torch.float32
        self.trainSavePath=trainSavePath
        self.labelSavePath=labelSavePaht

        self.dataName=dataNameSavePath
    def parameterInit(self,sizeFig,time):
        self.dimNum=2
        self.figureWidth=sizeFig
        self.figureHidth=sizeFig
        self.noseScale = 2 / sizeFig
        self.figureTime=time
        self.quantizedInternal=37.5
        self.vlim=[100/(self.figureWidth*self.quantizedInternal),200/(self.figureWidth*self.quantizedInternal)]
        self.alim=[10/(self.figureWidth*self.quantizedInternal),20/(self.figureWidth*self.quantizedInternal)]

    def traceProduce_mul(self,traceNum):
        '''
        产生极坐标下的航迹信息-duo
        :returm:
        '''
        #产生初始位置信息
        # locOriginal=torch.from_numpy(np.random.uniform(low=[0,0],high=[1,2*np.pi],
        # size=(self.traceNum,self.dimNum)))
        while True:
            locOriginal = torch.from_numpy(
                np.random.uniform(low=[0, 0], high=[1, 1], size=(traceNum, self.dimNum)))

            loc_new=RecToPolar(locOriginal)#3*2
            locOriginal=loc_new
            vOriginal=torch.from_numpy(
                np.random.uniform(low=[self.vlim[0],0],
                                  high=[self.vlim[1],2*np.pi],
                                  size=(traceNum,self.dimNum)
                                  ))

            accOriginal = torch.from_numpy(
                np.random.uniform(low=[self.alim[0],0],
                                  high=[self.alim[1],2*np.pi],
                                  size=(traceNum,self.dimNum)
                                  ))

            for iterm in range(1,self.figureTime,1):
                # print(iterm)
                vOriginal[:,0] = vOriginal[:,0].clamp(self.vlim[0], self.vlim[1])
                loc_new=self.vecOp.polAdd(pol1=loc_new,pol2=vOriginal)
                locOriginal=torch.cat((locOriginal,loc_new),0)
                vOriginal=self.vecOp.polAdd(vOriginal,accOriginal)
                accOriginal = torch.from_numpy(
                    np.random.uniform(
                        low=[self.alim[0], 0],
                        high=[self.alim[1], 2 * np.pi],
                        size=(traceNum, self.dimNum)))
            if(torch.max(locOriginal[:,0:1])<=1):
                break

        time=torch.ones(size=(traceNum*self.figureTime,1),dtype=locOriginal.dtype)

        for i in range(self.figureTime):
            time[i*traceNum:(i+1)*traceNum,:]=\
                i*time[i*traceNum:(i+1)*traceNum,:]
        locOriginal=torch.cat((locOriginal,time),dim=1)

        return locOriginal

    def noseProdece_mul(self,noseNumber):
        import random
        noseSave=np.zeros(shape=(0,3))
        noseBase=np.random.uniform(
            low=0,
            high=1,
            size=(noseNumber,2))
        for i in range(self.figureTime):
            noseBank=np.repeat(noseBase,repeats=10,axis=0)
            choseRate=np.random.uniform(low=0.4,high=0.6,size=(1))
            choseDir=random.sample(range(0,noseBank.shape[0],1),int(choseRate*noseBank.shape[0]))
            noseBank=np.random.normal(noseBank,scale=self.noseScale)
            noseBank=noseBank[choseDir,:]
            time=i*np.ones(shape=(noseBank.shape[0],1))
            data=np.concatenate((noseBank,time),axis=1)
            noseSave=np.concatenate((noseSave,data),axis=0)
        #挑选在0-1之间的数据
        dir=np.where(noseSave[:, 0]>0)[0]
        noseSave=noseSave[dir,:]
        dir = np.where(noseSave[:, 1] > 0)[0]
        noseSave = noseSave[dir, :]
        dir = np.where(noseSave[:, 0] < 1)[0]
        noseSave = noseSave[dir, :]
        dir = np.where(noseSave[:, 1] < 1)[0]
        noseSave = noseSave[dir, :]


        noseSave=torch.tensor(data=torch.from_numpy(noseSave),dtype=self.faultDtype)
        noseSave[:,0:2]=RecToPolar(noseSave[:,0:2])
        return noseSave



    def saveDataFuc(self,txtName,openModel='a',randomTraceNum=True,traceNum=2):
        import os
        if not os.path.exists(self.dataName,):
            os.makedirs(self.dataName)
        if(not os.path.exists(self.dataName+txtName)):
            fileDir = open(self.dataName + txtName, 'w')
        else:
            fileDir = open(self.dataName+txtName,openModel)

    def traceProduce(self):
        '''
        产生极坐标下的航迹信息
        :returm:
        '''
        #产生初始位置信息
        locOriginal=torch.from_numpy(np.random.uniform(low=[0,0],high=[1,2*np.pi],size=(self.traceNum,self.dimNum)))
        loc_new=locOriginal
        #产生速度和加速度信息
        vOriginal=torch.from_numpy(np.random.uniform(low=[self.vlim[0],0],high=[self.vlim[1],2*np.pi],size=(self.traceNum,self.dimNum)))
        accOriginal = torch.from_numpy(np.random.uniform(low=[self.alim[0],0],high=[self.alim[1],2*np.pi],size=(self.traceNum,self.dimNum)))

        for iterm in range(1,self.figureTime,1):
            vOriginal[:, 0:1] = vOriginal[:, 0:1].clamp(self.vlim[0], self.vlim[1])
            loc_new=self.vecOp.polAdd(pol1=loc_new,pol2=vOriginal)
            locOriginal=torch.cat((locOriginal,loc_new),0)
            vOriginal=self.vecOp.polAdd(vOriginal,accOriginal)
            accOriginal = torch.from_numpy(np.random.uniform(low=[self.alim[0], 0], high=[self.alim[1], 2 * np.pi], size=(self.traceNum, self.dimNum)))
        return locOriginal

    def pltFucRectangle(self,data,setLim=True,scale=4):
        # data=np.random.uniform(1,10,(10,2))
        left, width = 0.05, 0.90
        bottom, height = 0.05, 0.90
        rect_scatter = [left, bottom, width, height]
        plt.figure(figsize=(10, 10))
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_scatter.scatter(data[:,0], data[:,1],s=scale)
        if setLim:
            ax_scatter.set_xlim((-1, 1))
            ax_scatter.set_ylim((-1, 1))
        plt.show()

    def pltPolar(self,data,colors='red',area=5,cmap='hsv',alpha=0.75,setRlim=True):
        # data=torch.tensor(data,dtype=torch.float)
        # data = np.random.uniform(1, 10, (10, 2))
        colors=np.repeat(20*np.ones(shape=(1)),data.shape[0],axis=0)
        fig = plt.figure(figsize=(10,10),dpi=200)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('E')
        ax.set_thetagrids(np.arange(0.0, 360.0, 180.0))
        if setRlim:
            ax.set_rgrids(np.arange(0.0, 1.5, 0.3))
        if setRlim:
            ax.set_rlim(0, 1.5)
        ax.scatter(data[:,1], data[:,0],c=colors, s=area, cmap=cmap, alpha=alpha)
        plt.show()
    def valueTracePro(self,num):
        traceArray=torch.zeros(size=(num,self.figureTime,3),dtype=self.faultDtype)
        num1=num
        while(num>0):
            if(num%50==0):
                print(1-num/num1)
            num-=1
            while True:
                array=self.traceProduce()
                array=self.vecOp.pol2car(array)
                min,_=array.min(dim=0)
                max, _ = array.max(dim=0)
                flag1=(max<torch.ones(size=(1,2),dtype=array.dtype))
                flag2=(min > torch.zeros(size=(1, 2), dtype=array.dtype))
                flag=torch.tensor((flag1+flag2).sum(),dtype=min.dtype)
                if(flag==4):
                    break
            # print(array.size())
            array=torch.cat((array, torch.tensor(range(self.figureTime),dtype=array.dtype).resize_(self.figureTime,1) ),dim=1)

            traceArray[num:num+1,:,:]=array
        return traceArray

    def arrayToImage(self,array):

        arrarNumpy=array.numpy()
        meetConditionDir=np.where(arrarNumpy<0)
        arrarNumpy=np.delete(arrarNumpy, meetConditionDir[0], axis=0)
        meetConditionDir=np.where(arrarNumpy[:,0:2]>=1)
        arrarNumpy=np.delete(arrarNumpy, meetConditionDir[0], axis=0)
        array=torch.from_numpy(arrarNumpy)

        arrayImage=array
        # print('array[:,2].max()',array[:,2].max())
        arrayImage[:,0:2]=array[:,0:2]*self.figureWidth
        image=torch.zeros(size=(self.figureWidth,self.figureHidth,self.figureTime),dtype=array.dtype)
        imageTraceDir=torch.floor(arrayImage).long()
        image[imageTraceDir[:,0],imageTraceDir[:,1],imageTraceDir[:,2]]=255
        return image
    def ImageToArray(self,image):

        dix=np.where(image>0)
        dix=np.array(dix,dtype=float).transpose()

        dix[:,0:1]=dix[:,0:1] /self.figureWidth
        dix[:,1:2]=dix[:,1:2] /self.figureHidth
        return dix

    def arrayTranse(self,array):
        return (self.ImageToArray(self.arrayToImage(array)))

    def injectionNoise(self,numOfNosePoint,traceArray,txtName,openModel='a'):
        import  os
        if not os.path.exists(self.dataName):
            os.makedirs(self.dataName)
        if(not os.path.exists(self.dataName+txtName)):
            fileDir = open(self.dataName + txtName, 'w')
        else:
            fileDir = open(self.dataName+txtName,openModel)

        for j in range(traceArray.size()[0]):
            noseArrayOrSave=np.zeros(shape=(0,3))
            noseArrayOr=np.random.uniform(low=[0,0],high=[1,1],size=(numOfNosePoint,2))#[numOfNosePoint 2]
            noseArrayOr=np.repeat(noseArrayOr,10,axis=0)#[numOfNosePoint*20 2]
            for i in range(self.figureTime):
                noseDir = random.sample(range(noseArrayOr.shape[0]), int(0.1 * noseArrayOr.shape[0]))
                noseArrayOrnext=np.random.normal(loc=noseArrayOr[noseDir,:],scale=self.noseScale)
                noseArrayOrnext=np.concatenate((noseArrayOrnext,i*np.ones(shape=(noseArrayOrnext.shape[0],1))),axis=1)
                noseArrayOrSave=np.concatenate((noseArrayOrSave,noseArrayOrnext),axis=0)
            noseArrayOrSave=torch.tensor(noseArrayOrSave,dtype=traceArray.dtype)
            noseArrayOrSave=torch.cat((noseArrayOrSave,traceArray[j]),dim=0)

            ct = time.time()
            trainDataName = "%s_%3d" % (time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(ct)), (ct - int(ct)) * 1000)
            fileNameTrain=str(trainDataName)+'.pt'

            noseArrayOrSave=self.arrayTranse(noseArrayOrSave)

            torch.save(noseArrayOrSave,f=self.trainSavePath+fileNameTrain)
            torch.save(traceArray[j], f=self.labelSavePath + fileNameTrain)
            fileDir.write(fileNameTrain+'\n')

        fileDir.close()
        return noseArrayOrSave

    def xmlWite(self,folder,filename,source,size,):
        from lxml import etree, objectify
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('VOC2014_instance'),
            E.filename("test.jpg"),
            E.source(
                E.database('COCO'),
                E.annotation('COCO'),
                E.image('COCO'),
                E.url("http://test.jpg")
            ),
            E.size(
                E.width(800),
                E.height(600),
                E.depth(3)
            ),
            E.segmented(0),
        )
        etree.ElementTree(anno_tree).write("text.xml", pretty_print=True)

if __name__=="__main__":
    a=traceImageProduce(1,sizeFig=256,time=5,
                        trainSavePath='../../data/Images/',
                        labelSavePaht='../../data/Annatations/trace/',
                        dataNameSavePath='../../data/Main/')
    print(a.traceProduce_mul(7).size())
    print(a.noseProdece_mul(50).size())
    a.pltPolar(a.traceProduce_mul(7)[:,0:2])









