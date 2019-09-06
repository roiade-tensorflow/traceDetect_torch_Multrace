import matplotlib.pyplot as plt
import numpy as np

def imageShow(image,scale=2):
    data=np.array(image)
    plt.xlim(0,image.shape[0])
    plt.ylim(0,image.shape[1])
    dir=np.where(data>0)
    plt.scatter(dir[0],dir[1],s=scale)
    plt.show()

def imageShow2(imageNose ,imageEval,imageLable,scale=2):

    dataNose=np.array(imageNose)
    dataEval=np.array(imageEval)
    dataLabel=np.array(imageLable)

    plt.figure(figsize=(10,10))

    plt.xlim(0,imageNose.shape[0])
    plt.ylim(0,imageNose.shape[1])

    dirNose=np.where(dataNose>0)
    dirEval=np.where(dataEval>0)
    dirLabel=np.where(dataLabel>0)



    axNose=plt.axes([0,0,0.5,0.5])
    axEval=plt.axes([0,0.5,0.5,0.5])
    axLabel=plt.axes([0.5,0.5,0.5,0.5])

    axNose.scatter(dirNose[0],dirNose[1],s=scale)
    axEval.scatter(dirEval[0],dirEval[1],s=scale)
    axLabel.scatter(dirLabel[0],dirLabel[1],s=scale)

    plt.show()
