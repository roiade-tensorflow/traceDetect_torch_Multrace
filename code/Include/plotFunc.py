import matplotlib.pyplot as plt
import numpy as np

def imageShow(image,scale=2):
    data=np.array(image)
    plt.xlim(0,image.shape[0])
    plt.ylim(0,image.shape[1])
    dir=np.where(data>0)
    plt.scatter(dir[0],dir[1],s=scale)
    plt.show()
