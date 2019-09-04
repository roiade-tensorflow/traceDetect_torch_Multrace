import torch
import numpy as np

a=torch.from_numpy(np.random.uniform(0,2,(3,3)))
b=a.clamp(0,1)
print(b)