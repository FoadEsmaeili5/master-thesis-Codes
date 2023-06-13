import rpy2.robjects as robj
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

Rshapes = importr("shapes")



import numpy as np
import os
# mydir = "H:\\data\\LVQuan19\\TrainingData_LVQuan19"
# files = os.listdir(mydir)
# files = np.array([mydir + "\\" + i for i in files])
# files = files[[i.endswith(".mat") for i in files]]


from final_codes.saving_images import image_frame
from final_codes.landmark_extractor import landmarks
from final_codes.plotShapes import plotShapes
from final_codes.ASM import ASM



# df_epi = image_frame(0,files,part = "epi")
# df_endo = image_frame(0,files,part="endo")


# proc1,joinline = ASM(df_epi,df_endo)
# proc1.names



def distance(a,b):
    dist = np.linalg.norm(a-b)
    return dist






def simulator(proc):
    x = np.random.uniform(-1.5,1.5,size = 1)
    res = np.matmul(proc.rx2["pcar"], + x * proc.rx2["pcasd"])
    k = int(len(res)/2)
    r = np.c_[res[:int(k)],res[int(k):]]
    simulated = proc.rx2["mshape"] + r
    return simulated


def simulator2(proc):
    mshape = proc.rx2["mshape"]
    rotated = proc.rx2["rotated"]
    k = mshape.shape[0]
    n = rotated.shape[2] # number of samples
    mshape2 = np.r_[mshape[:,0],mshape[:,1]]
    d = np.zeros((np.prod(mshape.shape),n))
    for i in range(n):
        xx = np.r_[rotated[:,0,i],rotated[:,1,i]]
        d[:,i] = xx - mshape2
    
    sigma = 1.0/56 * np.matmul(d,d.transpose())
    sample = np.random.multivariate_normal(mean = mshape2,cov = sigma)
    sample2 = np.c_[sample[:k],sample[k:]]
    return sample2




# sim = simulator(proc1)
# sim.shape

# plotShapes(sim, k =18,marker = "*", markersize = 5)

