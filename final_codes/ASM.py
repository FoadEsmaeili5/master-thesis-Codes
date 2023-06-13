import rpy2.robjects as robj
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

Rshapes = importr("shapes")



import numpy as np

import matplotlib.pyplot as plt

#---------------------------
from final_codes.saving_images import image_frame
from final_codes.landmark_extractor import landmarks



#--------------------------------
# import os
# mydir = "H:\\data\\LVQuan19\\TrainingData_LVQuan19"
# files = os.listdir(mydir)
# files = np.array([mydir + "\\" + i for i in files])
# files = files[[i.endswith(".mat") for i in files]]


# #-----------------------------------------------------
# df_image = image_frame(0,files,part = "image")
# df_epi = image_frame(0,files,part = "epi")
# df_endo = image_frame(0,files,part="endo")

def ASM(frame):
    k= int(frame.shape[0]/2)
    joinline = np.append(np.arange(k+1),1)
    joinline = np.append(joinline,np.nan)
    joinline = np.append(joinline,np.append(np.arange(start=k+1,stop = 2*k+1),k+1))
    #-----------------------
    proc = Rshapes.procGPA(frame,scale = False)
    return proc,joinline


# proc1,joinline = ASM(np.concatenate((df_epi,df_endo)))

# Rshapes.shapepca(proc1,joinline =  joinline,type = "r",pcno = np.array([1,2,3,4]))
# Rshapes.shapepca(proc1,joinline =  joinline,type = "v",pcno = np.array([1,2,3,4]))
# Rshapes.shapepca(proc1,joinline =  joinline,type = "s",pcno = np.array([1,2,3,4]))
# Rshapes.shapepca(proc1,joinline =  joinline,type = "g",pcno = np.array([1,2,3,4]))