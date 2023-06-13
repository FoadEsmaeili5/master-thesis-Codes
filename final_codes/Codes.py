import numpy as np
import matplotlib.pyplot as plt


from final_codes.saving_images import image_frame
from final_codes.plotShapes import plotShapes
from final_codes.landmark_extractor import landmarks


from final_codes.ASM import ASM
from final_codes.shape_simulation import simulator,distance,simulator2
from final_codes.TPS import thin_plate_spline_warp



from scipy.io import loadmat
import os
mydir = "H:\\data\\LVQuan19\\New Folder"
files = os.listdir(mydir)

files = np.array([mydir + "\\" + i for i in files])
files = files[[i.endswith(".mat") for i in files]]



#-----------------------------------------------------
files[0]
# importing images!
df_image = image_frame(0,files,part = "image")
df_epi = image_frame(0,files,part = "epi")
df_endo = image_frame(0,files,part="endo")
#-----------------------------------------------------
# finding landmarks

n_landmark = 20

epi_frame = np.zeros(shape = (n_landmark,2,df_epi.shape[2]))
epi_x_center = np.zeros(shape = (df_epi.shape[2]))
epi_y_center = np.zeros(shape = (df_epi.shape[2]))

endo_frame = np.zeros(shape = (n_landmark,2,df_endo.shape[2]))
endo_x_center = np.zeros(shape = (df_endo.shape[2]))
endo_y_center = np.zeros(shape = (df_endo.shape[2]))




for i in range(df_endo.shape[2]):
    epi = landmarks(df_epi[:,:,i],n=n_landmark)
    epi_frame[:,:,i] = epi.coord
    epi_x_center[i] = epi.x_center
    epi_y_center[i] = epi.y_center
    epi_frame[:,0,i] += epi.x_center
    epi_frame[:,1,i] += epi.y_center

    endo = landmarks(df_endo[:,:,i],n = n_landmark)
    endo_frame[:,:,i] = endo.coord
    endo_x_center[i] = endo.x_center
    endo_y_center[i] = endo.y_center
    endo_frame[:,0,i] += endo.x_center
    endo_frame[:,1,i] += endo.y_center


# plot for the first image
plt.imshow(df_image[:,:,0],cmap = "gray")
plotShapes(epi_frame[:,:,0],x_center=epi_x_center[0],y_center=epi_y_center[0])
plotShapes(endo_frame[:,:,0],x_center=endo_x_center[0],y_center=endo_y_center[0])
plt.show()

# Point distribution model

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

Rshapes = importr("shapes")

frame = np.concatenate((epi_frame,endo_frame))
proc,joinline = ASM(frame)
proc.names
Rshapes.shapepca(proc,joinline =  joinline,type = "s",pcno = np.array([1,2,3,4]))



#-----------------------------------------------------



#-----------------------------------------------------
# data augmentation
from scipy.spatial.distance import mahalanobis
mahalanobis(frame[:,:,0].reshape(-1),frame[:,:,1].reshape(-1),np.cov(frame[:,:,0],frame[:,:,1]))

my_sim = simulator2(proc)
mahalanobis(my_sim.reshape(-1),proc.rx2["mshape"].reshape(-1),np.cov(my_sim,proc.rx2["mshape"]))
distance(my_sim,proc.rx2["mshape"])
distance(my_sim,proc.rx2["rotated"][:,:,1])
distance(my_sim,frame[:,:,1])


def create_simulation(proc,frame):
    d = np.zeros(frame.shape[2])
    while True:
        my_sim = simulator2(proc)
        for i in range(df_endo.shape[2]):
            # d[i] = mahalanobis(my_sim.reshape(-1),frame[:,:,i].reshape(-1),cov)
            d[i] = distance(my_sim,frame[:,:,i])
        
        if d.min()<100:
            return my_sim,d


df_endo.shape



my_sim,d = create_simulation(proc,frame)
idx = np.where(d == d.min())[0][0]

k = int(my_sim.shape[0]/2)
img2 = thin_plate_spline_warp(df_image[:,:,idx],frame[:,:,idx],my_sim)
img2 = img2/img2.max()

mask = df_epi - df_endo
epi2 = thin_plate_spline_warp(df_epi[:,:,idx],frame[:,:,idx][:k,:],my_sim[:k,:])
endo2 = thin_plate_spline_warp(df_endo[:,:,idx],frame[:,:,idx][k:,:],my_sim[k:,:])

mask2 = thin_plate_spline_warp(mask[:,:,idx],frame[:,:,idx],my_sim)

plt.subplot(121)
plt.imshow(img2,cmap = "gray")
plt.contour(mask2,colors = "red",alpha = 0.1,camp = "gray")
plt.title("simulated")
plt.subplot(122)
plt.imshow(df_image[:,:,idx],cmap= "gray")
plotShapes(frame[:,:,idx][:k,:],x_center = epi_x_center[idx],y_center=epi_y_center[idx])
plotShapes(frame[:,:,idx][k:,:],x_center = epi_x_center[idx],y_center=epi_y_center[idx])
plt.title("original: " + str(idx))
plt.show()
#-------------------

# plt.subplot(121)
# plt.imshow(img2,cmap = "gray")
# plotShapes(my_sim[:k,:],x_center=epi_x_center[idx],y_center = epi_y_center[idx])
# plotShapes(my_sim[k:,:],x_center=epi_x_center[idx],y_center = epi_y_center[idx])
# plt.title("simulated")
# plt.subplot(122)
# plt.imshow(df_image[:,:,idx],cmap= "gray")
# plotShapes(frame[:,:,idx][:k,:],x_center = epi_x_center[idx],y_center=epi_y_center[idx])
# plotShapes(frame[:,:,idx][k:,:],x_center = epi_x_center[idx],y_center=epi_y_center[idx])
# plt.title("original: " + str(idx))
# plt.show()


