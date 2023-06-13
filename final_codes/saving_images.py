import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os

from scipy.io import loadmat
# mydir = "H:\\data\\LVQuan19\\TrainingData_LVQuan19"
# files = os.listdir(mydir)

# files = np.array([mydir + "\\" + i for i in files])
# files = files[[i.endswith(".mat") for i in files]]

# loadmat(files[2])["image"].shape # there is 20 image frames per person

# loadmat('H:\\data\\LVQuan19\\TrainingData_LVQuan19\\patient43.mat')

# reading images per frame!
import cv2
# images-------------------------------
def image_frame(frame,files,size = (256,256),part = "image"):
    size = list(size)
    size.append(len(files))
    # print(size.shape)
    df_image = np.zeros(shape = size)
    
    for i in range(len(files)):
        f = loadmat(files[i])
        df_image[:,:,i] = cv2.resize(f[part][:,:,frame],dsize = size[:2])
    return df_image

# importing person!
def image_person(person,files,size = (256,256),part = "image"):
    f = loadmat(files[person])
    return cv2.resize(f[part],dsize= size)

# person0 = image_person(0,files,(256,256))

# #--------------------------------
# df_image = image_frame(0,files,part = "image")
# df_endo = image_frame(0,files,part = "endo")
# df_epi = image_frame(0,files,part = "epi")

# calculating mask
# X = df_epi - df_endo

# plotting for the 1st frame!
# plt.subplot(221)
# plt.imshow(df_image[:,:,0])
# plt.title("image")
# plt.subplot(222)
# plt.imshow(df_epi[:,:,0])
# plt.title("epi")
# plt.subplot(223)
# plt.imshow(df_endo[:,:,0])
# plt.title("endo")
# plt.subplot(224)
# plt.imshow(X[:,:,0])
# plt.title("mask")
# plt.imshow(X[:,:,0])
# plt.show()
