import rpy2.robjects as robj
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# from scipy.io import loadmat
# mydir = "H:\\data\\LVQuan19\\TrainingData_LVQuan19"
# files = os.listdir(mydir)

# files = np.array([mydir + "\\" + i for i in files])
# files = files[[i.endswith(".mat") for i in files]]
#---------------------------------------------------

# myCodes.!
from final_codes.saving_images import image_frame
from final_codes.plotShapes import plotShapes


landmark_extractor = robj.r("""
    function(img,nmark = 18) {
        img = t(img)
        x <- contourLines(img,nlevels = 1)
        x <- do.call(cbind,x[[1]])
        
        # regularRadius
        regularraduis = function(Rx,Ry,n){
            le = length(Rx)
            M <- matrix(c(Rx,Ry),le,2)
            M1 <- matrix(c(Rx - mean(Rx), Ry - mean(Ry)), le,2)
            V1 <- complex(real = M1[,1], imaginary = M1[,2])
            #z = 3 +2i
            #Arg(z)
            #atan(2/3)
            #Mod(z)
            #sqrt(2^2+3^2)
            M2 <- matrix(c(Arg(V1), Mod(V1)), le, 2)
            V2 <- NA
            
            for (i in 0:(n-1)) {
                V2[i+1] <- which.max((cos(M2[,1] - 2 * i * pi/n)))
            }
            V2 <- sort(V2)
            list("pixindex" = V2,"radii"= M2[V2,2],"coord" = M1[V2,],
                "x_center"=mean(x[,2]),"y_center" = mean(x[,3]))
        }
        reg_epi <- regularraduis(x[,2],x[,3],nmark)
        return(reg_epi)
        }
""")

# img = image_frame(0,np.array([files[0]]), part = "image")[:,:,0]
# endo = image_frame(0,np.array([files[0]]), part = "endo")[:,:,0]
# epi = image_frame(0,np.array([files[0]]), part = "epi")[:,:,0]


class landmarks:
    def __init__(self,img, n = 18):
        mark = landmark_extractor(img,nmark = n)
        self.coord = mark.rx2["coord"] * 255
        self.x_center = mark.rx2["x_center"] * 255
        self.y_center = mark.rx2["y_center"] * 255
    
    def plot(self,**kwargs):
        plotShapes(self.coord,self.x_center,self.y_center,**kwargs)



### using sobel for finding the contourLine:
# sobel x, y for finding out the outline of a contour!
#epi = epi[:,:,0]
#endo = endo[:,:,0]
# sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# Compute the gradient of the image in the x and y directions using the Sobel kernels
#grad_x = np.zeros_like(epi)
#grad_y = np.zeros_like(epi)

#for i in range(1, epi.shape[0]-1):
#    for j in range(1, epi.shape[1]-1):
#        grad_x[i, j] = np.sum(epi[i-1:i+2, j-1:j+2] * sobel_x)
#        grad_y[i, j] = np.sum(epi[i-1:i+2, j-1:j+2] * sobel_y)

#mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

# Display the resulting image
#plt.imshow(mag, cmap='gray')
#plt.show()
