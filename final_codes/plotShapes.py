import matplotlib.pyplot as plt
import numpy as np


#mat = proc1.rx2["mshape"]

def plotShapes(mat,x_center = 0,y_center = 0,k = None,**kwargs):
    if k == None:
        coord = np.vstack([mat,mat[0,:]])
        plt.plot(coord[:,0]+ x_center,coord[:,1]+ y_center,**kwargs)
    else:
        mat1 = mat[:k,:]
        mat2 = mat[k:,:]
        coord1 = np.vstack([mat1,mat1[0,:]])
        coord2 = np.vstack([mat2,mat2[0,:]])
        plt.plot(coord1[:,0]+ x_center,coord1[:,1]+ y_center,**kwargs)
        plt.plot(coord2[:,0]+ x_center,coord2[:,1]+ y_center,**kwargs)


# plotShapes(mat,k= int(len(mat)/2))
# plt.show()

