from scipy import ndimage
import numpy as np

def _U(x):
    _small = 1e-100
    return (x **2) * np.where(x<_small,0,np.log(x**2))

def _interpoint_distances(points):
    # matrix of the a-b(inter point distances)
    xd = np.subtract.outer(points[:,0],points[:,0])
    yd = np.subtract.outer(points[:,1],points[:,1])

    return np.sqrt(xd **2 + yd **2)


# _interpoint_distances(from_point)


def _make_L_matrix(points):
    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n,3))
    P[:,1:] = points # cbind(1,x)
    O = np.zeros((3,3))
    LU = np.c_[K,P]
    LD = np.c_[P.transpose(),O]
    L = np.r_[LU,LD]
    # L = np.asarray(np.bmat([[K,P],[P.transpose(),O]]))
    return L

# _make_L_matrix(from_point)

def _calculate_f(coeffs, points, x,y):
    w = coeffs[:-3]
    a1,ax,ay = coeffs[-3:] # tps coefficients
    summation = 0
    for wi,Pi in zip(w,points):
        summation += wi * _U(np.sqrt((x - Pi[0])**2 + (y - Pi[1])**2))
    return a1 + ax * x + ay * y + summation





def _make_warp(from_points,to_points, x_val,y_val):
    L = _make_L_matrix(from_points)
    V = np.resize(to_points,(len(to_points)+3,2))
    V[-3:,:] = 0
    coeffs = np.dot(np.linalg.pinv(L),V)

    x_warp = _calculate_f(coeffs[:,0],from_points, x_val,y_val)
    y_warp =  _calculate_f(coeffs[:,1],from_points, x_val,y_val)
    return [x_warp, y_warp]

# from_point = frame1[:,:,0]
# to_point = frame1[:,:,1]
# _make_warp(from_point,to_point,0,1)

# np.mgrid((-1,1))
#----------------------------------------




def _make_inverse_warp(from_points,to_points, output_region,approximate_grid):
    x_min, y_min, x_max, y_max = output_region

    if approximate_grid is None: approximate_grid = 1
    x_steps = (x_max - x_min) / approximate_grid
    y_steps = (y_max - y_min) / approximate_grid
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]

    # make the reverse transform warping from the to_points to the from_points, because we
    # do image interpolation in this reverse fashion
    transform = _make_warp(to_points, from_points, x, y)
    return transform

from scipy import ndimage
def warp_images(from_points, to_points, images, output_region, interpolation_order = 1, approximate_grid=1):
    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    return ndimage.map_coordinates(images,transform,order = interpolation_order, mode = "reflect")


def thin_plate_spline_warp(image,src_points,dst_points,keep_corners = True):
    width, height = image.shape
    if keep_corners:
        corner_points = np.array(
            [[0,0],[0,width],[height,0],[height,width]]
        )
        src_points = np.concatenate((src_points,corner_points))
        dst_points = np.concatenate((dst_points,corner_points))
    
    out = warp_images(src_points, dst_points,image,(0,0,width,height))
    return np.asarray(out)



#----------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("H:/data/cats/images/images/Abyssinian_1.jpg",)
image.shape
image = cv2.resize(image[:,:,0],dsize = (256,256))
image.shape
from_point= np.array([[100,120],[160,200]])
to_point = np.array([[100,20],[100,200]])
# to_point
plt.subplot(121)
plt.imshow(image, cmap = "gray")
plt.scatter(from_point[:,0],from_point[:,1],label = "from",color = "blue")
plt.annotate("1",(100,120),color = "blue",size = 20)
plt.annotate("2",(160,200),color= "blue",size = 20)
plt.scatter(to_point[:,0],to_point[:,1],color= "red",label = "target")
plt.annotate("1",(100,20),color="red",size = 20)
plt.annotate("2",(100,200),color= "red",size = 20)
plt.legend(["from","target"])
plt.subplot(122)
plt.imshow(thin_plate_spline_warp(image,from_point,to_point),cmap = "gray")
plt.legend(loc = "upper right")
plt.show()



image2 = cv2.GaussianBlur(image,(3,3),1)
plt.subplot(121)
plt.imshow(image,cmap = "gray")
plt.title("original image")
plt.subplot(122)
plt.imshow(image2,cmap = "gray")
plt.title("image with gaussian noise")
plt.show()