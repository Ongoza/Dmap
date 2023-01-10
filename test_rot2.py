import sys
import cv2
import numpy as np
import random
import math

def rotate(points, angle):
    #https://stackoverflow.com/questions/69542049/how-to-rotate-rectangle-shape-cv2-python
    ANGLE = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
            ]
            for px, py in points
        ]
    ).astype(int)

N = 112 # y
M = N
img1 =  cv2.imread("Melitopol.jpg")
imgheight=img1.shape[0]
imgwidth=img1.shape[1]

def rand_rect():
    y_r = int(random.random()*imgheight*0.7)
    x_r = int(random.random()*imgwidth*0.7)
    angle = int(360*random.random())
    __xy = np.array([[x_r,y_r],[x_r+M,y_r],[x_r+M,y_r+N],[x_r,y_r+N]])    
    __xy2 = rotate(__xy, angle)
    return __xy2, x_r, y_r, angle

def rotateImage(image, angle):
    row,col = image.shape[:2]
    center=tuple(np.array([row,col])/2.0)
    #print("rot", row,col,"\n",np.array([row,col])/2.0)
    rot_mat = cv2.getRotationMatrix2D(center,angle+180,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

__xy2, x_r, y_r, angle = rand_rect()
__xy_test = np.greater(__xy2, 0)
__x_test = np.less(__xy2[:,0], imgwidth)
__y_test = np.less(__xy2[:,1], imgheight)

count = 100
while (not np.all(__xy_test)) or (not np.all(__x_test)) or (not np.all(__y_test)):
    __xy2, x_r, y_r, angle = rand_rect()
    __xy_test = np.greater(__xy2, 0)
    __x_test = np.less(__xy2[:,0], imgwidth)
    __y_test = np.less(__xy2[:,1], imgheight)    
    count -= 1
    if count == 0: 
        print("Error!!! Count=0")
        exit()
print("count",count, __xy2)
#print("__xy", imgwidth, imgheight, "\n",__xy2)
#center_0 = [x_r+int(M/2),y_r+int(N/2)]
#cv2.circle(img1, center_0, radius=5, color=(0, 0, 255), thickness=-1)


__xyA = np.array([[x_r+int(M/2),y_r+int(N/3)],[x_r+int(M/2),y_r+int(2*N/3)]])
__xyA2 = rotate(__xyA, angle)


print('angle', angle)

cv2.polylines(img1, [__xy2], True, (255, 255, 255), 2)
cv2.arrowedLine(img1, __xyA2[0], __xyA2[1], (0,0,255), 2)

img2 = np.zeros((M,N,3), np.uint8)

rect = cv2.boundingRect(__xy2)
x,y,w,h = rect
img2_0 = img1[y:y+h, x:x+w].copy()
img2_r = rotateImage(img2_0, angle)
#print("shape", img2_r[:2].shape)

center = (np.array(img2_r.shape[:2])/2).astype(int)
print("center", center)
x_min = max(center[0]-int(M/2), 0)
y_min = max(center[1]-int(N/2), 0)
img2 = img2_r[y_min:y_min+N, x_min:x_min+M]
img2 = cv2.resize(img2,(int(imgwidth/2),int(imgheight/2)))
img1 = cv2.resize(img1,(int(imgwidth/2),int(imgheight/2)))
Hori = np.concatenate((img1, img2), axis=1)

cv2.imshow('sample', Hori)
cv2.moveWindow('sample', 30, 30)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image