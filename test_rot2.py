import cv2
import numpy as np
import random

N = 112 # y
M = N
img1 =  cv2.imread("Melitopol.jpg")
imgheight=img1.shape[0]
imgwidth=img1.shape[1]

def rotateImage(image, angle):
    row, col = image.shape[:2]
    center=np.array([row,col])/2.0
    #center=tuple(np.array([row,col])/2.0)
    rot_mat = cv2.getRotationMatrix2D(center,angle+180,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate(points, angle):
    ANGLE = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [[ c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
           c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
          ] for px, py in points ]
    ).astype(int)

def cam_virt(img, N, M, arrow=False, draw=True):
    # generate random coordinates inside image
    count = 100
    __xy_test = False
    __x_test = False
    __y_test = False
    while (not np.all(__xy_test)) or (not np.all(__x_test)) or (not np.all(__y_test)):
        y_r = int(random.random()*imgheight*0.7)
        x_r = int(random.random()*imgwidth*0.7)
        angle = int(360*random.random())
        __xy = np.array([[x_r,y_r],[x_r+M,y_r],[x_r+M,y_r+N],[x_r,y_r+N]])    
        __xy2 = rotate(__xy, angle)
        __xy_test = np.greater(__xy2, 0)
        __x_test = np.less(__xy2[:,0], imgwidth)
        __y_test = np.less(__xy2[:,1], imgheight)    
        count -= 1
        if count == 0: return False, False
    # draw arrow for checking orientation
    if arrow: 
        __xyA = np.array([[x_r+int(M/2), y_r+int(N/6)],[x_r+int(M/2),y_r+int(5*N/6)]])
        __xyA2 = rotate(__xyA, angle)        
        cv2.arrowedLine(img, __xyA2[0], __xyA2[1], (0,0,255), 2)    
    # rotate the image to North
    rect = cv2.boundingRect(__xy2)
    x, y, w, h = rect
    img2_0 = img[y:y+h, x:x+w].copy()
    img2_r = rotateImage(img2_0, angle)
    # cut empty pixels
    center = (np.array(img2_r.shape[:2])/2).astype(int)
    x_min = max(center[0]-int(M/2), 0)
    y_min = max(center[1]-int(N/2), 0)
    img2 = img2_r[y_min:y_min+N, x_min:x_min+M]
    if draw: cv2.polylines(img, [__xy2], True, (255, 255, 255), 2)
    return True, img2

tr, img2 = cam_virt(img1, N, M, True)
if tr:    
    img1 = cv2.resize(img1,(int(imgwidth/2),int(imgheight/2)))
    img2_big = np.zeros((int(imgheight/2), int(imgwidth/2), 3), np.uint8)
    img2_big[2*N:N*2+2*N,60:M*2+60] = cv2.resize(img2,(N*2, M*2))
    Hori = np.concatenate((img1, img2_big), axis=1)
    cv2.imshow('sample', Hori)
    cv2.moveWindow('sample', 30, 30)
    cv2.waitKey(0) # waits until a key is pressed
else: print("Error generate virt camera image")
cv2.destroyAllWindows() # destroys the window showing image