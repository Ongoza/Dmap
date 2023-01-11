import cv2
import numpy as np
import random
import math

N = 112 # y
M = N
M2 = int(M/2)
N2 = int(N/2)
img1 =  cv2.imread("Melitopol.jpg")
imgheight=img1.shape[0]
imgwidth=img1.shape[1]


route = np.array([[100,100],[150,150],[200,175],[300,220],[360,250],[400,320],[360,400],[300,460],[294,524],[330,580],[418,622],[470,720],[530,800],[600,900],[660,970],[710,1030],[730,1100],[720,1212],[680,1260],[600,1300],[525,1360],[532,1430],[570,1460],[580,1520],[550,1600],[464,1640]])

def rotateImage(image, angle):    
    row, col = image.shape[:2]
    center=tuple(np.array([row,col])/2.0)
    rot_mat = cv2.getRotationMatrix2D(center,np.degrees(angle),1.0)    
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate(loc, angle, N2, M2):    
    rec = [[-N2,-M2],[N2,-M2],[N2,M2],[-N2,M2]]
    sinA = np.sin(angle)
    cosA = np.cos(angle)
    res = np.empty((4, 2), dtype=int)
    for i in range(4):
        res[i][0] = loc[0] + cosA * rec[i][0] - sinA * rec[i][1]
        res[i][1] = loc[1] + sinA * rec[i][0] + cosA * rec[i][1]
    return res

def cam_virt(img, N, M, loc, step, arrow=False, draw=True):
    #vector_2 = np.array([1,0])
    #vec_2 = vector_2 / np.linalg.norm(vector_2)
    vec_2 = np.array([1., 0.])
    vec_1 = loc / np.linalg.norm(loc)
    dot_ = np.dot(vec_1, vec_2)
    angle = np.arccos(dot_)    
    print('loc', math.degrees(angle), loc, vec_2)
    #__xy = np.array([[x_r,y_r],[x_r+M,y_r],[x_r+M,y_r+N],[x_r,y_r+N]]) 
    #__xy = np.array([loc+[-M2,-N2],loc+[M2,-N2],loc+[M2,N2],loc+[-M2,N2]])
    __xy2 = rotate(loc, angle, N2, M2)
    print(__xy2)
    __xy_test = np.greater(__xy2, 0)
    __x_test = np.less(__xy2[:,0], imgwidth)
    __y_test = np.less(__xy2[:,1], imgheight)    

    # rotate the image to North
    rect = cv2.boundingRect(__xy2)
    x, y, w, h = rect
    if arrow: 
        #__xyA = np.array([[x_r+int(M/2), y_r+int(N/6)],[x_r+int(M/2),y_r+int(5*N/6)]])
        __xyA = np.array([loc+[int(M/2), int(N/6)], loc+[int(M/2),int(5*N/6)]])
        __xyA2 = rotate(__xyA, angle)        
        cv2.arrowedLine(img, __xyA2[0], __xyA2[1], (0,0,255), 2) 
    img2 = img[y:y+h, x:x+w].copy()
    try:
        img2_r = rotateImage(img2, angle)
        # cut empty pixels
        center = (np.array(img2_r.shape[:2])/2).astype(int)
        x_min = max(center[0]-int(M/2), 0)
        y_min = max(center[1]-int(N/2), 0)
        img2 = img2_r[y_min:y_min+N, x_min:x_min+M]

        if draw: 
            cv2.polylines(img, [__xy2], True, (255, 255, 255), 2)
            cv2.putText(img, f"{i}", loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # draw arrow for checking orientation
   
    except: print("error")
    return True, img2

#tr, img2 = cam_virt(img1, N, M, route[0], True)
cv2.polylines(img1, [route], False, (255,0,0), 2)

for i, loc in enumerate(route):
    cv2.circle(img1, loc, 10, (0,200,0), -1)
    print(i)
    tr, img2 = cam_virt(img1, N, M, loc, True)
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