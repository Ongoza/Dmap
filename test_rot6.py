import cv2
import numpy as np
import random
import math

N = 128 # x
M = N
radius = math.floor(N*math.sqrt(2)/2)
step = 96
M2 = int(M/2)
N2 = int(N/2)
mask2 = np.zeros((N,N,3),np.uint8)
img1 =  cv2.imread("Melitopol.jpg")
img1_0 = img1.copy()
imgheight=img1.shape[0]
imgwidth=img1.shape[1]
limit = [imgwidth,imgheight]
rec = [[-M2,-N2],[M2,-N2],[M2,N2],[-M2,N2]]

route = np.array([[100,100],[360,250],[400,320],[360,400],[300,460],[294,524],[330,580],[418,622],[470,720],[530,800],[600,900],[660,970],[710,1030],[730,1100],[710,1212],[680,1260],[600,1300],[525,1360],[532,1430],[570,1460],[580,1520],[550,1600],[464,1640]])

def rotateImage(image, angle):    
    row, col = image.shape[:2]
    center=tuple(np.array([row,col])/2.0)
    rot_mat = cv2.getRotationMatrix2D(center,np.degrees(angle)+90,1.0)    
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate(loc, angle, rec, limit):    
    sinA = np.sin(angle)
    cosA = np.cos(angle)
    res = np.empty((len(rec), 2), dtype=int)
    tr = True
    for i, (resi, reci) in enumerate(zip(res, rec)):
        resi[0] = loc[0] + cosA * reci[0] - sinA * reci[1]
        resi[1] = loc[1] + sinA * reci[0] + cosA * reci[1]
        if resi[0]<0 or resi[1]<0 or resi[0]>limit[0] or resi[0]>limit[1]: 
            tr = False
            break
    return tr, res

def cam_virt(loc, loc1, arrow=False, draw=True):
    imgs = []
    locs = []
    l = loc1-loc
    steps = math.floor(math.hypot(l[0], l[1])/step)+1
    angle = - math.atan2(l[0],l[1])
    #print("steps",steps)
    for i in range(steps):
        rec_temp = [[0,step*i]]
        tr_0, loc_temp = rotate(loc, angle, rec_temp, limit) 
        #l_test = loc - __xy2[0]
        print("loc_temp",loc_temp)
        #print(i, l_test, math.hypot(l_test[0],l_test[1]),math.hypot(l[0], l[1]))
        #if i:
        #    cv2.circle(img1, __xy2[0], 10, (250,200,200), -1)
        if tr_0:
            tr, img = cam_virt_shot(loc_temp[0], angle, arrow, draw)
            if tr: imgs.append(img)
    if not imgs:imgs.append(mask2)
    return imgs

def cam_virt_shot(loc, angle, arrow=False, draw=True):
    #print('angle', math.degrees(angle), steps, type(steps))
    tr, __xy2 = rotate(loc, angle, rec, limit)    
    # rotate the image to East
    if tr:
        x, y, w, h = cv2.boundingRect(__xy2)
        if draw: 
            cv2.circle(img1, loc, radius, (255, 0, 255), 2)
            cv2.polylines(img1, [__xy2], True, (255, 255, 255), 2)            
            cv2.putText(img1, f"{i}", loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if arrow: cv2.circle(img1, __xy2[0], 10, (0,200,200), -1)
        #cut rect outside circle
        img2 = img1_0[loc[1]-radius:loc[1]+radius, loc[0]-radius:loc[0]+radius].copy()
        st_sh = img2.shape
        try:
            img2 = rotateImage(img2, angle)
            # cut empty pixels
            center = (np.array(img2.shape[:2])/2).astype(int)
            x_min = max(center[0]-int(M/2), 0)
            y_min = max(center[1]-int(N/2), 0)
            #cut rect inside circle
            img2 = img2[y_min:y_min+M, x_min:x_min+N]
            img2 = cv2.resize(img2, (N,M))
            # draw arrow for checking orientation               
        except: 
            print("error")
            tr = False
    else:
        out = False
        print("Out if image size. Loc:", loc)
    return tr, img2

#tr, img2 = cam_virt(img1, N, M, route[0], True)
cv2.polylines(img1, [route], False, (255,0,0), 2)
route_imgs = []
for i in range(len(route)-1):
    cv2.circle(img1, route[i], 10, (0,200,0), -1)
    img2 = cam_virt(route[i], route[i+1], True)
    route_imgs += img2
    print("len img2",len(img2), np.array(route_imgs).shape)
if route_imgs:
    img1 = cv2.resize(img1,(int(imgwidth/2),int(imgheight/2)))
    rows = math.floor(int(imgheight/2)/N)
    columns = math.ceil(len(route_imgs)/rows)
    img2_big = np.zeros((int(imgheight/2), N*columns, 3), np.uint8)
    col = 0 
    row = 0    
    for i, img in enumerate(route_imgs): 
        #print("shape", i, row, col, img.shape, row*M,row*M+M, N*col,N*col+N, img2_big.shape)
        img2_big[row*M:row*M+M, N*col:N*col+N] = img
        #img2_big[row*N:row*N+N, M*col:M*col+M] = img
        cv2.putText(img2_big, f"{i}", (N*col+4,row*M+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        row += 1
        if row == rows: 
            col += 1
            row = 0
    img_sep = np.zeros((int(imgheight/2), 10, 3), np.uint8)
    Hori = np.concatenate((img_sep, img1, img_sep, img2_big, img_sep), axis=1)
    cv2.imshow('sample', Hori)
    cv2.moveWindow('sample', 30, 30)
    cv2.waitKey(0) # waits until a key is pressed

cv2.destroyAllWindows() # destroys the window showing image