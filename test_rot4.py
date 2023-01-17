import sys
import cv2
import numpy as np
import random
import math

step = 32
radius = 96
mask = np.zeros((radius*2,radius*2,3),np.uint8)
mask2 = np.zeros((radius,radius,3),np.uint8)
mask = cv2.circle(mask, (radius,radius), radius, (1,1,1), -1)
img1 =  cv2.imread("Melitopol.jpg")
img1_0 = img1.copy()
imgheight=img1.shape[0]
imgwidth=img1.shape[1]


route = np.array([[100,100],[370,250],[400,320],[360,400],[300,460],[294,524],[330,580],[418,622],[470,720],[530,800],[600,900],[660,970],[710,1030],[690,1100],[700,1212],[680,1260],[600,1300],[525,1360],[532,1430],[570,1460],[580,1520],[550,1600],[464,1640]])

def rotateImage(image, angle):    
    row, col = image.shape[:2]
    center=np.array([row,col])/2.0
    rot_mat = cv2.getRotationMatrix2D(center,np.degrees(angle)+90,1.0)    
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate(loc, angle, rec):    
    sinA = np.sin(angle)
    cosA = np.cos(angle)
    res = np.empty((len(rec), 2), dtype=int)
    for i, (resi, reci) in enumerate(zip(res, rec)):    
        resi[0] = max(loc[0] + cosA * reci[0] - sinA * reci[1], 0)
        resi[1] = max(loc[1] + sinA * reci[0] + cosA * reci[1], 0)
    return res

def cam_virt(loc, loc1, arrow=False, draw=True):
    out = []
    locs = []
    l = loc1-loc
    angle = - math.atan2(l[0],l[1])
    steps = math.floor(math.hypot(l[0], l[1])/step) - 1
    #print('angle', math.degrees(angle), steps, type(steps))
    img2 = img1_0[loc[1]-radius:loc[1]+radius, loc[0]-radius:loc[0]+radius].copy()
    try:
        img2 = img2*mask
        #print("img2", img2.shape)
        img2 = rotateImage(img2, angle)
        img2 = cv2.resize(img2, (radius,radius))
        if draw: 
            cv2.circle(img1, loc, radius, (255, 0, 255), 2)                        
            cv2.putText(img1, f"{i}", loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # draw arrow for checking orientation
        out.append(img2)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        #fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("error",exc_type, exc_tb.tb_lineno, loc)
        out.append(mask2)

        #cv2.imwrite('lena.jpg', img2)
    #if arrow: 

    return True, out

#tr, img2 = cam_virt(img1, N, M, route[0], True)
cv2.polylines(img1, [route], False, (255,0,0), 2)
route_imgs = []
for i in range(len(route)-1):
    cv2.circle(img1, route[i], 10, (0,200,0), -1)
    tr, img2 = cam_virt(route[i], route[i+1], True)
    if tr: route_imgs += img2
    else: print("Error generate virt camera image")    
if route_imgs:
    img1 = cv2.resize(img1,(int(imgwidth/2),int(imgheight/2)))
    rows = math.floor(int(imgheight/2)/(radius))
    columns = math.ceil(len(route_imgs)/rows)
    img2_big = np.zeros((int(imgheight/2), (radius)*columns, 3), np.uint8)
    col = 0 
    row = 0    
    for i, img in enumerate(route_imgs): 
        #print("shape", i, row, col, img.shape, row*radius,row*radius+radius, radius*col,radius*col+radius, img2_big.shape)
        img2_big[row*radius:row*radius+radius, radius*col:radius*col+radius] = img
        #img2_big[row*N:row*N+N, M*col:M*col+M] = img
        cv2.putText(img2_big, f"{i}", (radius*col,row*radius+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
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