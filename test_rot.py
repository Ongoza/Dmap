import cv2
import numpy as np

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
height = 300
width = 300
x_r = 100
y_r = 50
N = 100 # y
M = N
__xy = np.array([[x_r,y_r],[x_r+M,y_r],[x_r+M,y_r+N],[x_r,y_r+N]])

img1 = np.zeros((height,width,3), np.uint8)
#print(__xy)
cv2.polylines(img1, [__xy], True, (255, 255, 255), 2)
__xyA = np.array([[x_r+int(M/2),y_r+int(N/3)],[x_r+int(M/2),y_r+int(2*N/3)]])
cv2.arrowedLine(img1, __xyA[0], __xyA[1], (0,0,255), 2)

img2 = np.zeros((height,width,3), np.uint8)
__xy2 = rotate(__xy, 30)
cv2.polylines(img2, [__xy2], True, (255, 255, 255), 2)
__xyA2 = rotate(__xyA, 30)
cv2.arrowedLine(img2, __xyA2[0], __xyA2[1], (0,0,255), 2)

img3 = np.zeros((height,width,3), np.uint8)
__xy3 = rotate(__xy, 60)
cv2.polylines(img3, [__xy3], True, (255, 255, 255), 2)
__xyA3 = rotate(__xyA, 60)
cv2.arrowedLine(img3, __xyA3[0], __xyA3[1], (0,0,255), 2)

angle4 = 360
img4 = np.zeros((height,width,3), np.uint8)
__xy4 = rotate(__xy, angle4)
cv2.polylines(img4, [__xy4], True, (255, 255, 255), 2)
__xyA4 = rotate(__xyA, angle4)
cv2.arrowedLine(img4, __xyA4[0], __xyA4[1], (0,0,255), 2)

Hori = np.concatenate((img1, img2, img3, img4), axis=1)

cv2.imshow('sample', Hori)
cv2.moveWindow('sample', 30, 30)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image