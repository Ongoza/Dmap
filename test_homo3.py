import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('Melitopol_1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Melitopol_3.jpg', cv2.IMREAD_GRAYSCALE) 

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)
# draw only keypoints location, not size and orientation
img_12 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img_22 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
#plt.imshow(img_22), plt.show()
if False:
    Hori = np.concatenate((img_12, img_12), axis=1)
    cv2.imshow('sample', Hori)
    cv2.moveWindow('sample', 30, 30)
    cv2.waitKey(0) # waits until a key is pressed

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
print(5)
# Need to draw only good matches, so create a mask
# Sort them in the order of their distance.
top_feat = 16
matches = sorted(matches, key = lambda x:x.distance)[:top_feat]
# Draw first 10 matches.
print('matches', len(matches))
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# ratio test as per Lowe's paper

res_f = np.empty((top_feat,2))
#res_angl = np.array((10))
if True:
    for i, match in enumerate(matches):
         p1 = kp1[match.queryIdx]
         p2 = kp2[match.trainIdx]
         x = np.array(p2.pt) - np.array(p1.pt)
         mag = np.sqrt(x.dot(x))
         angl = p2.angle - p1.angle
         #print("m", i, mag, angl)     
         res_f[i][0] = mag
         res_f[i][1] = angl + 360 if angl < -180 else angl
avg_delta = res_f.mean(axis=0)
print(f"avg dist {avg_delta[0]} and angle {avg_delta[1]}")     
#draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)

#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#plt.imshow(img3,),plt.show()

cv2.imshow('sample', img3)
cv2.moveWindow('sample', 30, 30)
cv2.waitKey(0) # waits until a key is pressed
