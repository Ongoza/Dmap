import numpy as np
import cv2

#video = "2023_01_14_11_08_19.mp4"
video = "test2.avi"

#video = "DJI_0006_s.MP4"
v_scale = (1280, 720)
out = cv2.VideoWriter("out_1.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, v_scale)
top_feat = 16
counter_start = 4
cur_loc = np.array([2000, 400])
scale_xy = 35
def angle_p(p1, p2): 
    res_d = np.array(p2) - np.array(p1)
    #length = np.sqrt(res_d.dot(res_d))
    length = max(np.linalg.norm(res_d), 0.00001)
    angle = np.arccos(res_d[1]/length)
    if res_d[0]<0: angle *= -1 
    print("length", length, np.degrees(angle), angle, res_d[1]) 
    cv2.putText(img, f"Angle (deg): {np.degrees(angle):.0f}", (400,100), 0, 1,  (255,255,255), 2)
    return angle

cap = cv2.VideoCapture(video)
video_fh_fw =  (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

#from matplotlib import pyplot as plt
orb = cv2.ORB_create()
kp1 = None
drow = True
route = [cur_loc]
res_f = np.empty((top_feat, 2))
res_d = np.empty((top_feat, 2))
#path = [cur_loc]
counter = counter_start
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
cnt = 0

while True:
    r, frame = cap.read()
    if r:
        counter -= 1
        cnt += 1
        if not counter:
            counter = counter_start
            img = cv2.resize(frame, v_scale)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
            if not kp1:
                kp1 = orb.detect(img,None)
                kp1, des1 = orb.compute(img, kp1)
                img = cv2.drawKeypoints(img, kp1, None, color=(0,255,0), flags=0)
                cv2.imshow('video', img)
                cv2.moveWindow('video', 30, 30)
                continue
            else:
                #print("counter", cnt )
                kp2 = orb.detect(img,None)
                kp2, des2 = orb.compute(img, kp2)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key = lambda x:x.distance)[:top_feat]
                #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                for i, match in enumerate(matches):
                     p1 = kp1[match.queryIdx]
                     p2 = kp2[match.trainIdx]                     
                     res_d[i] = np.array(p2.pt) - np.array(p1.pt)
                     #print("x", res_d[i], p2.pt, p1.pt)     
                     #mag = np.sqrt(l_delta.dot(l_delta))
                     #angl = p2.angle - p1.angle
                     # print("x2", i, mag, angl, l_delta)     
                     #res_f[i][0] = mag
                     #res_f[i][1] = angl + 360 if angl < -180 else angl
                #avg_delta = res_f.mean(axis=0).astype(int)
                loc_delta = res_d.mean(axis=0).astype(int)
                #print(f"cur_loc {cur_loc} and loc_delta {loc_delta}")     
                cur_loc = np.add(cur_loc, loc_delta)
                route.append(cur_loc.copy()) 
                angle_p(route[-2],route[-1])
                #print(f"loc {cur_loc} route {route}")     
                if drow: 
                    img = cv2.drawKeypoints(img, kp2, None, color=(0,255,0), flags=0)
                    #print(f"route {np.array(route).shape}")
                    cv2.polylines(img, [np.array(route)//scale_xy], False, (255,255,0), 3)
                    cv2.circle(img, route[-1]//scale_xy, 6, (0,0,255), -1)
                cv2.imshow('video', img)
                out.write(img)
                kp1 = tuple(kp2) # tuple
                des1 = des2.copy() #np array
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): break

cap.release()    
if out: out.release()
cv2.destroyAllWindows()
