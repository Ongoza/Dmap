import numpy as np
import cv2
import geopy.distance

file_srt = "DJI_0006.SRT"
file_map = "50.65796820554269_30.392857254847865.jpg"
points=(50.6579682, 30.39285725, 50.649708, 30.409688)

srt_xy = []
srt_gps = []
len_frames = 0
v_scale = (1280, 720)
v_scale_out = (int(v_scale[0]/2),int(v_scale[1]/2))
#start_loc = np.array([.0, .0])
img_map = cv2.imread(file_map, cv2.IMREAD_COLOR)
img_sep = np.zeros((int(v_scale[1]/2), 10, 3), np.uint8)
map_hw = [img_map.shape[1], img_map.shape[0]]
print("img hw", map_hw, v_scale_out)
#start_loc = []
cur_loc = []
start_angle = 0

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

def drawKeyPts(im, keyp, size, col):
    for curKey in keyp:
        #print("curKey:",curKey.pt,type(curKey.pt))
        x=int(curKey.pt[0])
        y=int(curKey.pt[1])
        cv2.circle(im, (x,y), size, col, -1)

def angle_p(p1, p2):
    tr = True
    res_d = np.array(p2) - np.array(p1)
    #length = np.sqrt(res_d.dot(res_d))
    length = np.linalg.norm(res_d)
    if not length:
        tr = False
        length = 0.00001
    angle = np.arccos(res_d[1]/length)
    if res_d[0]<0: angle *= -1 
    #print("length", length, np.degrees(angle), angle, res_d[1]) 
    #cv2.putText(img, f"Angle (deg): {np.degrees(angle):.0f}", (400,100), 6, 1,  (255,255,255), 2)
    return tr, angle

def scale_to_img(lat_lon):
    """
    Conversion from latitude and longitude to the image pixels.
    It is used for drawing the GPS records on the map image.
    :param lat_lon: GPS record to draw (lat1, lon1).
    :param h_w: Size of the map image (w, h).
    :return: Tuple containing x and y coordinates to draw on map image.
    """
    # https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445
    old = (points[2], points[0])
    new = (0, map_hw[1])
    y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
    old = (points[1], points[3])
    new = (0, map_hw[0])
    x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
    # y must be reversed because the orientation of the image in the matplotlib.
    # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
    return [int(x), map_hw[1] - int(y)]

start_angle = -100
with open(file_srt) as file:
    lines = file.readlines()    
    #for i in range (4,100,6):
    for i in range (4, len(lines), 6):
        line = lines[i][:-27].replace("]","").split("[")        
        srt_gps.append(np.array([float(line[8][10:]),float(line[9][11:])]))
        x_y = scale_to_img(srt_gps[-1])
        srt_xy.append(x_y)
        if len(srt_xy) > 2 and start_angle == -100: 
            tr, angle = angle_p(srt_xy[-2],srt_xy[-1])
            if tr: start_angle = angle
        len_frames += 1
print("start_angle", start_angle)
start_angle = srt_xy[0]
#start_loc = srt_xy[0]
cur_loc = srt_xy[0].copy()
srt_gps = np.array(srt_gps)
srt_xy = np.array(srt_xy)
print("Srt open is ok ",srt_xy[0], cur_loc, geopy.distance.geodesic(srt_gps[0][:2], srt_gps[-1][:2]).m)
#print("srt_xy",srt_xy)
cap = cv2.VideoCapture("DJI_0006_s.MP4")
video_fh_fw =  (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#out = cv2.VideoWriter("out_1.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, v_scale)
out = cv2.VideoWriter("out_1.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (v_scale_out[0]*2+3*10,v_scale_out[1]))
#from matplotlib import pyplot as plt
orb = cv2.ORB_create()
kp1 = None
drow = True
top_feat = 16
#route = [start_loc]
#route_xy = [start_loc]
res_f = np.empty((top_feat, 2))
res_d = np.empty((top_feat, 2))
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
cnt = 0
counter_start = 4
video_to_map_scale = np.array([14.,17])
counter = counter_start
while cnt < len_frames:
    r, frame = cap.read()
    if r:
        #print("cnt", cnt)
        counter -= 1
        img = cv2.resize(frame, v_scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not kp1:
            kp1 = orb.detect(img,None)
            kp1, des1 = orb.compute(img, kp1)
            img = cv2.drawKeypoints(img, kp1, None, color=(0,255,0), flags=0)
            img1 = cv2.resize(img, v_scale_out)
            img2 = cv2.resize(img_map, v_scale_out)
            print("rs", img_sep.shape, img1.shape, img2.shape)
            Hori = np.concatenate((img_sep, img1, img_sep, img2, img_sep), axis=1)
            cv2.imshow('video', Hori)
            cv2.moveWindow('video', 30, 30)
            continue
        else:         
            if not counter:
                counter = counter_start
                kp2 = orb.detect(img,None)
                kp2, des2 = orb.compute(img, kp2)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key = lambda x:x.distance)[:top_feat]
                #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                for i, match in enumerate(matches):
                        p1 = kp1[match.queryIdx]
                        p2 = kp2[match.trainIdx]                        
                        res_d[i] = np.array(p2.pt) - np.array(p1.pt)
                        #print("x", l_delta, p2.pt, p1.pt)     
                        res_f[i][0] = np.sqrt(res_d[i].dot(res_d[i])) # length
                        angl = p2.angle - p1.angle
                        res_f[i][1] = angl + 360 if angl < -180 else angl # angle
                        #print("x2", i,res_f[i], res_d[i])                         
                avg_delta = res_f.mean(axis=0).astype(int)
                loc_delta = res_d.mean(axis=0)/video_to_map_scale#*np.array([1.,1.])
                cur_loc_new = np.add(cur_loc, loc_delta)
                tr1, angle_v = angle_p(cur_loc, cur_loc_new)
                #print(f"angle {np.degrees(angle_v)} {np.add(cur_loc, loc_delta)} {cur_loc}")
                #tr_0, loc_temp = rotate(loc, angle, rec_temp, limit) 
                #route.append(cur_loc.copy())
                #route_xy.append(srt_xy[cnt].astype(int))                
                loc_xy = srt_xy[cnt].astype(int)
                tr2, angle_xy = angle_p(srt_xy[cnt-counter_start], srt_xy[cnt])
                print(f"{cnt} cur {cur_loc} {cur_loc_new} {loc_delta} {np.subtract(srt_xy[cnt], srt_xy[cnt-counter_start])} a {np.degrees(angle_v):.0f} a_xy {np.degrees(angle_xy):.0f}")
                cur_loc = cur_loc_new
                #print(f"{cnt}  {loc_delta} cur_loc {cur_loc} loc_xy {loc_xy} xy_delta:{srt_xy[cnt]-srt_xy[cnt-1]}")     
                if drow: 
                    img = cv2.drawKeypoints(img, kp2, None, color=(0,255,0), flags=0)
                    #print(f"route {np.array(route).shape}")
                    #cv2.polylines(img_map, [np.array(route)], False, (155,155,0), 2)
                    #print("poly", cnt, route_xy[-1])                    
                    #cv2.polylines(img_map, [np.array(route_xy)], False, (255,255,0), 3)
                    cv2.circle(img_map, loc_xy, 4, (0,255,0), -1)
                    cv2.circle(img_map, cur_loc.astype(int), 3, (0,0,255), -1)
                drawKeyPts(img, kp2, 2, (255,0,0))
                #img = cv2.drawKeypoints(img, kp2, None, color=(255,0,0), flags=0)
                #cv2.imshow('video', img)
                img1 = cv2.resize(img, v_scale_out)
                img2 = cv2.resize(img_map, v_scale_out)
                #print("rs", img_sep.shape, img1.shape, img2.shape)
                Hori = np.concatenate((img_sep, img1, img_sep, img2, img_sep), axis=1)
                cv2.imshow('video', Hori)
                out.write(Hori)
                kp1 = tuple(kp2) # tuple
                des1 = des2.copy() #np array
        cnt += 1   
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): break

cap.release()    
if out: out.release()
cv2.destroyAllWindows()
