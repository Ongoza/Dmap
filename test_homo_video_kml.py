import numpy as np
import cv2
import geopy.distance
import pyproj
import xml.dom.minidom
import math

file_kml_src = "testP3.kml"
file_kml_out = "testP3_out.kml"
file_video_in = "test3.avi"
file_video_out = "test3_out.avi"
scale_img2gps = 0.4
gps_in = []
gps_out = []
len_frames = 0
v_scale = (1280, 720)
v_scale_out = (int(v_scale[0]/2),int(v_scale[1]/2))
#start_loc = np.array([.0, .0])
start_loc = []
cur_loc = []
start_angle = 0

wgs84_geod = pyproj.CRS('WGS 84').get_geod()

def createKML(fileName, locs):
    # This constructs the KML document from the CSV file.
    kmlDoc = xml.dom.minidom.Document()
    kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
    kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
    kmlElement = kmlDoc.appendChild(kmlElement)
    documentElement = kmlElement.appendChild(kmlDoc.createElement('Document'))
    name = documentElement.appendChild(kmlDoc.createElement('name'))
    name.appendChild(kmlDoc.createTextNode(fileName))    
    Placemark = documentElement.appendChild(kmlDoc.createElement('Placemark'))    
    nameP = Placemark.appendChild(kmlDoc.createElement('name'))
    nameP.appendChild(kmlDoc.createTextNode("P11"))
    Style = Placemark.appendChild(kmlDoc.createElement('Style'))    
    LineStyle = Style.appendChild(kmlDoc.createElement('LineStyle'))
    color = LineStyle.appendChild(kmlDoc.createElement('color'))
    color.appendChild(kmlDoc.createTextNode("7fff0000"))
    width = LineStyle.appendChild(kmlDoc.createElement('width'))
    width.appendChild(kmlDoc.createTextNode("7"))    

    LineString = Placemark.appendChild(kmlDoc.createElement('LineString'))    
    tessellate = LineString.appendChild(kmlDoc.createElement('tessellate'))
    tessellate.appendChild(kmlDoc.createTextNode("1"))    
    coorElement = LineString.appendChild(kmlDoc.createElement('coordinates'))

    out_loc = ''
    for i, loc in enumerate(locs):
        out_loc += str(loc[0]) +','+ str(loc[1]) + ',0 '
    coorElement.appendChild(kmlDoc.createTextNode(out_loc))
    #print("out_loc",out_loc)
    with open(fileName, 'wb') as file:
        file.write(kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8')  )

def read_kml_line(filename):
    locs = []
    doc = xml.dom.minidom.parse(filename)  
    # doc.getElementsByTagName returns the NodeList
    doc_el = doc.getElementsByTagName("Document")[0]
    name = doc_el.getElementsByTagName("name")[0].firstChild.data
    print(name)
    place = doc_el.getElementsByTagName("Placemark")
    lst = place[0].getElementsByTagName("LineString")
    data = lst[0].getElementsByTagName("coordinates")[0].firstChild.data.split(' ')
    #print("3:", len(data))
    for loc in data:
        loc_a = loc.split(',')
        if len(loc_a)>1: locs.append([loc_a[0],loc_a[1]])
    #print(locs)
    #print(name)
    return name, np.array(locs)

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
    return tr, angle, length

path_name, gps_in = read_kml_line(file_kml_src)
cur_az, _, dist = wgs84_geod.inv(gps_in[0][0], gps_in[0][1], gps_in[1][0], gps_in[1][1])
#print("cur_az",cur_az,np.degrees(cur_az))
#cur_az = cur_az + 2*math.pi if cur_az < -math.pi else cur_az # angle
#cur_az = cur_az - 2*math.pi if cur_az > math.pi else cur_az # angle
cur_az = np.radians(cur_az)
gps_out = [gps_in[0]]
print("Kml open is ok ", gps_in[0], cur_loc, geopy.distance.geodesic(gps_in[0][:2], gps_in[-1][:2]).m)
#print("srt_xy",srt_xy)
cap = cv2.VideoCapture(file_video_in)
video_fh_fw =  (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("video_fh_fw:",video_fh_fw)
out = cv2.VideoWriter("out_kml.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))
#out = cv2.VideoWriter(file_video_out, cv2.VideoWriter_fourcc(*'XVID'), 10, (v_scale_out[0]*2+3*10,v_scale_out[1]))
#from matplotlib import pyplot as plt
orb = cv2.ORB_create()
kp1 = None
drow = True
top_feat = 16
#route = [start_loc]
#route_xy = [start_loc]
res_f = np.empty((top_feat, 2))
res_d = np.empty((top_feat, 2))
res_xy = np.empty((top_feat))
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
cnt = 0
gps2img = 1
cur_loc = np.array([600*gps2img,360*gps2img])
counter_start = 4
counter = counter_start
err_cnt = 4
center = ((np.array(video_fh_fw)[[1,0]])/2).astype(int)
center_max = np.linalg.norm(center)
print('center=',center,center_max )
route = [cur_loc.copy().astype(int)]
while True:
    r, frame = cap.read()
    if r:
        counter -= 1
        err_cnt = 4
        #print(f"  cnt:{cnt}, {counter}", end='\r')        
        img_0 = cv2.resize(frame, v_scale)
        img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)        
        if not kp1:
            kp1 = orb.detect(img_0,None)
            kp1, des1 = orb.compute(img_0, kp1)
            if not counter: counter = 1
            cv2.imshow('video', img_0)
            cv2.moveWindow('video', 30, 30)
            continue
        else:    
            if not counter:
                img = img_0.copy()
                #print("start")
                #cv2.circle(img, center, 10, (255,0,0),-1)
                #cv2.circle(img, (300,300), 100, (255,0,0),-1)
                #print("start2")
                counter = counter_start
                kp2 = orb.detect(img,None)
                kp2, des2 = orb.compute(img, kp2)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key = lambda x:x.distance)[:top_feat]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                #img = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                for i, match in enumerate(matches):
                    p1 = kp1[match.queryIdx]
                    p2 = kp2[match.trainIdx]                        
                    res_d[i] = np.array(p2.pt) - np.array(p1.pt)
                    #print(i, type(p2.pt),p2.pt)
                    delta_1 = center - np.array(p1.pt)
                    delta_2 = center - np.array(p2.pt)
                    res_xy[i] = np.absolute(np.sqrt(delta_1.dot(delta_1)) - np.sqrt(delta_2.dot(delta_2)))/center_max  # norm distance from center
                    res_f[i][0] = np.sqrt(res_d[i].dot(res_d[i])) # length
                    #cv2.line(img, center, (int(p2.pt[0]),int(p2.pt[1])), (255,0,0), 1)
                    angl = p2.angle - p1.angle
                    res_f[i][1] = angl - 360 if angl > 180 else angl
                    res_f[i][1] = res_f[i][1] + 360 if res_f[i][1] < -180 else res_f[i][1]                       
                    #cv2.putText(img, f"{i}_{res_f[i][1]:.0f}_{res_xy[i][0]:.2f}/ {res_f[i][0]/center_max:.2f}",(int(p2.pt[0]),int(p2.pt[1])),0,.4,(255,255,255),1)
                avg_delta = res_f.mean(axis=0)#.astype(int)
                xy_delta = res_xy.mean()#.astype(int)
                #print("xy_delta",xy_delta)
                #max = np.argmax(res_xy)
                #min = np.argmin(res_xy)
                #print(f"min:{res_xy[min]} max:{res_xy[max]}, {avg_delta[1]}")
                #cv2.putText(img, f"min:{res_xy[min]:.2f} max:{res_xy[max]:.2f}, {avg_delta[1]:.2f}",center,0,1,(255,0,0),1)
                #loc_delta = res_d.mean(axis=0)#.astype(int)
                #cur_loc_new = np.add(cur_loc, loc_delta)                
                #tr1, angle_v, length = angle_p(cur_loc, cur_loc_new)
                cur_az += np.radians(avg_delta[1])
                cur_az = cur_az + 2*math.pi if cur_az < -math.pi else cur_az # angle
                cur_az = cur_az - 2*math.pi if cur_az > math.pi else cur_az # angle
                #print("az",cur_az,avg_delta[1])
                delta = np.array([avg_delta[0]*xy_delta*np.cos(cur_az),avg_delta[0]*xy_delta*np.sin(cur_az)])# if xy_delta > 0.02 else 0
                cur_loc = np.add(cur_loc, delta)#.astype(int)
                route.append(cur_loc.copy().astype(int))
                #cur_loc_new = fwd_loc(cur_loc, cur_az, avg_delta[0])
                #gps_new = wgs84_geod.fwd(gps_out[-1][0], gps_out[-1][1], cur_az, avg_delta[0]*scale_img2gps)                
                #gps_out.append(gps_new[:2])
                cv2.polylines(img, [np.array(route)//gps2img], False, (255,255,0), 3)
                cv2.circle(img, route[-1]//gps2img, 10, (255,0,255), -1)                
                #print(f"Video: a:{np.degrees(cur_az):.0f} l:{avg_delta[0]:.0f} ", end='\r')                
                cv2.putText(img, f"Video: a:{np.degrees(cur_az):.0f} l:{xy_delta:.2f}", (200,100), 0, 1,  (255,255,255), 2)
                #cv2.putText(img, f"GPS  : a:{np.degrees(gps_az):.0f} l:{gps_dist} ", (400,100), 0, 1,  (255,255,255), 2)
                cv2.imshow('video', img)
                out.write(img)
                #cur_loc = cur_loc_new
                kp1 = tuple(kp2) # tuple
                des1 = des2.copy() #np array
        cnt += 1
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'): break
    else:
      err_cnt -= 1
      #print("err", err_cnt)
      if not err_cnt: break
cv2.imwrite("i1.jpg",img)
print("Kml is ready", len(gps_out))
createKML("PathVideo22.kml", gps_out)

cap.release()    
if out: out.release()
cv2.destroyAllWindows()

