import os
import cv2
import numpy as np
import onnxruntime as rt

sess_256 = rt.InferenceSession("osnet_x0_25_msmt17_dinamic_2.onnx")

img_name = "Melitopol"
tile_folder = '1'
# step size
N = 224 # Y
M = 224 # X
img =  cv2.imread(img_name+".jpg")

image_copy = img.copy() 
imgheight=img.shape[0]
imgwidth=img.shape[1]
y_cols = int(imgheight/N)
x_cols = int(imgwidth/M)

keys = np.empty((y_cols, x_cols, 256), dtype=np.float16)
print("image shape", imgheight, imgwidth, keys.shape)
counter = 0
counter100 = 100
x1 = 0
y1 = 0
y2 = -1
for y in range(0, imgheight, N):
    y2 += 1
    x2 = -1
    for x in range(0, imgwidth, M):
        x2 += 1
        if counter100 == 0:
            counter100 = 100
            print(counter)
        counter += 1        
        counter100 -= 1
        if (imgheight - y) < N or (imgwidth - x) < M: break
             
        y1 = y + N
        x1 = x + M
 
        # check whether the patch width or height exceeds the image width or height
        if x1 >= imgwidth: x1 = imgwidth - 1
        if y1 >= imgheight: y1 = imgheight - 1
        #Crop into patches of size MxM
        tiles = image_copy[y:y+N, x:x+M]
        image_256 = cv2.cvtColor(tiles, cv2.COLOR_BGR2RGB)
        image_256 = cv2.resize(image_256, (128, 256))
        image_256 = np.transpose(image_256, (2, 0, 1)).astype(np.float32)/255.0
        features_256 = sess_256.run(None, {'input':[image_256]})[0]        
        features_256 = features_256.reshape((1, -1, 2)).sum(axis=2)/2.0
        #print("shape", keys[y2][x2])
        #print("feate", features_256)

        keys[y2][x2] = np.float16(features_256)

        #Save each patch into file directory
        cv2.imwrite(tile_folder+'/t_'+str(counter)+'_'+str(x2)+'_'+str(y2)+'.jpg', tiles)
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

#Save full image into file directory
#cv2.imshow("Patched Image",img)
cv2.imwrite(img_name+"_p.jpg",img)
print(keys[0])
print(keys[-1])
with open(img_name+'.npy', 'wb') as f:
    np.save(f, keys)
print('Done!')
