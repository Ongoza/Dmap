import numpy as np
import cv2
import onnxruntime as rt

import torch
import torch.nn.functional as F
import random

img_name = "Melitopol"
# step size
N = 64 # Y
M = 32 # X

def cosine_similarity_torch(x1, x2, dim=-1):
    cross = (x1 * x2).sum(dim=dim)
    x1_l2 = (x1 * x1).sum(dim=dim)
    x2_l2 = (x2 * x2).sum(dim=dim)
    return torch.div(cross, (x1_l2 * x2_l2).sqrt())

def cosine_similarity(vector, matrix):
  return ( np.sum(vector*matrix, axis=1)/(np.sqrt(np.sum(matrix**2, axis=1)) * np.sqrt(np.sum(vector**2)) ) )#[::-1]


keys = np.load(img_name+'.npy')
sess_256 = rt.InferenceSession("osnet_x0_25_msmt17_dinamic_2.onnx")
img =  cv2.imread(img_name+"_p.jpg")
imgheight=img.shape[0]
imgwidth=img.shape[1]

print("keys", keys.shape)

y_r = int(random.random()*imgheight*0.8)
x_r = int(random.random()*imgwidth*0.8)

cv2.rectangle(img, (x_r, y_r), (x_r+M, y_r+N), (0, 255, 255), 1)

tiles = img[y_r:y_r+N, x_r:x_r+M]
image_256 = cv2.cvtColor(tiles, cv2.COLOR_BGR2RGB)

image_256 = cv2.resize(image_256, (128, 256))
image_256 = np.transpose(image_256, (2, 0, 1)).astype(np.float32)/255.0
features_256 = sess_256.run(None, {'input':[image_256]})[0]
features_256 = features_256.reshape((1, -1, 2)).sum(axis=2)/2.0
print("shape", features_256.shape, features_256.dtype)
#x1 = torch.from_numpy(np.float16(features_256))
x1 = np.float16(features_256)

for y in range(keys.shape[0]):    
    cos = cosine_similarity(x1, keys[y])
    #cos = cosine_similarity(x1, torch.from_numpy(keys[x]), dim=-1)        
    for x in range(keys.shape[1]):
        print(f"{y} {x} {x*M:.2f}, {y*N+10:.2f}")
        cv2.putText(img, f"{cos[x]:.2f}", (x*M, y*N+20), 0, 0.4, (250,250,0), 1)

cv2.imwrite(img_name+"_p2.jpg",img)
