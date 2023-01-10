import cv2
import numpy as np
import os
from img_to_vec import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import random

#from google.colab.patches import cv2_imshow


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

# Задаємо назву файлу, папку для тайлів та розмір тайлів
img_name = "Melitopol"
tile_folder = 'tiles'
# step size
mod_size = 224
N = 112 # Y
M = N # X
mod_scale = mod_size/N
from PIL import Image
# Створюємо нову чи очищаємо існуючу директорію для тайлів
if not os.path.exists(tile_folder):
   # Create a new directory because it does not exist
   os.makedirs(tile_folder)
   print("The new directory is created!")
else:
  print("The directory is exist!")
  if os.listdir(tile_folder):
    print("The directory is not empty!")
    for f in os.listdir(tile_folder):
      os.remove(os.path.join(tile_folder, f))
    print("The new directory is empty now!")

# Розбиваємо картинку на тайли
img =  cv2.imread(img_name+".jpg")
image_copy = img.copy() 
imgheight=img.shape[0]
imgwidth=img.shape[1]
y_cols = int(imgheight/N)
x_cols = int(imgwidth/N)

img2vec = Img2Vec(cuda=True, model='efficientnet_b2', layer='default', layer_output_size=1408)

# For each test image, we store the filename and vector as key, value in a dictionary
pics = []
counter = 0
counter100 = 100
x1 = 0
y1 = 0
y2 = -1
for y in range(0, imgheight, N):
    y2 += 1
    x2 = -1
    for x in range(0, imgwidth, N):
        if (imgheight - y) < N or (imgwidth - x) < M: break  
        x2 += 1
        if counter100 == 0:
            counter100 = 100
            print(counter)
        counter += 1        
        counter100 -= 1           
        y1 = y + N
        x1 = x + M 
        # check whether the patch width or height exceeds the image width or height
        if x1 >= imgwidth: x1 = imgwidth - 1
        if y1 >= imgheight: y1 = imgheight - 1
        #Crop into patches of size MxM
        tile = image_copy[y:y+N, x:x+M]
        #Save each patch into file directory
        filename = 't_'+str(x2)+'_'+str(y2)+'_.jpg'
        #cv2.imwrite(os.path.join(tile_folder, filename), tile)
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
        #img = Image.open(os.path.join(tile_folder, filename)).convert('RGB')
        tile = cv2.resize(tile,(mod_size, mod_size))
        tile = Image.fromarray(tile).convert('RGB')
        vec = img2vec.get_vec(tile)
        pics.append([filename, vec])
cv2.imwrite(img_name+"_p.jpg",img)
print(str(counter), ' images ready!', len(pics))




#__xy = rotate(__xy, angle)
def test_random():
    #Обираємо довільну картинку на карті
    y_r = int(random.random()*imgheight*0.6)
    x_r = int(random.random()*imgwidth*0.6)
    __xy = np.array([[x_r,y_r],[x_r+M,y_r],[x_r+M,y_r+N],[x_r,y_r+N]])
    #__xy = np.array([(x_r, y_r), (x_r+M, y_r+N)])
    angle = int(360*random.random())
    __xy2 = rotate(__xy, angle)
    __xyA = np.array([[x_r+int(M/2),y_r+int(N/3)],[x_r+int(M/2),y_r+int(2*N/3)]])
    __xyA2 = rotate(__xyA, angle)
    
    img_copy = img.copy()

    tile = img[y_r:y_r+N, x_r:x_r+M]
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    tile = Image.fromarray(tile).convert('RGB')
    cv2.rectangle(img_copy, (__xy[0][0], __xy[0][1]), (__xy[1][0], __xy[1][1]), (255, 255, 255), 2)

    vec = img2vec.get_vec(tile)
    sims = {}
    for i in range(len(pics)):
        key = pics[i][0]
        sims[key] = cosine_similarity(vec.reshape((1, -1)), pics[i][1].reshape((1, -1)))[0][0]
        d_view = [(v, k) for k, v in sims.items()]
    d_view.sort(reverse=True)
    color = [255, 0, 0]
    for i in range(4):
      print("try:", i, d_view[i][0])
      _xy = d_view[i][1].split('_')
      x = int(_xy[1])*N
      y = int(_xy[2])*M
      cv2.rectangle(img_copy, (x, y), (x+N, y+M), color, 2)
      cv2.putText(img_copy, str(int(100*d_view[i][0])), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

      color[1] += 70
    scale = 0.4
    cv2.polylines(img_copy, [__xy2], True, (255, 255, 255), 2)
    cv2.arrowedLine(img_copy, __xyA2[0], __xyA2[1], (0,0,255), 2)

    return cv2.resize(img_copy,(int(imgwidth*scale),int(imgheight*scale)))
    # cv2_imshow(img_copy_s)

 


img1 = test_random()

img2 = test_random()
img3 = test_random()
img4 = test_random()
Hori = np.concatenate((img1, img2, img3, img4), axis=1)
cv2.imshow('sample', Hori)
cv2.moveWindow('sample', 30, 30)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image