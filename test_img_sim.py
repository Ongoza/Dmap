import sys
import os
# sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
#import cv2
cur_indx = 2
input_path = './1'
print("Getting vectors for test images...\n")
img2vec = Img2Vec()

# For each test image, we store the filename and vector as key, value in a dictionary
pics = []
for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename)).convert('RGB')
    #img =  cv2.imread(os.path.join(input_path, filename))
    #img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vec = img2vec.get_vec(img)
    pics.append([filename, vec])
    
pic_name = ""

while pic_name != "exit":
    pic_name = int(input("\nWhich filename would you like similarities for?\nAvailable options: " + str(len(pics)) + "\n"))
    try:
        sims = {}
        for key in range(len(pics)):
            # if key == pic_name: continue
            sims[key] = cosine_similarity(pics[pic_name][1].reshape((1, -1)), pics[key][1].reshape((1, -1)))[0][0]

        d_view = [(v, k) for k, v in sims.items()]
        d_view.sort(reverse=True)
        for v, k in d_view:
            print(v, k)

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
