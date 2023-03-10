import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json

fname = 'dog.jpg'
#fname = 'Melitopol.jpg'

transform = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), transforms.Normalize(mean=0., std=1.)])
image = Image.open(str(fname))
#plt.imshow(image)
model = models.resnet18(weights="DEFAULT")
#model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#model = models.resnet34(weights="DEFAULT")
#print(model)
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
#print(f"Total convolution layers: {counter}")
#print("conv_layers")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

image = transform(image)
#print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
#print(f"Image shape after: {image.shape}")
image = image.to(device)
outputs = []
names = []
j = 0
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer).split('(')[0]+"_"+str(j))
    j += 1
#print(len(outputs))
#print feature_maps
#for feature_map in outputs: print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
#for fm in processed: print(fm.shape)

fig = plt.figure(figsize=(30, 50))
def add_plot(i, img, end):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(img)
    a.axis("off")
    a.set_title(names[end]+'_'+str(np.array(processed[end].shape).prod()), fontsize=30)
#for i in range(len(processed)): add_plot(i)
add_plot(0,processed[0],0,)
print("len=", len(processed))
num = 16
end  = len(processed) - num 
for i in range(1, num):
    end += 1
    add_plot(i, processed[end], end)
plt.savefig(str(fname+'_features2.jpg'), bbox_inches='tight')
#plt.figure(figsize=(2, 2))
#plt.rcParams['figure.figsize'] = [10, 10]
#plt.show()
print("Done!")