import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import pandas as pd
import json
from PIL import Image

#defining mandatory and optional arguments
argp=argparse.ArgumentParser(description="Testing Script")

argp.add_argument('load_directory', help="Checkpoint path, mandatory", type=str)
argp.add_argument('image_directory', help="Image path, mandatory", type=str)
argp.add_argument('--dvc', help="Device: GPU or CPU?, optional", type=str)
argp.add_argument('--topk', help="most likely topkclasses, optional", type=int)
argp.add_argument('--cat_names', help="Categories to real names mapping, provide JSON file name, optional", type=str)

#loading the checkpoint to rebuild the model
def load_checkpoint(path):
    
    checkpoint=torch.load(path)
    
    if checkpoint ['arch'] == 'vgg13':
        model = models.vgg13 (pretrained = True)
    else: #vgg11
        model = models.vgg11 (pretrained = True)
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier=checkpoint['classifier']
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer=checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    return model

#Processing a PIL image for use in PyTorch model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #defining means and standard dev.
    means=np.array([0.485,0.456,0.406])
    sd=np.array([0.229,0.224,0.225])
    
    
    #defining transforms for resize and center crop
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    
    
    #opening the image
    image_pil=Image.open(image)
    
    #resizing and center cropping the PIL image
    image_pil=transform(image_pil)
    
    #converting the PIL image to a Numpy array
    np_image=np.array(image_pil)
    
    #Normalizing the image
    normalized_im=(np.transpose(np_image,(1,2,0))-means)/sd
    
    #transposing to color channel being the first dimension
    normalized_im=np.transpose(normalized_im, (2,0,1))
    
    return normalized_im

#Prediction function
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #processing the image
    image=process_image(image_path)
    
    #converting the numpy image to a tensor
    tensor=torch.from_numpy(image)
    tensor=tensor.float().unsqueeze_(0).to(device)
    
    model.eval()
    
    with torch.no_grad():
        
        model=model.to(device)
        
        ps=torch.exp(model(tensor))
        top_prob,top_cls=ps.topk(topk, dim=1)
        
    return top_prob, top_cls

args=argp.parse_args()
filepath=args.image_directory

#select the device here
if args.dvc=='GPU':
    device='cuda'
else:
    device='cpu'

#load the JSON file if it is provided, otherwise use default
if args.cat_names:
    with open(args.cat_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

#model load using checkpoint
model=load_checkpoint(args.load_directory)

#how many top classes to be predicted? choose what is given, else default to 1
if args.topk:
    clas_num = args.topk
else:
    clas_num = 1
    
#calculate probs and classes

top_prob, top_cls = predict(filepath, model, clas_num)

#class names using mapping with cat_to_name
classnames = [cat_to_name[i] for i in top_cls]

#looping over the classes
for a in range(clas_num):
    print("No. {}/{}.. ".format(a+1, clas_num),
          "Class name: {}.. ".format(classnames[a]),
          "Probability in percent: {.3f}..% ".format(top_prob[a]*100),
         )



