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

argp=argparse.ArgumentParser(description="Settings for the Neural Network")

#adding the arguments
argp.add_argument("--arch", help="model architecture", type=str)
argp.add_argument("--epc", help="Number of epochs", type=int)
argp.add_argument("--dvc", help="device to run the model on:CPU/GPU?", type=str)
argp.add_argument("--lr", help="choose the learning rate", type=float)
argp.add_argument("--hidden_units", help="choose the number of hidden units", type=int)
argp.add_argument("data_dir", help="Please provide data directory, this is required.", type=str) #mandatory argument
argp.add_argument("--save_directory", help="Please provide save directory.", type=str)

#parsing the arguments
args=argp.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define the device
if args.dvc =='GPU':
    device='cuda'
else:
    device='cpu'

if data_dir:                                                                                                                                      
# TODO: Define your transforms for the training, validation, and testing sets

#Training Transform with rotation, resizing, cropping and vertical flip
    train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

#Validation and testing transforms--Resizing and then Cropping
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network

def model_create(arch, hidden_units):
    
    #if arch entered is vgg13                         
    if arch== 'vgg13':
        
        model=models.vgg13(pretrained=True)
        
        for param in model.features.parameters():
            param.requires_grad = False
        
        #if hidden units are provided
        if hidden_units:
        
            classifier = nn.Sequential(nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units,len(cat_to_name)),
            nn.LogSoftmax(dim=1))
        #if hidden_units are not provided, default it to 256                      
        else:
            classifier = nn.Sequential(nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,len(cat_to_name)),
            nn.LogSoftmax(dim=1))                       
                             
    #Else Loading the pre-trained model from PyTorch vgg11
    else:
        arch='vgg11'                     
        model=models.vgg11(pretrained=True)
    
        #Freeze training for all "features" layers
    
        for param in model.features.parameters():
                             
            param.requires_grad = False
        
        #if hidden units are provided
        if hidden_units:
            classifier = nn.Sequential(nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units,len(cat_to_name)),
            nn.LogSoftmax(dim=1))
        #if hidden_units are not provided, default it to 256                      
        else:
            classifier = nn.Sequential(nn.Linear(25088,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256,len(cat_to_name)),
            nn.LogSoftmax(dim=1))                    

    #Replacing the classifier in the pre-trained classifier with our classifier
    model.classifier=classifier
    
    return model, arch

#loading the defined model
model, arch=model_create(args.arch, args.hidden_units)

#Loss Function and Optimizer

#Specifying negative Log likelihood loss function
criterion = nn.NLLLoss()

#Specifying optimizer depending on whether or not learning rate was provided,defaulting to 0.10
if args.lr:
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.10)                         

#Move our model to whichever device is available, cuda or CPU

model.to(device);

#Training the Model
#Setting epocs

if args.epc:#if number of epochs are provided
    epochs=args.epc
else:
    epochs=10 #setting a default of 10 epochs if number of epochs are not provided
     
steps=0
running_loss=0
print_every=100

#looping through the number of epochs
for epoch in range(epochs):
    
    #looping through the data
    for images,labels in trainloader:
        
        #steps increment through each batch
        steps+=1
        
        #Moving images and labels to the default device
        images,labels = images.to(device), labels.to(device)
        
        #making the gradients zero to avoid gradient accumulation
        optimizer.zero_grad()
        
        #calculating the log probabilities
        logps=model(images)
        
        #calculating the loss
        loss=criterion(logps, labels)
        
        #backward pass
        loss.backward()
        
        #taking a step with the optimizer to update the parameters
        optimizer.step()
        
        running_loss += loss.item()
        
        #checking the model performance on validation set
        if steps % print_every == 0:
            
            #setting the validation loss and accuracy to zero
            valid_loss=0
            valid_accuracy=0
            
            #turn the model in evaluation mode
            model.eval()
            
            with torch.no_grad():
                
                for images,labels in validloader:
                    
                    #transferring images and labels to GPU
                    images,labels = images.to(device), labels.to(device)
                    
                    #calculating log probabilities for data in validation set
                    logps=model(images)
                    
                    #calculating loss
                    batch_loss=criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    #calculate validation probabilities
                    ps = torch.exp(logps)
                    
                    #get the most likely class
                    top_ps, top_class = ps.topk(1, dim=1)
                    
                    #calculating equality after comparing top_class and labels
                    equality = top_class == labels.view(*top_class.shape)
                    
                    #accuracy calculation
                    valid_accuracy += torch.mean(equality.type(torch.cuda.FloatTensor)).item()
                
            #Printing the results depending on print_every
            print(f"Epoch {epoch}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation Accuracy: {valid_accuracy/len(validloader):.3f}")
            
            running_loss=0
            #setting model back in training mode
            model.train()
            
#Saving Trained Model
model.to('cpu')
#checkpoint save
model.class_to_idx= train_data.class_to_idx
#defining the checkpoint into a dictionary
checkpoint={
    'class_to_idx':train_data.class_to_idx,
    'state_dict':model.state_dict(),
    'optimizer_dict':optimizer.state_dict(),
    'classifier':model.classifier,
    'optimizer':optimizer,
    'arch':arch
}

#save the model for later use
if args.save_directory:
    torch.save(checkpoint, args.save_directory + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')                         




