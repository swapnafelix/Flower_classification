import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import  optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
import argparse
from pathlib import Path


parser = argparse.ArgumentParser (description = "train script parser")
parser.add_argument('data_dir', help= 'Provide data directory. Mandatory argument', type=str)
parser.add_argument('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument('--arch', help= 'Choose architecture, optional argument', type = str, default = 'vgg16')
parser.add_argument('--learning_rate', help = 'Learning rate for the model',  type = float, default = 0.001)
parser.add_argument('--epochs', help = 'Number of epochs in the model', type = int, default = 5)
parser.add_argument ('--gpu', help = "Option to use GPU. Optional", type = str, default = 'cpu')
                     
                   
#Setting value loading features
args = parser.parse_args()


data_dir = ''
data_dir = args.data_dir
train_dir = data_dir + '/train'

save_dir = args.save_dir
if save_dir is not None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
else:
   save_dir = ''     

arch = args.arch
if arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

learning_rate = args.learning_rate
if learning_rate is not None: 
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    
epochs = args.epochs
if epochs is not None:
    epochs = epochs


if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
print('Train directory: ' +  train_dir )
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
# TODO: Build and train your network
model = models.vgg16(pretrained=True)
print(model)

for parameter  in model.parameters():
    parameter.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088,  4096)),
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(4096, 1024)),
                        ('relu', nn.ReLU()),
                        ('fc3', nn.Linear(1024, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
model.classifier = classifier
# Use GPU if it's available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define a new, untrainted feed-forward network as a classifier, using ReLU activations and dropout
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                 nn.Linear(4096, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1)
                                )
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
# change to device
model.to(device);

epochs = 5
print_every = 32 # Prints every 30 images out of batch of 64 images
steps = 0
running_loss = 0
for epoch in range(epochs):
    for inputs, labels in trainloaders:
        steps +=1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        #model.to(device)
        optimizer.zero_grad()
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps %print_every == 0:
            valid_loss = 0
            valid_accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    #model.to(device)
                    output = model.forward(inputs)
                    batch_loss = criterion(output, labels)
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
            running_loss = 0
            model.train()   
    
print("\nTraining complete!!")
print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Training loss: {running_loss/print_every:.4f}.. "
                  f"Validation loss: {valid_loss/len(validloaders):.4f}.. "
                  f"Validation accuracy: {valid_accuracy/len(validloaders):.4f}")


# TODO: Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx

#save the state dict with torch.save.

checkpoint = {'architecture': models.vgg16, 
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}
torch.save(checkpoint, save_dir + '/my_checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    #    Loads deep learning model checkpoint.
    checkpoint = torch.load('my_checkpoint.pth')
    #pretrained model
    model = models.vgg16(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    #Load from checkpoing
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

print(args)


