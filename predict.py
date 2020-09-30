#importing necessary libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser (description = "Prediction script parser")
#parser.add_argument ('image_path', help = "Pass a single Image", default = 'flowers/test/41/image_02302.jpg')
parser.add_argument('load_checkpoint', help = 'Provice path to checkpoint',  type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument('--category_names', help = 'Map the flower to the name', type = str)
parser.add_argument ('--gpu', help = "Option to use GPU. Optional", type = str)
                    
                     
#load checkpoint
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
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    test_image = Image.open(image)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cpu')
    image = process_image(image_path)
    #convert numpy array to tensor
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(dim=0)
    
    with torch.no_grad():
        output = model.forward(image_tensor)
    # Convert to probabilities
    out_probability = torch.exp(output)
     # Find the top 5 results
    probability, label_val = out_probability.topk(topk)
    #top 5 probabilities to numpy array
    probability = probability.numpy()
    label_val = label_val.numpy()
    #convert to list
    probability = probability.tolist()[0]
    label_val= label_val.tolist()[0]
    mapping = {val: key for key, val in model.class_to_idx.items()}
    label = [mapping[item] for item in label_val]
    flowers = [cat_to_name[item] for item in label]
    
    return probability, label, flowers
image_path = 'flowers/test/41/image_02302.jpg'
image =  process_image(image_path)
model = load_checkpoint(image_path)
probability, label, flowers = predict(image_path, model, 5)

print(flowers, probability)
