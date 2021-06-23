# Imports here
import argparse
import matplotlib.pyplot as plt
import seaborn as sb
import json
import time
import numpy as np
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
import PIL
from workspace_utils import active_session



# Create the label mapping using json module
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



# Parse the user input arguments
parser = argparse.ArgumentParser(prog='predict.py',
                                 usage='predict flower species from provided '
                                 'file')
parser.add_argument(dest='img_pth', action='store', metavar='image_path',
                    type=str, default='flowers/test/102/image_08004.jpg')
parser.add_argument('--top_k', dest='top_k', action='store', type=int,
                    default=5)
parser.add_argument('--category_names', dest='cat_to_name', action='store',
                    type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_const', const='cuda', dest='device',
                    default='cpu')

args = parser.parse_args()



# Create the label mapping using json module and user input
with open(args.cat_to_name, 'r') as f:
    cat_to_name = json.load(f)


# Define a function to load the checkpoint later on
def load_checkpoint(filepath):
    ''' Loads the checkpoint from the provided filepath. Then rebuilds the
        model, optimizer, criterion, and normalization values.
    '''

    checkpoint = torch.load(filepath)
    model = checkpoint['arch']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.fc = checkpoint['classifier']
    criterion = checkpoint['criterion']
    model.class_to_idx = checkpoint['class_to_idx']
    norm_mean = checkpoint['norm_mean']
    norm_std = checkpoint['norm_std']

    return model, optimizer, criterion, norm_mean, norm_std



# Load the checkpoint
(model, optimizer, criterion,
 norm_mean, norm_std) = load_checkpoint('checkpoint.pth')



# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''

    # Open the image
    img = PIL.Image.open(image_path)

    # Resize
    if img.size[0] > img.size[1]:
        img.resize((500, 256))
    else:
        img.resize((256, 500))

    # Crop
    crop_box = [(img.width - 224)/2, (img.height - 224)/2,
                (img.width - 224)/2 + 224, (img.height -224)/2 + 224]
    img = img.crop(crop_box)

    # Normalize the image
    img = np.array(img)/255
    img = (img - np.array(norm_mean)) / np.array(norm_std)

    # Move the color channel to the first dimension of the tensor
    img = img.transpose((2, 0, 1))

    return img



# Function to display an image from a pytorch tensor
def imshow(img, title=None):
    """Imshow for Tensor."""

    fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    img = img.numpy().transpose((1, 2, 0))

    # Undo preprocessing, normalize by default
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean

    # Image needs to be clipped between 0 and 1 or it looks like
    # noise when displayed
    img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_title(title)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax



def predict(image_path, model, topk = args.top_k):
    ''' Predict the class (or classes) of an image using a trained
        deep learning model.
    '''
    model.to(args.device)

    # TODO: Implement the code to predict the class from an image file
    # Process the image so it is a NumPy array and no longer a PyTorch Tensor
    img = process_image(image_path)

    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Process the image with GPU if selected
    if args.device == 'cuda':
        model_input = img_tensor.unsqueeze(0).cuda()
    else:
        model_input = img_tensor.unsqueeze(0)

    # Calculate the probabilities with softmax
    with torch.no_grad():
        output = model.forward(model_input)
        ps = torch.exp(output)
        top_probs, top_labels = ps.topk(topk)
        top_probs = top_probs.cpu().numpy().tolist()[0]
        top_labels = top_labels.cpu().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    top_probs = [probs for probs in top_probs]
    top_labels = [idx_to_class[labels] for labels in top_labels]
    top_flowers = [cat_to_name[labels] for labels in top_labels]

    return top_probs, top_labels, top_flowers



'''Display an image along with the top K classes'''

# Define and display the image path
print('testdir:', args.img_pth)
image_path = args.img_pth

# process the provided image
img = process_image(image_path)

# Find the key from the image_path
for directory in image_path.split('/'):
    if directory.isnumeric() == True:
        key = directory

# Numpy -> Tensor
image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
img = image_tensor

# Find the image title
title = cat_to_name[key]

prediction_results = predict(image_path, model)
top_probs, top_labels, top_flowers = prediction_results
print('Top probs:', top_probs)
print('Top labels:', top_labels)
print('Top flowers:', top_flowers)
