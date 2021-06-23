# Imports here
import argparse
import matplotlib.pyplot as plt
import seaborn as sb
import json
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
import PIL
from workspace_utils import active_session



# retrieve user inputs using argparse. Start with the data set
parser = argparse.ArgumentParser(prog='train.py',
                                 usage='Input directories and model arch')
parser.add_argument(dest='data_dir', action='store', type=str,
                    default='flowers', metavar='data_directory')
parser.add_argument('--save_dir', action='store', type=str, dest='save_dir',
                    default='')
parser.add_argument('--arch', action='store', type=str, dest='arch',
                    default='resnet152',
                    help = 'choose between resnet101, resnet152,'
                    'vgg19')
parser.add_argument('--learning_rate', action='store', type=float, dest='lr',
                    default=0.0005)
parser.add_argument('--epochs', action='store', type=int, dest='epochs',
                    default=5)
parser.add_argument('--gpu', action='store_const', const='cuda', dest='device',
                    default='cpu')

args = parser.parse_args()



# Normalization means and standard deviations
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# Define the data directory from user input
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define the transforms for the training, validation, and testing sets
train_trans = transforms.Compose([transforms.RandomRotation((45)),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.ToTensor(),
                                  transforms.Normalize(norm_mean, norm_std)])

valid_trans = transforms.Compose([transforms.Resize(225),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(norm_mean, norm_std)])

test_trans = transforms.Compose([transforms.Resize(225),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(norm_mean, norm_std)])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_trans)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_trans)
test_data = datasets.ImageFolder(test_dir, transform=test_trans)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64,
                                          shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64,
                                          shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64,
                                         shuffle = True)



# Use CUDA GPU acceleration or CPU based on user input
print('device in use:', args.device)
device = torch.device(args.device)



# Define the pretrained model to use
if args.arch == 'resnet152':
    model = models.resnet152(pretrained = True)
    # Define the classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 1024)),
                                            ('relu', nn.ReLU()),
                                            ('drop1', nn.Dropout(p=0.25)),
                                            ('fc2', nn.Linear(1024, 512)),
                                            ('relu', nn.ReLU()),
                                            ('drop2', nn.Dropout(p=0.25)),
                                            ('fc3', nn.Linear(512, 256)),
                                            ('relu', nn.ReLU()),
                                            ('drop3', nn.Dropout(p=0.25)),
                                            ('fc4', nn.Linear(256, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

elif args.arch == 'resnet101':
    model = models.resnet101(pretrained = True)
    # Define the classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 1024)),
                                            ('relu', nn.ReLU()),
                                            ('drop1', nn.Dropout(p=0.25)),
                                            ('fc2', nn.Linear(1024, 512)),
                                            ('relu', nn.ReLU()),
                                            ('drop2', nn.Dropout(p=0.25)),
                                            ('fc3', nn.Linear(512, 256)),
                                            ('relu', nn.ReLU()),
                                            ('drop3', nn.Dropout(p=0.25)),
                                            ('fc4', nn.Linear(256, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))



elif args.arch == 'vgg19':
    model = models.vgg19(pretrained = True)
    # Define the classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('drop1', nn.Dropout(p=0.25)),
                                            ('fc2', nn.Linear(4096, 1024)),
                                            ('relu', nn.ReLU()),
                                            ('drop2', nn.Dropout(p=0.25)),
                                            ('fc3', nn.Linear(1024, 256)),
                                            ('relu', nn.ReLU()),
                                            ('drop3', nn.Dropout(p=0.25)),
                                            ('fc4', nn.Linear(256, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))



# Block backpropogation
for param in model.parameters():
    param.requires_grad = False


# Define the model
model.fc = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)


# device will be chosen based on the availability of a CUDA enabled GPU
model.to(device);



## Use a pretrained model to classify the flower images
epochs = args.epochs
steps = 0
print_every = 5
running_loss = 0

train_losses, valid_losses = [], []
valid_accuracy = []

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1

        # move input and label tensor to the default device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Validation on the seperate validater set
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0

            for images, labels in validloader:

                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                batch_loss = criterion(log_ps, labels)
                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))
            valid_accuracy.append(accuracy/len(validloader))

            # Print training results as the training progresses
            print("Epoch: {}/{}..".format(epoch+1, epochs),
                  "Training Loss: {:.3f}".format(running_loss/len(trainloader)),
                  "Valid Loss: {:.3f}".format(running_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
            model.train()



# print out training time and plot results
plt.plot(train_losses, label = 'Training Loss')
plt.plot(valid_losses, label = 'Validation Loss')
plt.legend(frameon='False');
plt.plot(valid_accuracy, label = 'Validation Accuracy')



# Test the accuracy of the model and print the results as the test progresses
device = torch.device(args.device)
model.to(device)
model.eval()

epochs = 5
steps = 0
print_every = 5
test_losses, test_accuracy = [], []

for epoch in range(epochs):
    for images, labels in testloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0

            for images, labels in testloader:

                # move input and label tensor to the default device
                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                test_batch_loss = criterion(log_ps, labels)
                test_loss += test_batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            test_losses.append(test_loss/len(testloader))
            test_accuracy.append(accuracy/len(testloader))

            print(("Epoch: {}/{}..".format(epoch+1, epochs)),
                  ("Test Loss: {:.3f}..".format(test_loss/len(testloader))),
                  ("Test Accuracy: {:.3f}".format(accuracy/len(testloader))))

            test_loss = 0


# Save the trained model to a checkpoint file
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch': model,
              'norm_mean': norm_mean,
              'norm_std': norm_std,
              'criterion': criterion,
              'classifier': classifier,
              'optimizer': optim.Adam(model.fc.parameters(), lr = .0005),
              'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, (args.save_dir + 'checkpoint.pth'))
