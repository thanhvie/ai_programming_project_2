import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import json

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from math import floor

from get_input_args import get_input_args

def load_data(data_dir):
    """
    Function to load data into data loader
    :param data_dir: data directory
    :return: a tuple of (train_data, trainloader, validloader)
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return train_data, trainloader, validloader

def build_model():
    """
    Function to build a model from pretrained model (vgg16)
    :param: None
    :return: a tuple of (optimizer, model)
    """
    
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # update model classifier for our project
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 1024)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(1024, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    return optimizer, model

def train_model(trainloader, validloader, optimizer, model):
    """
    Function to train model
    :param trainloader: trainloader
    :param validloader: validloader
    :param optimizer: optimizer
    :param model: untrained model
    :return model: trained model
    """
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    criterion = nn.NLLLoss()
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {val_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model

def save_model(model, train_data, checkpoint_path):
    """
    Function to save a model into .pth file extension
    :param model: model
    :param train_data: trained data
    :checkpoint_path: checkpoint path
    :return: None
    """
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx
                  }
    torch.save(checkpoint, checkpoint_path)
    
if __name__ == '__main__':
    
    # get data directory from user input
    data_dir = get_input_args()
    data_dir = data_dir.data_dir
    print(data_dir)
    
    # load data
    train_data, trainloader, validloader = load_data(data_dir) 
    
    # build model
    optimizer, model = build_model()
    
    # train the model
    trained_model = train_model(trainloader, validloader, optimizer, model)
    
    # save the model
    save_model(trained_model, train_data, 'checkpoint.pth')
    print("Complete saving trained model.")
    
    
    
    
    
    
    
    