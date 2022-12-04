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

from get_predict_input_args import get_predict_input_args

def process_image(image):
    """ 
    Function to scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    :param image: input image
    :return np_image: processed image
    """
    img = Image.open(image)
    
    width, height = img.size
    
    if width == height:
        img.thumbnail((256, 256))
    
    elif height > width:
        ratio = float(width) / float(height)
        newwidth = ratio * 256
        img = img.resize((int(floor(newwidth)), 256))
    
    elif width > height:
        ratio = float(height) / float(width)
        newheight = ratio * 256
        img = img.resize((256, int(floor(newheight))))
    
    left = int(img.size[0]/2-224/2)
    upper = int(img.size[1]/2-224/2)
    right = left + 224
    lower = upper + 224
    
    img = img.crop((left, upper,right,lower))
    
    np_image = np.array(img)    
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std    
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """
    Function to show a Tensor image.
    :param image: image
    :param ax: axis of plot
    :param title: title of plot
    :return ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    """ 
    Function to predict the class (or classes) of an image using a trained deep learning model.
    :param image_path: image path
    :param model: model
    :return: tuple of (top_p_list, flower_names, cat_to_name)
    """
    # preprocess image
    image = process_image(image_path)
    
    img_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
    model_input = img_tensor.unsqueeze(0)

    with torch.no_grad():
        logps = model(model_input)
        ps = torch.exp(logps)
        
        # get top probabilities and respective classes (0-indexed)
        top_p, top_class = ps.topk(topk, dim=1)
        
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        
        top_p_list = []
        idx_class_list = []
        for i in range(top_p.shape[1]):
            top_p_list.append(top_p.data[0][i].item())
            idx_class_list.append(top_class.data[0][i].item())
        
        # convert index to class
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        
        top_labels = [idx_to_class[i] for i in idx_class_list]
        flower_names = [cat_to_name[i] for i in top_labels]

        return top_p_list, flower_names, cat_to_name

def load_checkpoint(filepath):
    """
    Function to load a checkpoint
    """
    checkpoint = torch.load(filepath)
    # make sure to create same model used as before
    model = models.vgg16(pretrained=True)
    # freezing parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    # load model parameters
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def show_result(probs, classes, cat_to_name):
    """
    Function to show predict result
    """
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[-2]
    title_flower = cat_to_name[flower_num]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax)

    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=classes, color=sns.color_palette()[0]);
    plt.xlabel('Probability')
    plt.ylabel('Flower')
    plt.title(title_flower)
    plt.show()
    
if __name__ == '__main__':
    
    inputs = get_predict_input_args()
    
    image_path = inputs.image_path
    checkpoint = inputs.checkpoint
    
    model = load_checkpoint(checkpoint)
    
    probs, classes, cat_to_name = predict(image_path, model)
    
    #show_result(probs, classes, cat_to_name)
    
    print(f'Top 5 predictions: {classes}')
    print(f'Top 5 propabilities according with top 5 predictions: {probs}')
    
    