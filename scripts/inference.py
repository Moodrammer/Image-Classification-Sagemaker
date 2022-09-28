import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    num_classes = 133
    # load the pretrained model
    model = models.resnet50(pretrained=True)
    # freeze the different parameters of the model to use for feature extraction
    for param in model.parameters():
        param.requires_grad = False
    # find the number of inputs to the final layer of the network
    num_inputs = model.fc.in_features
    # replace the fc layer trained on imageNet with the fc for our dataset
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model

#-----------------------------------------------------------------------------------------------------------
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model