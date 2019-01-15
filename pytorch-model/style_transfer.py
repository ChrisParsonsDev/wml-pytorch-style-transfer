#!/usr/bin/env python
# coding: utf-8

'''
Style Transfer in PyTorch (VGG). To be used with the Watson Machine Learning service in IBM Cloud!

Author: Chris Parsons

Code: https://github.com/chrisparsonsdev/wml-pytorch-style-transfer

'''
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models

# Training Params.
content_image_file = ""
style_image_file = ""
training_iters = 2000
save_image_every = 400

# Location to store saved images (COS bucket)..
OUTPUT_PATH = os.environ["RESULT_DIR"]+"/results"

def main(argv):
    """Set WML Training parameters from user"""

    if len(argv) < 6:
        sys.exit("Not enough arguments provided.")

    global content_image_file, style_image_file, training_iters, save_image_every

    i = 1
    while i <= 3:
        arg = str(argv[i])
        if arg == "--contentImageFile":
            content_image_file = str(argv[i+1])
        elif arg == "--styleImageFile":
            style_image_file = str(argv[i+1])
        elif arg == "--trainingIters":
            training_iters = int(argv[i+1])
        elif arg == "--saveImageEvery":
            save_image_every = int(argv[i+1])
        i += 2

if __name__ == "__main__":
    main(sys.argv)

# ## Load in VGG19 (features)
#
# VGG19 is split into two portions:
# * `vgg19.features`, which are all the convolutional and pooling layers
# * `vgg19.classifier`, which are the three linear, classifier layers at the end
# get the "features" portion of VGG19 (do not need the "classifier" portion)

print("Downloading VGG...")
VGG = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in VGG.parameters():
    param.requires_grad_(False)

# move the model to GPU, if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG.to(DEVICE)
print("done.")

# ### Load in Content and Style Images
##
# this helper function squishes content/style images to make sure they're the same size.

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''

    image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image

print("Loading images..")
# load in content and style image
CONTENT = load_image(content_image_file).to(DEVICE)
# Resize style to match content, makes code easier
STYLE = load_image(style_image_file, shape=CONTENT.shape[-2:]).to(DEVICE)
print("done.")

# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# ---
# ## VGG19 Layers
#
# To get the content and style representations of an image,
# we have to pass an image forward through the VGG19 network
# until we get to the desired layer(s) and then get the output from that layer.
#
## Content and Style Features
def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2', ##Content Representation
                  '28': 'conv5_1'}

    features = {}
    x = image

    # model._modules is a dictionary holding each module in the model
    for name, current_layer in model._modules.items():
        x = current_layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


# ---
# ## Gram Matrix
#
# The output of every convolutional layer is a Tensor with dimensions associated
# with the `batch_size`, a depth, `d` and some height and width (`h`, `w`).
# The Gram matrix of a convolutional layer can be calculated as follows:
# * Get the depth, height, and width of a tensor using `batch_size, d, h, w = tensor.size`
# * Reshape that tensor so that the spatial dimensions are flattened
# * Calculate the gram matrix by multiplying the reshaped tensor by it's transpose
#
# *Note: You can multiply two matrices using `torch.mm(matrix1, matrix2)`.*
#


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    ## get the batch_size, depth, height, and width of the Tensor
    _, depth, height, width = tensor.size()

    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)

    ## calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


# get content and style features only once before forming the target image
CONTENT_FEATURES = get_features(CONTENT, VGG)
STYLE_FEATURES = get_features(STYLE, VGG)

# calculate the gram matrices for each layer of our style representation
STYLE_GRAMS = {layer: gram_matrix(STYLE_FEATURES[layer]) for layer in STYLE_FEATURES}

# create a third "target" image and prep it for change
# it is a good idea to start of with the target as a copy of our *content* image
# then iteratively change its style
TARGET = CONTENT.clone().requires_grad_(True).to(DEVICE)

# ---
## Loss and Weights

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
STYLE_WEIGHTS = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# you may choose to leave these as is
CONTENT_WEIGHT = 1  # alpha
STYLE_WEIGHT = 1e6  # beta


# ## Updating the Target & Calculating Losses
#
# #### Content Loss
#
# The content loss will be the mean squared difference between the target and content
# features at layer `conv4_2`. This can be calculated as follows:
# ```
# content_loss = torch.mean((target_features['conv4_2'] - CONTENT_FEATURES['conv4_2'])**2)
# ```
#
# #### Style Loss
#
# The style loss is calculated in a similar way,
# only you have to iterate through a number of layers, specified by name in our
# dictionary `STYLE_WEIGHTS`.
##
# Intermittently, we'll print out this loss/save the image to our output dir.
#

# iteration hyperparameters
OPTIMIZER = optim.Adam([TARGET], lr=0.003)

print("Training model...")
for ii in range(1, training_iters+1):
    ## Get the features from your target image
    ## Then calculate the content loss
    target_features = get_features(TARGET, VGG)
    content_loss = torch.mean((target_features['conv4_2'] - CONTENT_FEATURES['conv4_2'])**2)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # iterate through each style layer and add to the style loss
    for layer in STYLE_WEIGHTS:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        ## Calculate the target gram matrix
        target_gram = gram_matrix(target_feature)

        ## Get the "style" style representation
        style_gram = STYLE_GRAMS[layer]
        ## Calculate the style loss for one layer, weighted appropriately
        layer_style_loss = STYLE_WEIGHTS[layer] * torch.mean((target_gram - style_gram)**2)

        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)


    ## Calculate the *total* loss
    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    # update your target image
    OPTIMIZER.zero_grad()
    total_loss.backward()
    OPTIMIZER.step()


    # print loss every few iterations
    if  ii % save_image_every == 0:
        print("Training: ", ii, {'Total Loss: ': total_loss.item()})
        # print('Total loss: ', total_loss.item())
        # Generate unique filename
        filename = str(int(total_loss.item()))+'.png'
        plt.imsave(OUTPUT_PATH + filename, im_convert(TARGET))
