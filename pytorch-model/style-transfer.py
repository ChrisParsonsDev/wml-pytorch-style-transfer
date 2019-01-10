#!/usr/bin/env python
# coding: utf-8

'''
Style Transfer in PyTorch (VGG). To be used with the Watson Machine Learning service in IBM Cloud!

Author: Chris Parsons

Code: https://github.com/chrisparsonsdev/wml-pytorch-style-transfer

'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.optim as optim
from torchvision import transforms, models

content_image_file = ""
style_image_file = ""

# Parameters
training_iters = 5000
save_every = 400
# Location to store saved images (COS bucket)..
output_path = os.environ["RESULT_DIR"]+"/results"

# Main method for - accepts the command line args.
def main(argv):

    if len(argv) < 6:
        sys.exit("Not enough arguments provided.")

    global content_image_file, style_image_file, training_iters

    i = 1
    while i <= 3:
        arg = str(argv[i])
        if arg == "--contentImageFile":
            content_image_file = str(argv[i+1])
        elif arg == "--styleImageFile":
            style_image_file = str(argv[i+1])
        elif arg =="--trainingIters":
            training_iters = int(argv[i+1])
        i += 2

if __name__ == "__main__":
    main(sys.argv)

# ## Load in VGG19 (features)
#
# VGG19 is split into two portions:
# * `vgg19.features`, which are all the convolutional and pooling layers
# * `vgg19.classifier`, which are the three linear, classifier layers at the end
# get the "features" portion of VGG19 (do not need the "classifier" portion)

print("I CAN LOG")

vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)


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
    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image

# load in content and style image
content = load_image(content_image_file).to(device)
# Resize style to match content, makes code easier
style = load_image(style_image_file, shape=content.shape[-2:]).to(device)

# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# ---
# ## VGG19 Layers
#
# To get the content and style representations of an image, we have to pass an image forward through the VGG19 network until we get to the desired layer(s) and then get the output from that layer.
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
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


# ---
# ## Gram Matrix
#
# The output of every convolutional layer is a Tensor with dimensions associated with the `batch_size`, a depth, `d` and some height and width (`h`, `w`). The Gram matrix of a convolutional layer can be calculated as follows:
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
    _, d, h, w = tensor.size()

    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)

    ## calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


# get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start of with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)


# ---
## Loss and Weights

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# you may choose to leave these as is
content_weight = 1  # alpha
style_weight = 1e6  # beta


# ## Updating the Target & Calculating Losses
#
# #### Content Loss
#
# The content loss will be the mean squared difference between the target and content features at layer `conv4_2`. This can be calculated as follows:
# ```
# content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
# ```
#
# #### Style Loss
#
# The style loss is calculated in a similar way, only you have to iterate through a number of layers, specified by name in our dictionary `style_weights`.
##
# Intermittently, we'll print out this loss/save the image to our output dir.
#

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 5000  # decide how many iterations to update your image

for ii in range(1, steps+1):
    print("Training...")
    ## Get the features from your target image
    ## Then calculate the content loss
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # iterate through each style layer and add to the style loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        ## Calculate the target gram matrix
        target_gram = gram_matrix(target_feature)

        ## Get the "style" style representation
        style_gram = style_grams[layer]
        ## Calculate the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)


    ## Calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    # print loss every few iterations
    if  ii % save_every == 0:
        print("Training: ",ii,{'Total Loss: ': total_loss.item()})
        # print('Total loss: ', total_loss.item())
        # Generate unique filename
        filename = str(int(total_loss.item()))+'.png'
        plt.imsave(output_path + filename, im_convert(target))
