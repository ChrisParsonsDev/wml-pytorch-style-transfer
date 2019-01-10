# Watson Machine Learning - PyTorch Style Transfer

This readme acts as a step by step guide to using the [Watson Machine Learning](https://www.ibm.com/cloud/machine-learning) service for GPUs in the IBM Cloud. The code itself is for a Style Transfer workload, applying the "style" of one image to the "content" of another.

If you're interested in finding out more about style transfer check out [this paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

So what does this repo contain?

1. pytorch-model - Location of the PyTorch script that does the style transfer.
2. wml-style-transfer.ipynb - Notebook that executes the PyTorch script via the WML service.

## PyTorch Script Usage

You are of course free to use the python script `./pytorch-model/style-transfer.py` anywhere you like. Actually, I'd encourage it! You'll need either GPUs or a whole lot of time.

The script takes a couple of command line arguments to let you specify style/content images as well as the number of training iterations.

Usage:

`python3 ./tf-model/style-transfer.py --styleImageFile ${DATA_DIR}/emma.jpg --contentImageFile ${DATA_DIR}/chris.jpeg --trainingIters 2000

`

## Tutorial..
