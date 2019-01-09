'''
Style Transfer (VGG) in TensorFlow

Author: Chris Parsons
Project: https://github.com/chrisparsonsdev/wml-tf-style-transfer/
'''

import tensorflow as tf
import input_data
import sys
import os
import itertools
import re
import time
from random import randint

style_image_file = ""
content_image_file = ""

# Parameters
display_step = 400
training_iters = 2000

model_path = os.environ["RESULT_DIR"]+"/model"

# This helps distinguish instances when the training job is restarted.
instance_id = randint(0,9999)

def main(argv):

    if len(argv) < 6:
        sys.exit("Not enough arguments provided.")

    global style_image_file, content_image_file, training_iters

    i = 1
    while i <= 6:
        arg = str(argv[i])
        if arg == "--styleImageFile":
            style_image_file = str(argv[i+1])
        elif arg == "--contentImageFile":
            content_image_file = str(argv[i+1])
        elif arg =="--trainingIters":
            training_iters = int(argv[i+1])
        i += 2

if __name__ == "__main__":
    main(sys.argv)

print(tf.__version__)
print(style_image_file)
