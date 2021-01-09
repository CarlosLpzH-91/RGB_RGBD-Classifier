# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:33:51 2021

In this test, the learning es done in one imagen considering RGB and Depth values.
Then, in the same image, a inference of each pixel is donde. 

To change the image, modify "train_image_RGB = cv2.imread('RGBD/1_RGB.png')"
 and "train_image_D = cv2.imread('RGBD/1_D.png')" (line 51 & 53). 

@author: Carlos & Josue
"""

import cv2
import numpy as np
from methods import Classifier

# Get_info give the information of the selected pixel.
def get_info(event, column, row, flags, params):
    # num_clks holds the number of times the user has clicked in
    # the image.
    global num_clks
    
    # Set the samples required for training.
    samples = params
    
    # If user clicked in the image, then increase by 1 num_clcks 
    # and get the information.
    if event == cv2.EVENT_LBUTTONDOWN:
        num_clks += 1
        print(f'Sample {num_clks}')
        values[num_clks] = np.concatenate((train_image_RGB[row,column], 
                                           [row, column, train_image_D[row][column][0]]))
        # If num_clks is equal to the samples, then the sampling phase
        # has ended. Destroy the window holding the image.
        if num_clks == samples - 1:
            cv2.destroyAllWindows()


# Define the two classes to be train. First with the no-success one.
classes = ['No Object', 'Object']
# Set the number of samples to be acquired.
samples = 10
# Create the object Classifier()
classifier = Classifier()

# Array of the values with which the two classes are trained in order.
class_values = []

# Set the training RGB image.
train_image_RGB = cv2.imread('RGBD/1_RGB.png')
# Set the training Depth image.
train_image_D = cv2.imread('RGBD/1_D.png')

# Iterate for each class.
for class_name in classes:
    print(f'Sampling for {class_name}')
    
    # Set an empty array for each sample.
    values = np.zeros((samples, 6))
    
    # Start num_clks.
    num_clks = -1
    
    # Show the image.
    cv2.imshow(f'{class_name} Sampler', train_image_RGB)
    # Connect the frame with the callback get_info.
    cv2.setMouseCallback(f'{class_name} Sampler', get_info, samples)
    
    # Wait until the user action.
    cv2.waitKey(0)
    
    # Append the training samples.
    class_values.append(values)

# Initiate both classes of the classifier with a lambda value of 50%
classifier.set_classes(classes, class_values, 0.5)

# Make the inference on the training image.
img_mask_nobject = classifier.classify_image(train_image_RGB, 'No Object', 0.5, train_image_D)
img_mask_object = classifier.classify_image(train_image_RGB, 'Object', 0.5, train_image_D)

# Show the results.
cv2.imshow('Original RGB', train_image_RGB)
cv2.imshow('Original Depth ', train_image_D)
cv2.imshow('No Object ', img_mask_nobject)
cv2.imshow('Object ', img_mask_object)

# Wait for the user action and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()