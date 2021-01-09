# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 19:52:03 2021

In this test, the learning es done in one imagen considering RGB values.
Then, in the same image, a inference of each pixel is donde. 

To change the image, modify "train_image = cv2.imread('RGB/Tests/t3.jpg')"
 (line 51).

@author: Carlos & Josue
"""

import cv2
import numpy as np
from methods import Classifier

# get_info give the information of the selected pixel.
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
        values[num_clks] = train_image[row,column]
        
        # If num_clks is equal to the samples, then the sampling phase
        # has ended. Destroy the window holding the image.
        if num_clks == samples - 1:
            cv2.destroyAllWindows()


# Define the two classes to be train. First with the no-success one.
classes = ['No Skin', 'Skin']
# Set the number of samples to be acquired.
samples = 15
# Create the object Classifier()
classifier = Classifier()

# Array of the values with which the two classes are trained in order.
class_values = []

# Set the training image.
train_image = cv2.imread('RGB/Tests/t3.jpg')

# Iterate for each class.
for class_name in classes:
    print(f'Sampling for {class_name}')
    
    # Set an empty array for each sample.
    values = np.zeros((samples, 3))
    
    # Start num_clks.
    num_clks = -1
    
    # Show the image.
    cv2.imshow(f'{class_name} Sampler', train_image)
    # Connect the frame with the callback get_info.
    cv2.setMouseCallback(f'{class_name} Sampler', get_info, samples)
    
    # Wait until the user action.
    cv2.waitKey(0)
    
    # Append the training samples.
    class_values.append(values)

# Initiate both classes of the classifier with a lambda value of 50%
classifier.set_classes(classes, class_values, 0.5)

# Make the inference on the training image.
img_mask_nskin = classifier.classify_image(train_image, 'No Skin', 0.5)
img_mask_skin = classifier.classify_image(train_image, 'Skin', 0.5)

# Show the results.
cv2.imshow('Original', train_image)
cv2.imshow('Skin', img_mask_skin)
cv2.imshow('No Skin', img_mask_nskin)

# Wait for the user action and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
