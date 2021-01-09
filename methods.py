# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:49:49 2021

@author: Alberto
"""
import numpy as np
import cv2

# Class to obtain the probabilities of a given class
class Class:
    # Constructor
    def __init__(self, np_values, success, lambda_value):
        self.values = np_values  # Training values.
        self.mu = np.mean(np_values, axis=0)  # Mean vector.
        self.cov = np.cov(np_values.T)  # Covariance Matrix.
        self.det = np.linalg.det(self.cov)  # Determinant value of the covariance matrix.
        self.inv = np.linalg.inv(self.cov)  # Inverse Matrix of the covariance matrix.
        self.dim = len(self.mu)  # Dimention of vectors.
        self.binaryC = success  # Whether is success case (1) or not (0)
        self.w = self.setW(lambda_value) # Bernoulli.
    
    # Calculates the likelihood of a given value.
    def likelihood(self, v_x):
        dif = v_x - self.mu 
            
        # Normal Distribution Formula.
        denominator = ((2 * np.pi)**(self.dim / 2)) * np.sqrt(self.det)
        nominator = np.exp(-0.5 * dif.T @ self.inv @ dif)
        
        return nominator / denominator
    
    # Calculates the Bernoulli distribution.
    def setW(self, lambda_value):
        return(lambda_value**self.binaryC) * ((1 - lambda_value)**(1 - self.binaryC))

# Class to classifie values.
class Classifier:
    # Constructor.
    def __init__(self):
        self.states = {} # Dict of binary states.
    
    # Recibes and calculates the probabilities of both classes.
    def set_classes(self, classes, class_values, lambda_value=0.5):
        # Iterate for each given class
        for i, (class_name, values) in enumerate(zip(classes, class_values)):
            self.states[class_name] = Class(values, i, lambda_value)
    
    # Calculates the a posteriori probability of a value being of a given class.
    def classify(self, class_name, value, threshold=0):
        # Initialize variable
        normalization = 0
        
        # Itarate for each train class.
        for c in self.states.keys():
            # Calculates the likelihood.
            likelihood = self.states[c].likelihood(value) * self.states[c].w
            # Sum the likelihood.
            normalization += likelihood
            
            # If this is the desiare class, then save the likelihood.
            if c == class_name:
                prob = likelihood
        
        # probability = prob / normalization
        # Calculates the a posteriori probability.
        inference = prob / normalization
        
        # If the probability es greater than the threshold, then
        # classify the values as 1. If not, as 0
        inference = 1 if inference > threshold else 0
        
        # Return the inference.
        return inference
    
    # Return a filter image of the given class.
    def classify_image(self, image, class_name, threshold, image_depth=None):
        #filter_image = np.empty(image.shape, dtype=np.uint8)
        #probabilities = []
        
        # If a image_depth was given, then consider it to filter.
        # Use just image otherwise.
        if image_depth is not None:
            # Get the dimensión of the depth image.
            h, w, _ = image_depth.shape
            # Create a zero array of the same dimension.
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Iterate for each pixel of the given image.
            for i,row in enumerate(image):
                for j, rgb_v in enumerate(row):
                    # Concatenate the RGB value and the depth information.
                    value_RGBD = np.concatenate((rgb_v, [i, j, image_depth[i][j][0]]))
                    
                    # Fill the correspondent element with the inference
                    mask[i][j] = self.classify(class_name, value_RGBD, 0.5)

        else:
            # Create the array with the inferences matrix by list comprehension.
            mask = np.array([[self.classify(class_name, 
                                            rgb_v, 
                                            threshold) 
                             for rgb_v in row] 
                             for row in image], dtype=np.uint8)
        
        # Filter the image by the True (1) values of mask.
        image_masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Return the filtered image.
        return image_masked
        

        
            
        