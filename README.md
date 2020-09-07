# EELSfitter

This repository contains the EELSfitter code developed for the ZLP-subtraction in EEL spectra, 
developed in te work of (...).

{\bf Fitter.ipynb} contains both the data preparation for training and the NN model that 
carries out the training.

{\bf Fuctions.py} contains a couple of functions that are used {\it e.g.} 
for calculating means, errors and smoothing functions.

{\bf Load_data.py} imports the raw spectra and returns the intensities, calibrates with the maximum intensity
at zero energy loss, and returns both the original and normalized intensities.

{\bf Predictions.ipynb} can be used to evaluate the predictions after training. 

{\bf Predictions_pretrained_nets.ipynb} is similar to the Predictions.ipynb file, but here
one can import the network parameters of pre-trained nets to create predictions on any input array. 

The folders {\bf Data} and {\bf Models} contain the datasets, the results from training and the saved model 
parameters respectively. 
