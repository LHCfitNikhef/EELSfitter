# EELSfitter

This repository contains the EELSfitter code developed for the ZLP-subtraction in EEL spectra, 
developed in Python 3.8. 

**Fitter.ipynb** contains both the data preparation for training and the NN model that 
carries out the training.

**Fuctions.py** contains a couple of functions that are used, *e.g.* 
for calculating means, errors and smoothing functions.

**Load_data.py** imports the raw spectra and returns the intensities, calibrates with the maximum intensity
at zero energy loss, and returns both the original and normalized intensities.

**Predictions.ipynb** can be used to evaluate the predictions after training. 

**Predictions_pretrained_nets.ipynb** is similar to the Predictions.ipynb file, but here
one can import the network parameters of pre-trained nets to create predictions on any input array. 

The folders **Data** and **Models** contain the datasets, the results from training and the saved model 
parameters respectively. 

We kindly refer to ***(...)*** for the citation instructions when making use of this software. 
