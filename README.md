# EELSfitter

This repository contains the EELSfitter code developed for the ZLP-subtraction in EEL spectra, 
as published in [DOI:(...)](https://google.com/). 
The code was developed in Python 3.8 and requires Tensorflow v2. 

### 

`Fitter.ipynb` contains both the data preparation for training and the NN model that 
carries out the training.

`Functions.py` contains a couple of functions that are used, *e.g.* 
for calculating means, errors and smoothing functions.

`Load_data.py` imports the raw spectra and returns the intensities, calibrates with the maximum intensity
at zero energy loss, and returns both the original and normalized intensities.

`Predictions.ipynb` can be used to evaluate the predictions after training. 

`Predictions_pretrained_net.ipynb` is similar to the Predictions.ipynb file, but here
one can import the network parameters of pre-trained nets to create predictions on any input array. 

The folders `Data`and `Models` contain the datasets, the results from training and the saved model 
parameters respectively. 


If you make use of this software, please cite our article using the following bibtex entry:

```
@book{roest2020eelsfitter,
    title={Charting the low-loss region in Electron Energy Loss Spectroscopy with machine learning},
    author={Laurien I. Roest, Sabrya E. van Heijst, Luigi Maduro, Juan Rojo and Sonia Conesa-Boj},
    doi={\url{google.com}},
    year={2020}
}
```


