# EELSfitter

This repository contains the **EELSfitter** code. EELSfiter is an open-source Python-based code developed for the analysis and interpretation of Electron Energy Loss Spectroscopy (EELS) measurements in Transmission Electron Microscopy (TEM). **EELSfitter** is based on machine learning techniques developed by the [NNPDF Collaboration](http://nnpdf.mi.infn.it/) in the context of applications in high energy physics. The current version of **EELSfitter** includes, among others, the following features:

- Construct a model-independent parametrisation of the zero-loss peak (ZLP) intensity distribution in terms of artificial neural networks and the Monte Carlo replica method for faithful uncertainty estimate.

- Implements an efficient subtraction strategy of the ZLP for spectra recorded on specimens where all relevant sources of uncertainty are propagated to physical predictoons withut the meed to rely on approximations such as *e.g.* linear error propagation.

- Exploits the information in the onset region of the ZLP-subtracted spectra to provide information about the value and type of the bandgap of the analysed specimen.



for the ZLP-subtraction in EEL spectra, 
as published in [DOI:(...)](https://google.com/). 
The code was developed in Python 3.8 and requires Tensorflow v2. 

### 

`fitter.ipynb` contains both the data preparation for training and the NN model that 
carries out the training.

`functions.py` contains a couple of functions that are used, *e.g.* 
for calculating means, errors and smoothing functions.

`load_data.py` imports the raw spectra and returns the intensities, calibrates with the maximum intensity
at zero energy loss, and returns both the original and normalized intensities.

`predictions.ipynb` can be used to evaluate the predictions after training. 

`predictions_pretrained_net.ipynb` is similar to the Predictions.ipynb file, but here
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


