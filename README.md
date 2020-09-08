![Banner](Data/Banner.pdf)


# EELSfitter

Welcome to the repository containing the **EELSfitter** code! **EELSfiter** is an open-source Python-based framework developed for the analysis and interpretation of Electron Energy Loss Spectroscopy (EELS) measurements in Transmission Electron Microscopy (TEM). **EELSfitter** is based on the machine learning techniques developed by the [NNPDF Collaboration](http://nnpdf.mi.infn.it/) in the context of applications in high energy physics, in particular feed-forward neural networks for unbiased regression in multidimensional problems.

The current version of **EELSfitter** includes, among others, the following features:

- Construct a model-independent parametrisation of the zero-loss peak (ZLP) intensity distribution in terms of artificial neural networks and the Monte Carlo replica method for faithful uncertainty estimate.

- Implements an efficient subtraction strategy of the ZLP for spectra recorded on specimens where all relevant sources of uncertainty are propagated to physical predictoons without the meed to rely on approximations such as *e.g.* linear error propagation.

- Exploits the information in the onset region of the ZLP-subtracted spectra to provide information about the value and type of the bandgap of the analysed specimen.

**EELSfitter** is developed and maintained by the following developers' team:

- [Sonia Conesa-Boj](mailto:S.C.ConesaBoj@tudelft.nl) (TU Delft), [conesabojlab](https://conesabojlab.tudelft.nl/)
- [Laurien Roest](mailto:L.I.Roest@student.tudelft.nl) (TU Delft)
- [Juan Rojo](mailto:j.rojo@vu.nl) (Nikhef and VU Amsterdam), [juanrojo.com](http://www.juanrojo.com)
- [Isabel Postmes](mailto:isabelpostmes@gmail.com) (TU Delft)

If you have any question about the code and its usage or would like to suggest additional new features, please do not hesitate to contact the developer's team.

## Citation policy

If you use **EELSfitter** in a scientific publication, please cite the original publication:

- *Charting the low-loss region in Electron Energy Loss Spectroscopy with machine learning*, 
Laurien I. Roest, Sabrya E. van Heijst, Luigi Maduro, Juan Rojo, and Sonia Conesa-Boj, [arXiv:2009.aaaaa](https://arxiv.org/abs/2009.00014).

as well as the [DOI:(...)](https://google.com/) associated to the code.

## Scientific publications

Up to now the **EELSfitter** code has been used in the following scientific publications:

- *Charting the low-loss region in Electron Energy Loss Spectroscopy with machine learning*, 
Laurien I. Roest, Sabrya E. van Heijst, Luigi Maduro, Juan Rojo, and Sonia Conesa-Boj, [arXiv:2009.aaaaa](https://arxiv.org/abs/2009.00014).

- *Fingerprinting 2H/3R Polytypism in WS2 Nanoflowers from Plasmons and Excitons to Phonons*,
Sabrya E. van Heijst, Mukai Masaki, E. Okunishi, H. Kurata, Laurien I. Roest, Louis Arcolino Maduro, Juan Rojo, and Sonia Conesa-Boj, [arXiv:2009.aaaaa](https://arxiv.org/abs/2009.00014).

If you use **EELSfitter** in your own publication, we would be glad if you could let us know so that we could add your publication to this list.

## Installation and usage

Below we provide some information about the installation and usage of **EELSfitter**. The code developes a working Python v3.8 installation and relies on the [TensorFlow library](https://www.tensorflow.org/) v2.0.

A most comprehensive version of this user instruction can be found in the Appendix of [this publication](https://arxiv.org/abs/2009.00014). 

### Data
In this folder, the raw spectrum files and the training results are stored.

### Models
Place where the neural network paramers and optimal configurations are saved, to be restored later in order to make predictions on pre-trained nets (see **predicitons_pretrained_net.ipynb**).

### functions.py
File that contains a couple of functions that are used, *e.g.* for calculating means, errors and smoothing functions.

### load_data.py
This file imports the raw spectra and returns the intensities, calibrates with the maximum intensity at zero energy loss, and returns both the original and normalized intensities. The output is two datasets, `df` and `df_vacuum` which contain the information on the in-sample and in-vacuum recorded spectra respectively. 
For each of the spectra the minimum and maximum value of the recorded energy loss need to be set manually in `Eloss_min` and `Eloss_max`.

### fitter.ipynb
This script is used to run the neural network training on the data that was uploaded using **load_data.py**. 
It involves a number of pre-processing steps to determine the hyper-parameters dEI and dEII and it automatically prepares and cuts the data before it is forwarded to the neural network to start the training.
We refer to the Appendix in [this publication](https://arxiv.org/abs/2009.00014) for the fully comprehensive user instructions.

The notebook is structured as follows:

I. Import libraries and spectral data from the **load_data.py** script.

II. Prepare the data for training: 
- Evaluate dEI from the intensity derivatives.
- Evaluate dEII from the ratio of the vacuum intensities to the experimental uncertainty. 
- Keep only the data points with energy loss < dEI and drop the points with higher energy losses.
- Calculate experimental central values and uncertainties by means of EWD.
- Add pseudo datapoints for energy loss > dEII.
- Put all training points together in one datafile `df_full`.

III. Initialize the neural network model: 
- Define the neural network architecture.
- Initiate placeholders for the training variables `x`, `y` and `sigma`.
- Create the MC replicas: output is an (Nin, Nrep)-dimensional dataframe that contains all the replicas to be used for training the neural network in the next step.

IV. Training the neural network.
The final part of the script is where the NN training is carried out, which is defined inside `function_train()`.
It creates a loop over the Nrep replicas to initiate a training session on each of the individual replicas in series. The script displays intermediate  results after each number of epochs defined by `display_step`. Once the maximum number of epochs had been reached, the optimal stopping point is determined by taking the absolute minimum of the validation cost and restoring the corresponding network parameters by means of the `Saver` function. It is also possible to make predictions on any input vector of choice by feeding  the vector `predict_x` to the network. 
The datafiles that are stored upon successfully executing this script are the following:
- `Prediction_k` contains the training x and y data of this replica, and the corresponding ZLP prediction.
- `Cost_k` contains the training and validation error for the k-th replica stored after each display step. 
- `Extrapolation_k` contains the arrays `predict_x` and the ZLP predictions made on these values. 


### predictions.ipynb
This script is used to analyse the predictions from the trained neural networks that have been stored in the text files indicated above. It is organised as follows:

I. Import libraries and spectral data from the **load_data.py** script.

II. Create dataframes with all individual spectra.
It organizes a datafile `original` which contains the intensity values for each of the original input spectra restricted to the region between `E_min` and `E_max`.

III. Load the result files and only select the ones that satisfy suitable post-fit selection criteria, such as the final error function being smaller than a certain threshold. 

IV. Matching of the predicted ZLP intensities with the original spectra.
At this step the code uses the function `matching()` to implement the matching procedure as described in the publication. 
If the user aims to extract the bandgap properties from the onset of the subtracted spectra, the `bandgap()` function can be used to fit a polynomial function to the onset region.
The results is a datafile `total_replicas` containing the predicted ZLP, the matched and the subtracted spectra for each of the Nrep replicas. This file is saved in `Data/results/replica_files` such that a user can retrieve them  at any time to calculate the statistical estimators such as prediction means and uncertainties. 

V. Evaluate the subtracted spectra.
In the last part, the code creates a  `mean_rep` file that contains all the median predictions and the upper and lower bounds of the 68\% confidence intervals for the predicted ZLP, matched spectra and the subtracted spectra, for each of the original recorded spectra originally given as an input. It outputs a graphical representation of the result. 


### predictions_pretrained_net.ipynb
This script, similar to the one previously described, can be executed stand-alone without the need to train again the neural networks. This means that step III in the previous section is now replaced by creating the model predictions from pre-trained net parameters, stored in the `Models` folder. 
This script can be executed provided that the model parameters corresponding to some previous training with the desired input settings are available in this folder.

