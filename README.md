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
Laurien Roest, Sabrya E. van Heijst, Luigi Maduro, Juan Rojo, and Sonia Conesa-Boj, [arXiv:2009.aaaaa](https://arxiv.org/abs/2009.00014).

as well as the [DOI:(...)](https://google.com/) associated to the code.

## Scientific publications

Up to now the **EELSfitter** code has been used in the following scientific publications:

- *Charting the low-loss region in Electron Energy Loss Spectroscopy with machine learning*, 
Laurien Roest, Sabrya E. van Heijst, Luigi Maduro, Juan Rojo, and Sonia Conesa-Boj, [arXiv:2009.aaaaa](https://arxiv.org/abs/2009.00014).

- *Fingerprinting 2H/3R Polytypism in WS2 Nanoflowers from Plasmons and Excitons to Phonons*,
Sabrya E. van Heijst, Mukai Masaki, E. Okunishi, H. Kurata ,Laurien Roest, Louis Arcolino Maduro, Juan Rojo, and Sonia Conesa-Boj, [arXiv:2009.aaaaa](https://arxiv.org/abs/2009.00014).

If you use **EELSfitter** in your own publication, we would be glad if you could let us know so that we could add your publication to this list.

## Installation and usage

Below we provide some information about the installation and usage of **EELSfitter**. The code developes a working Python v3.8 installation and relies on the [TensorFlow library](https://www.tensorflow.org/) v2.0.

Usage [this publication](https://arxiv.org/abs/2009.00014). 

### Data
In this folder, the raw spectrum files and the training results are stored.

### Models
Place where the neural network paramers and optimal configurations are saved, to be restored later in order to make predictions on pre-trained nets (see **predicitons_pretrained_net.ipynb**).

### Functions.py
File that contains a couple of functions that are used, *e.g.* for calculating means, errors and smoothing functions.

### Load_data.py
This file imports the raw spectra and returns the intensities, calibrates with the maximum intensity at zero energy loss, and returns both the original and normalized intensities. The output is two datasets, `df` and `df_vacuum` which contain the information on the in-sample and in-vacuum recorded spectra respectively. 
For each of the spectra the minimum and maximum value of the recorded energy loss need to be set manually in `Eloss_min` and `Eloss_max`.

### Fitter.ipynb
This script is used to run the neural network training on the data that was uploaded using **load_data.py**. 
It involves a number of pre-processing steps to determine the hyper-parameters $\Delta E_{\rm I}$ and $\Delta E_{\rm II}$ and tit automatically prepares and cuts the data before it is forwarded to the neural network to start the training.
We refer to the Appendix in [this publication](https://arxiv.org/abs/2009.00014) for the fully comprehensive user instructions.

The notebook is structured as follows:

- Import libraries and spectral data from the **load_data.py** script.
- Evaluate $\Delta E_{\rm I}$ from the intensity derivatives.
- Evaluate $\Delta E_{\rm II}$ from the ratio of the vacuum intensities to the experimental uncertainty. 
- Keep only the data points with $\Delta E \le \Delta E_{\rm I}$ and drop the points with higher energy losses.
- Calculate experimental central values and uncertainties by means of EWD.
- Add pseudo datapoints for $\Delta E \ge \Delta E_{\rm II}$







`predictions.ipynb` can be used to evaluate the predictions after training. 

`predictions_pretrained_net.ipynb` is similar to the Predictions.ipynb file, but here
one can import the network parameters of pre-trained nets to create predictions on any input array. 

The folders `Data`and `Models` contain the datasets, the results from training and the saved model 
parameters respectively. 


