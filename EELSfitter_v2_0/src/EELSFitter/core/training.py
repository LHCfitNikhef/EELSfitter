import numpy as np
import random
import os
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import datetime as dt
import torch.optim as optim
from sklearn.model_selection import train_test_split
from matplotlib import rc


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 22})
rc('text', usetex=True)

from ..plotting.hyperparameters import plot_dE1



class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) class. It uses the following architecture

    .. math::

       [n_i, 10, 15, 5, n_f],

    where :math:`n_i` and :math:`n_f` denote the number of input features and output target values respectively.

    Parameters
    ----------
    num_inputs: int
        number of input features
    num_outputs: int
        dimension of the target output.
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, 10)
        self.linear2 = nn.Linear(10, 15)
        self.linear3 = nn.Linear(15, 5)
        self.output = nn.Linear(5, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Propagates the input features ``x`` through the MLP.
        Parameters
        ----------
        x: torch.tensor
            input features

        Returns
        -------
        x: torch.tensor
            MLP outcome
        """
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.output(x)
        return x


def scale(inp, ab):
    """
    Rescale the training data to lie between 0.1 and 0.9

    Parameters
    ----------
    inp: numpy.ndarray, shape=(M,)
        training data to be rescaled, e.g. :math:`\Delta E`
    ab: numpy.ndarray, shape=(M,)
        scaling parameters, which can be find with :py:meth:`find_scale_var() <find_scale_var>`.
    Returns
    -------
    Rescaled training data
    """
    return inp * ab[0] + ab[1]


def find_scale_var(inp, min_out=0.1, max_out=0.9):
    """
    Computes the scaling parameters needed to rescale the training data to lie between ``min_out`` and ``max_out``.

    Parameters
    ----------
    inp: numpy.ndarray, shape=(M,)
        training data to be rescaled
    min_out: float
        lower limit. Set to 0.1 by default.
    max_out: float
        upper limit. Set to 0.9 by default

    Returns
    -------
    a, b: list
        list of rescaling parameters
    """
    a = (max_out - min_out) / (inp.max() - inp.min())
    b = min_out - a * inp.min()
    return [a, b]


def weight_reset(m):
    """
    Reset the weights and biases associated with the model ``m``.

    Parameters
    ----------
    m: MLP
        Model of type :py:meth:`MLP <MLP>`.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def loss_fn(output, target, error):
    r"""
    The loss function to train the ZLP takes the model ``output``, the raw spectrum ``target`` and
    the associated ``error``. The latter corresponds to the one sigma spread within a given cluster
    at fixed :math:`\Delta E`. It returns the cost function :math:`C_{\mathrm{ZLP}}^{(m)}` associated with the replica :math:`m` as

    .. math::

       C_{\mathrm{ZLP}}^{(m)} = \frac{1}{n_{E}}\sum_{k=1}^K \sum_{\ell_k=1}^{n_E^{(k)}} \frac{\left[ I^{(i_{m,k},
       j_{m,k})}(E_{\ell_k}) - I_{\rm ZLP}^{({\mathrm{ NN}})(m)} \left(E_{\ell_k},\ln \left( N_{\mathrm{ tot}}^{(i_{
       m,k},j_{m,k})}\right) \right) \right]^2}{\sigma^2_k \left( E_{\ell_k}\right) }.


    Parameters
    ----------
    output: torch.tensor
        Neural Network output
    target: torch.tensor
        Raw EELS spectrum
    error: torch.tensor
        Uncertainty on :math:`\log I_{\mathrm{EELS}}(\Delta E)`.

    Returns
    -------
    loss: torch.tensor
        Loss associated with the model ``output``.
    """
    loss = torch.mean(torch.square((output - target) / error))
    return loss


def smooth(data, window_len=10, window='hanning', keep_original=False):
    """
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters
    ----------
    data: numpy.ndarray, shape=(M,)
        The input signal
    window_len: int, optional
        The dimension of the smoothing window; should be an odd integer. Set to 10 by default.
    window: str, optional
        the type of window from ``"flat"``, '``"hanning"``, ``"hamming"``, ``"bartlett"``, ``"blackman"``
        ``"flat"`` window will produce a moving average smoothing.

    Returns
    -------
    smoothed_signal: numpy.ndarray, shape=(M,)
        The smoothed spectrum

    """
    window_len += (window_len + 1) % 2
    s = np.r_['-1', data[:, window_len - 1:0:-1], data, data[:, -2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    surplus_data = int((window_len - 1) * 0.5)
    smoothed_signal = np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=1, arr=s)[:,
           surplus_data:-surplus_data]
    return smoothed_signal


def smooth_clusters(image, clusters, window_len=None):
    """
    Smooth the spectrum for all clusters

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object
    clusters: numpy.ndarray, shape=(M,)

    window_len: int
        Smoothing window length

    Returns
    -------
    smoothed_clusters: numpy.ndarray, shape=(M,)
        The smoothed spectra for all clusters
    """
    smoothed_clusters = np.zeros((len(clusters)), dtype=object)
    for i in range(len(clusters)):
        smoothed_clusters[i] = smooth(clusters[i])
    return smoothed_clusters


def derivative_clusters(image, clusters):
    """
    Computes the derivative of the spectral image in each cluster.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object
    clusters: numpy.ndarray, shape=(M,)
        An array with size the number of clusters Each entry is a 2D array that contains all the spectra within the cluster.

    Returns
    -------
    der_clusters: numpy.ndarray, shape=(M,)
        Slope of the intensity spectrum for all clusters.
    """
    dx = image.ddeltaE
    der_clusters = np.zeros((len(clusters)), dtype=object)
    for i in range(len(clusters)):
        der_clusters[i] = (clusters[i][:, 1:] - clusters[i][:, :-1]) / dx
    return der_clusters


def find_min_de1(image, dy_dx, y_smooth):
    """
    Finds the location of the first local minimum of the intensity spectrum

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object
    dy_dx: numpy.ndarray, shape=(M,)
        Derivative of the intensity spectrum
    y_smooth: numpy.ndarray, shape=(M,)
        Smoothed intensity spectrum

    Returns
    -------
    pos_der: float
        Position of first local minimum
    """
    crossing = (dy_dx > 0)
    if not crossing.any():
        print("shouldn't get here")
        up = np.argmin(np.absolute(dy_dx)[np.argmax(y_smooth) + 1:]) + np.argmax(y_smooth) + 1
    else:
        up = np.argmax(crossing[np.argmax(y_smooth) + 1:]) + np.argmax(y_smooth) + 1
    pos_der = image.deltaE[up]
    return pos_der


def determine_de1(image, dy_dx_clusters, y_smooth_clusters, shift_de1=0.7, perc_de1=16, plot_de1=False, prob=True):
    """
    Computes the hyperparamter :math:`\Delta E_I` for every cluster in two ways:

    - Take a certain fraction of the location of the first local minimum. The fraction is tuned by ``shift_de1``.
    - Take :math:`\Delta E_I` such that only 16% of the replicas have a positive slope.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object.
    dy_dx_clusters: numpy.ndarray, shape=(M,)
        An array that contains an array for each cluster, which subsequently contains the slope of the spectrum at each
        pixel within the cluster.
    y_smooth_clusters: numpy.ndarray, shape=(M,)
        An array that contains an array for each cluster, which subsequently contains the smoothed spectrum at each
        pixel within the cluster.
    shift_de1: float, optional
        Shift the location of :math:`\Delta E_I` by a factor of ``shift_dE1`` w.r.t. to the first local minimum.
    perc_de1: float, optional
        Percentage of replicas that is allowed to have a positive slope at :math:`\Delta E_I`.
    plot_de1: bool, optional
        When set to `True`, the location of :math:`\Delta E_I` is plotted. Set to `False` by default.
    prob: bool, optional
        When set to `True`, the location of :math:`\Delta E_I` is determined by demanding that ``perc_de1``% of the EELS replicas should
        have a positive slope.

    Returns
    -------
    dE1_clusters: numpy.ndarray, shape=(M,)
        Array with the value of :math:`\Delta E_I` for each cluster.
    """

    # number of clusters
    n_clusters = len(y_smooth_clusters)

    # median EELS spectrum for each cluster
    y_smooth_clusters_avg = np.array([np.average(y_smooth_clusters[i], axis=0) for i in range(n_clusters)])

    # average slope of the EElS spectrum for each cluster
    dy_dx_avg = np.array([np.average(dy_dx_clusters[i], axis=0) for i in range(n_clusters)])

    # zeros of the first derivative for each cluster
    min_clusters = np.array([find_min_de1(image, dy_dx_avg[i], y_smooth_clusters_avg[i]) for i in range(n_clusters)])

    # find the replica such that 16% of the replicas have a steeper slope
    cl_high = 100 - perc_de1
    dy_dx_perc = np.array([np.nanpercentile(dy_dx_clusters[i], cl_high, axis=0) for i in range(n_clusters)])

    # find the index at which 16% of the replicas have a positive slope
    # and evaluate the corresponding value of deltaE.

    #TODO: not consistent: gives extra solutions sometimes depending on the clustering 29/07/21
    idx = np.argwhere(np.diff(np.sign(dy_dx_perc)))[:, 1][1::2]
    de1_prob = image.deltaE[idx]

    # find dE1 by taking a certain fraction of the location of the first local minimum.
    # The fraction is controlled by shift_dE1
    de1_shift = np.array([min_clusters[i] * shift_de1 for i in range(n_clusters)])

    # number of replicas per cluster
    n_rep_cluster = np.array([dy_dx_clusters[i].shape[0] for i in range(n_clusters)])

    # display the computed values of dE1 (both methods) together with the raw EELS spectrum
    if plot_de1:
        print("Plotting choice of dE1...")
        fig_1, fig_2 = plot_dE1(image, y_smooth_clusters, dy_dx_clusters, min_clusters, de1_prob, de1_shift, prob)
        fig_1.savefig(os.path.join(image.output_path, "der.pdf"))
        fig_2.savefig(os.path.join(image.output_path, "raw.pdf"))
        print("Plots stored in {}".format(image.output_path))
    # return the de1 selected by the bool prob
    if prob:
        return de1_prob
    else:
        return de1_shift


def filter_bad_bins(image, spectra):
    #TODO: add dimensions to spectra in docstring
    """
    Filter out experimental hiccups.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object.
    spectra:

    Returns
    -------
    output:
        Spectral image with experimental hiccups removed.
    """
    spectra_filtered = np.zeros(len(spectra), dtype=object)
    for j, spectra_cluster in enumerate(spectra):
        spectra_filtered_clustered = []
        for i, spectra_cluster_model in enumerate(spectra_cluster):
            hiccup_check = np.max(spectra_cluster_model[(image.deltaE > 1.0) & (image.deltaE < 3.0)])
            if hiccup_check > 2500:
                continue
            else:
                spectra_filtered_clustered.append(spectra_cluster_model)
        spectra_filtered_clustered = np.array(spectra_filtered_clustered)
        spectra_filtered[j] = spectra_filtered_clustered
    return spectra_filtered


def train_zlp_scaled(image, spectra, n_rep=500, n_epochs=30000, lr=1e-3, shift_dE1=0.7, shift_dE2=3,
                     path_to_models="models",
                     display_step=1000, bs_rep_num=0, perc_de1=16, plot_de1=False, prob=True):
    """
    Trains the ZLP on the (clustered) spectral image.

    One should train on two input features: the (scaled) log intensity :math:`\log I_{\mathrm{EELS}}` and the (scaled) :math:`\Delta E`.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object.
    spectra: numpy.ndarray, shape=(M,)
        An array of size ``n_clusters``, with each entry being a 2D array that contains all the spectra within the cluster.
    n_rep: int, optional
        Number of replicas to train on. Each replica consists of one spectrum from each cluster. Set to 500 by default.
    n_epochs: int, optional
        Number of epochs. Set to 30000 by default
    lr: float, optional
        Learning rate. Set to :math:`10^{-3}` by default.
    shift_dE1: float, optional
        Set to 0.7 by default. When given, :math:`\Delta E_I` is located at ``shift_dE1`` times the first local minimum in the spectrum.
    path_to_models: str, optional
        Path to where the trained models should be stored.
    display_step: int, optional
        Number of epochs after which to print output.
    bs_rep_num: int, optional
        For cluster users only: labels the parallel run.
    perc_de1: float, optional
        Percentage of replicas that is allowed to have a positive slope at :math:`\Delta E_I`.
    plot_de1: bool, optional
        When set to `True`, the location of :math:`\Delta E_I` is plotted. Set to `False` by default.
    prob: bool, optional
        When set to `True`, the location of :math:`\Delta E_I` is determined by demanding that 16% of the EELS replicas should
        have a positive slope.

    Examples
    --------
    To train the ZLP, one can run the following lines of code:

    >>> dm4_path = 'path to dm4 file'
    >>> im = SpectralImage.load_data(dm4_path)
    >>> training_data = im.get_cluster_spectra(conf_interval=1, signal='EELS')
    >>> train_zlp_scaled(im, training_data, n_clusters=5, n_rep=500, n_epochs=1000, path_to_models='path', display_step=10)
    """

    if display_step is None:
        print_progress = False
        display_step = 1E6
    else:
        print_progress = True

    if not os.path.exists(path_to_models):
        os.mkdir(path_to_models)

    # set all intensities smaller than 1 to 1
    for i in range(image.n_clusters):
        spectra[i][spectra[i] < 1] = 1

    loss_test_reps = np.zeros(n_rep)
    loss_train_reps = np.zeros(n_rep)

    sigma_clusters = np.zeros((image.n_clusters, image.l))  # shape = (n_clusters, n dE)
    for cluster in range(image.n_clusters):
        ci_low = np.nanpercentile(np.log(spectra[cluster]), 16, axis=0)
        ci_high = np.nanpercentile(np.log(spectra[cluster]), 84, axis=0)
        sigma_clusters[cluster, :] = np.absolute(ci_high - ci_low)

    wl1 = round(image.l / 20)
    wl2 = wl1 * 2

    # filter out bad bins
    #spectra_filtered = filter_bad_bins(image, spectra)

    # arrays with image.n_clusters sub-arrays, each having shape (n_spectra in cluster, n dE)
    #spectra_smooth = smooth_clusters(image, spectra_filtered, wl1)
    spectra_smooth = smooth_clusters(image, spectra, wl1)
    dy_dx = derivative_clusters(image, spectra_smooth)
    smooth_dy_dx = smooth_clusters(image, dy_dx, wl2)

    # hyperparameters: discard training data > dE1 and add pseudo data after dE2
    dE1 = determine_de1(image, smooth_dy_dx, spectra_smooth, shift_dE1, perc_de1, plot_de1, prob)
    dE2 = shift_dE2 * dE1

    if print_progress:
        print("dE1 & dE2:", np.round(dE1, 3), dE2)

    # rescale the dE features to [0.1, 0.9]
    ab_deltaE = find_scale_var(image.deltaE)
    deltaE_scaled = scale(image.deltaE, ab_deltaE)

    # collect all spectra from all clusters into a single array
    all_spectra = np.empty((0, image.l))
    for i in range(len(spectra)):
        all_spectra = np.append(all_spectra, spectra[i], axis=0)

    # the total integrated intensity for all pixels and rescale
    int_log_I = np.log(np.sum(all_spectra, axis=1)).flatten()
    ab_int_log_I = find_scale_var(int_log_I)
    del all_spectra

    path_scale_var = os.path.join(path_to_models, "scale_var.txt")
    if not os.path.exists(path_scale_var):
        np.savetxt(path_scale_var, ab_int_log_I)

    path_dE1 = os.path.join(path_to_models, "dE1.txt")
    if not os.path.exists(path_dE1):
        np.savetxt(path_dE1, np.vstack((image.clusters, dE1)))

    # path to file that keeps track of the cost function after training for each replica
    train_cost_path = os.path.join(path_to_models, "costs_train_{}.txt".format(bs_rep_num))
    test_cost_path = os.path.join(path_to_models, "costs_test_{}.txt".format(bs_rep_num))

    for i in range(n_rep):
        save_idx = i + n_rep * bs_rep_num - n_rep + 1
        nn_rep_path = os.path.join(path_to_models, "nn_rep_{}".format(save_idx))

        training_report_path = os.path.join(path_to_models, 'rep_{}'.format(save_idx))
        if not os.path.exists(training_report_path):
            os.makedirs(training_report_path)

        if print_progress: print("Started training on replica number {}".format(i) + ", at time ", dt.datetime.now())
        data_y = np.empty((0, 1))
        data_x = np.empty((0, 2))
        data_sigma = np.empty((0, 1))

        for cluster in range(image.n_clusters):
            n_cluster = len(spectra[cluster])  # number of spectra in a cluster

            # select a random spectrum from the cluster
            idx = random.randint(0, n_cluster - 1)

            # create slicing indices for the regions I, II and III
            select1 = len(image.deltaE[image.deltaE < dE1[cluster]])
            select2 = len(image.deltaE[image.deltaE > dE2[cluster]])

            # Only use the log intensity up to dE1 as data. Do this for a random spectrum for each cluster.
            data_y = np.append(data_y, np.log(spectra[cluster][idx][:select1]))
            # The log intensity is set to zero in region III
            data_y = np.append(data_y, np.zeros(select2))

            # pseudo_x contains the two input features: log integrated I and deltaE
            features = np.ones((select1 + select2, 2))
            features[:select1, 0] = deltaE_scaled[:select1]
            features[-select2:, 0] = deltaE_scaled[-select2:]

            # rescale the integrated intensity
            int_log_I_idx_scaled = scale(np.log(np.sum(spectra[cluster][idx])), ab_int_log_I)
            features[:, 1] = int_log_I_idx_scaled

            data_x = np.concatenate((data_x, features))
            data_sigma = np.append(data_sigma, sigma_clusters[cluster][:select1])
            data_sigma = np.append(data_sigma, 0.8 * np.ones(select2))

        model = MLP(num_inputs=2, num_outputs=1)
        model.apply(weight_reset)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_x, test_x, train_y, test_y, train_sigma, test_sigma = train_test_split(data_x, data_y, data_sigma,
                                                                                     test_size=0.4)

        n_test = len(test_x)
        n_train = len(train_x)

        test_x = test_x.reshape(n_test, 2)
        test_y = test_y.reshape(n_test, 1)
        train_x = train_x.reshape(n_train, 2)
        train_y = train_y.reshape(n_train, 1)
        train_sigma = train_sigma.reshape(n_train, 1)
        test_sigma = test_sigma.reshape(n_test, 1)

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        train_sigma = torch.from_numpy(train_sigma)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)
        test_sigma = torch.from_numpy(test_sigma)

        # store the test and train loss per epoch
        loss_test_n = []
        loss_train_n = []

        n_stagnant = 0
        n_stagnant_max = 5
        for epoch in range(1, n_epochs + 1):

            # set the model to training mode
            model.train()
            output = model(train_x.float())
            loss_train = loss_fn(output, train_y, train_sigma)
            loss_train_n.append(loss_train.item())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # set the model to evaluation mode
            model.eval()
            with torch.no_grad():
                output_test = model(test_x.float())
                loss_test = loss_fn(output_test, test_y, test_sigma)
                loss_test_n.append(loss_test.item())
                if epoch % display_step == 0 and print_progress:
                    print('Rep {}, Epoch {}, Training loss {}, Testing loss {}'.format(i, epoch,
                                                                                       round(loss_train.item(), 3),
                                                                                       round(loss_test_n[epoch - 1], 3)))

                # update the test and train loss of the replica
                loss_test_reps[i] = loss_test.item()
                loss_train_reps[i] = loss_train.item()

        # save the model state when the maximum number of allowed epochs has been reached
        torch.save(model.state_dict(), nn_rep_path)

        # make a training report for each replica
        training_report(training_report_path, loss_train_n, loss_test_n)

    with open(train_cost_path, "w") as text_file:
        for item in loss_train_reps:
            text_file.write("%s\n" % item)

    with open(test_cost_path, "w") as text_file:
        for item in loss_test_reps:
            text_file.write("%s\n" % item)


def training_report(path, loss_train, loss_test):
    """
    Produce a training report: evolution of the training and validation loss per epoch

    Parameters
    ----------
    path: str
        location where to store the evolution plots
    loss_train: numpy.ndarray, shape=(M,)
        Training loss
    loss_test: numpy.ndarray, shape=(M,)
        Validation loss
    """
    training_loss_path = os.path.join(path, 'training_loss.txt')
    with open(training_loss_path, 'w') as f:
        for item in loss_train:
            f.write("%s\n" % item)

    test_loss_path = os.path.join(path, 'test_loss.txt')
    with open(test_loss_path, 'w') as f:
        for item in loss_test:
            f.write("%s\n" % item)

    fig, ax = plt.subplots(figsize=(1.1*10, 1.1*6))
    plt.plot(loss_train, label=r"$\rm{Training\;loss}$")
    plt.plot(loss_test, label=r"$\rm{Validation\;loss}$")
    plt.xlabel(r"$\rm{Epochs}$")
    plt.ylabel(r"$\rm{Loss}$")
    plt.legend(loc='best', frameon=False)
    fig.savefig(os.path.join(path, 'loss.pdf'))


def ewd(x, nbins):
    #TODO: what is the purpose of this function?
    """
    Apply Equal Width Discretization (EWD) to ``x`` to determine variances

    Parameters
    ----------
    x: numpy.ndarray, shape=(M,)
        input data
    nbins: int
        number of bins

    Returns
    -------
    cuts1: numpy.ndarray, shape=(M,)
    cuts2: numpy.ndarray, shape=(M,)


    """

    cuts1, cuts2 = pd.cut(x, nbins, retbins=True)

    return cuts1, cuts2


def CI_high(data, confidence=0.68):
    """
    Computes the upper 68% confidence level (by default) of ``data``.

    Parameters
    ----------
    data: numpy.ndarray, shape=(M,)
        arbitrary array
    confidence: float
        Confidence level

    Returns
    -------
    high_a: float
        confidence level set by ``confidence``.
    """

    a = np.array(data)
    n = len(a)
    b = np.sort(data)

    highest = np.int((1 - (1 - confidence) / 2) * n)
    high_a = b[highest]

    return high_a


def CI_low(data, confidence=0.68):
    """
    Computes the lower 68% confidence level (by default) of ``data``.

    Parameters
    ----------
    data: numpy.ndarray, shape=(M,)
        arbitrary array
    confidence: float
        Confidence level

    Returns
    -------
    low_a: float
        confidence level set by ``confidence``.
    """

    a = np.array(data)
    n = len(a)
    b = np.sort(data)
    lowest = np.int(((1 - confidence) / 2) * n)
    low_a = b[lowest]

    return low_a

def get_mean(data):
    #TODO: trivial function, remove
    """
    Returns the mean of ``data``.

    Parameters
    ----------
    data: numpy.ndarray, shape=(M,)
        data to average
    Returns
    -------
    mean: numpy.ndarray, shape=(M,)
        The mean of ``data``.
    """
    mean = np.mean(data)
    return mean
