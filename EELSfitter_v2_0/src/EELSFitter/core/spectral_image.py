import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import math
from scipy.fftpack import next_fast_len
import logging
from ncempy.io import dm
import os
import copy
import warnings
import torch
import bz2
import pickle
import _pickle as cPickle
from matplotlib import rc
import re

from ..clustering.k_means_clustering import k_means
from .training import MLP, train_zlp_scaled
from ..plotting.training_perf import plot_cost_dist


_logger = logging.getLogger(__name__)


class SpectralImage:
    """
    The spectral image class that provides several tools to analyse spectral images with the zero-loss peak
    subtracted.

    Parameters
    ----------
    data: numpy.ndarray, shape=(M,N,L)
        Array containing the 3-D spectral image. The axes correspond to the x-axis (M), y-axis (N) and energy-loss (L).
    deltadeltaE: float
        bin width in energy loss spectrum
    pixelsize: numpy.ndarray, shape=(2,), optional
        width of pixels
    beam_energy: float, optional
        Energy of electron beam in eV
    collection_angle: float, optional
        Collection angle of STEM in rad
    name: str, optional
        Title of the plots
    dielectric_function_im_avg
        average dielectric function for each pixel
    dielectric_function_im_std
        standard deviation of the dielectric function at each energy per pixel
    S_s_avg
        average surface scattering distribution for each pixel
    S_s_std
        standard deviation of the surface scattering distribution at each energy for each pixel
    thickness_avg
        average thickness for each pixel
    IEELS_avg
        average bulk scattering distribution for each pixel
    IEELS_std
        standard deviation of the bulk scattering distribution at each energy for each pixel
    cluster_data: numpy.ndarray, shape=(M,)
        filled with 2D numpy arrays. If save_as_attribute set to True, the cluster data is also saved as attribute
    deltaE: numpy.ndarray, shape=(M,)
        shifted array of energy losses such that the zero point corresponds to the point of highest intensity.
    x_axis: numpy.ndarray, shape=(M,)
        x-axis of the spectral image
    y_axis: numpy.ndarray, shape=(M,)
        y-axis of the spectral image
    clusters: numpy.ndarray, shape=(M,)
        cluster means of each cluster
    clustered: numpy.ndarray, shape=(M,)
        A 2D array containing the index of the cluster to which each pixel belongs
    output_path: str, optional
        Location where to store plots

    Examples
    --------
    An example how to train and anlyse a spectral image::

        dm4_path = 'path to dm4 file'
        im = SpectralImage.load_data(dm4_path)
        im.train_zlp(n_clusters=n_clusters,
                 n_rep=n_rep,
                 n_epochs=n_epochs,
                 bs_rep_num=bs_rep_num,
                 path_to_models=path_to_models,
                 display_step=display_step)

    """

    #  signal names
    DIELECTRIC_FUNCTION_NAMES = ['dielectric_function', 'dielectricfunction', 'dielec_func', 'die_fun', 'df', 'epsilon']
    EELS_NAMES = ['electron_energy_loss_spectrum', 'electron_energy_loss', 'EELS', 'EEL', 'energy_loss', 'data']
    IEELS_NAMES = ['inelastic_scattering_energy_loss_spectrum', 'inelastic_scattering_energy_loss',
                   'inelastic_scattering', 'IEELS', 'IES']
    ZLP_NAMES = ['zeros_loss_peak', 'zero_loss', 'ZLP', 'ZLPs', 'zlp', 'zlps']
    THICKNESS_NAMES = ['t', 'thickness', 'thick', 'thin']
    POOLED_ADDITION = ['pooled', 'pool', 'p', '_pooled', '_pool', '_p']

    # meta data names
    COLLECTION_ANGLE_NAMES = ["collection_angle", "col_angle", "beta"]
    BEAM_ENERGY_NAMES = ["beam_energy", "beam_E", "E_beam", "E0", "E_0"]

    m_0 = 5.1106E5  # eV, electron rest mass
    a_0 = 5.29E-11  # m, Bohr radius
    h_bar = 6.582119569E-16  # eV/s
    c = 2.99792458E8  # m/s

    def __init__(self, data, deltadeltaE, pixelsize=None, beam_energy=None, collection_angle=None, name=None,
                 dielectric_function_im_avg=None, dielectric_function_im_std=None, S_s_avg=None, S_s_std=None,
                 thickness_avg=None, thickness_std=None, IEELS_avg=None, IEELS_std=None, clusters=None, clustered=None, cluster_data=None, deltaE=None, x_axis=None, y_axis = None,
                 ZLP_models = None, scale_var_deltaE=None, crossings_E=None, crossings_n=None, data_smooth=None, dE1=None, n=None, pooled=None,
                 scale_var_log_sum_I = None, output_path=None, **kwargs):

        self.data = data
        self.ddeltaE = deltadeltaE
        self.deltaE = self.determine_deltaE()

        if pixelsize is not None:
            # Input pixelsize is assumed to be in micron, this converts it too nanometer
            self.pixelsize = pixelsize * 1E3
        self.calc_axes()
        if beam_energy is not None:
            self.beam_energy = beam_energy
        if collection_angle is not None:
            self.collection_angle = collection_angle
        if name is not None:
            self.name = name

        self.dielectric_function_im_avg = dielectric_function_im_avg
        self.dielectric_function_im_std = dielectric_function_im_std
        self.S_s_avg = S_s_avg
        self.S_s_std = S_s_std
        self.thickness_avg = thickness_avg
        self.thickness_std = thickness_std
        self.IEELS_avg = IEELS_avg
        self.IEELS_std = IEELS_std
        self.clusters = clusters
        self.clustered = clustered
        self.cluster_data = cluster_data
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.ZLP_models = ZLP_models
        self.scale_var_deltaE = scale_var_deltaE
        self.crossings_E = crossings_E
        self.crossings_n = crossings_n
        self.data_smooth = data_smooth
        self.dE1 = dE1
        self.n = n
        self.pooled = pooled
        self.scale_var_log_sum_I = scale_var_log_sum_I
        if output_path is None:
            self.output_path = os.getcwd()
        else:
            self.output_path = output_path

    def save_image(self, filename):
        """
        Function to save image, including all attributes, in pickle (.pkl) format. Image will be saved \
        at indicated location and name in filename input.

        Parameters
        ----------
        filename : str
            path to save location plus filename. If it does not end on ".pkl", ".pkl" will be added.
        """
        if filename[-4:] != '.pkl':
            filename = filename + '.pkl'
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_compressed_image(self, filename):
        """
        Function to save image, including all attributes, in compressed pickle (.pbz2) format. Image will \
            be saved at location ``filename``. Advantage over :py:meth:`save_image() <EELSFitter.core.spectral_image.SpectralImage.save_image>` is that \
            the saved file has a reduced file size, disadvantage is that saving and reloading the image \
            takes significantly longer.


        Parameters
        ----------
        filename : str
            path to save location plus filename. If it does not end on ".pbz2", ".pbz2" will be added.

        """
        if filename[-5:] != '.pbz2':
            filename = filename + '.pbz2'
        self.compressed_pickle(filename, self)

    @staticmethod
    # Pickle a file and then compress it into a file with extension 
    def compressed_pickle(title, data):
        """
        Saves ``data`` at location ``title`` as compressed pickle.
        """
        with bz2.BZ2File(title, 'w') as f:
            cPickle.dump(data, f)

    @staticmethod
    def decompress_pickle(file):
        """
        Opens, decompresses and returns the pickle file at location ``file``.

        Parameters
        ----------
        file: str
            location where the pickle file is stored

        Returns
        -------
        data: SpectralImage
        """
        data = bz2.BZ2File(file, 'rb')
        data = cPickle.load(data)
        return data

    # %%GENERAL FUNCTIONS

    # %%PROPERTIES
    @property
    def l(self):
        """Returns length of :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object,
        i.e. num energy loss bins"""
        return self.data.shape[2]

    @property
    def image_shape(self):
        """Returns 2D-shape of :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object"""
        return self.data.shape[:2]

    @property
    def shape(self):
        """Returns 3D-shape of :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object"""
        return self.data.shape

    @property
    def n_clusters(self):
        """Returns the number of clusters in the :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object."""
        return len(self.clusters)

    @property
    def n_spectra(self):
        """
        Returns the number of spectra present in :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object

        Returns
        -------
        nspectra: int
            number of spectra in spectral image"""
        nspectra = np.product(self.image_shape)
        return nspectra

    @classmethod
    def load_data(cls, path_to_dmfile, load_additional_data=False):
        """
        Load the .dm4 spectral image and return a :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance.

        Parameters
        ----------
        path_to_dmfile: str
            location of .dm4 file
        load_additional_data: bool, optional
            Default is `False`. If `True`

        Returns
        -------
        SpectralImage
            :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance of the dm4 file
        """
        dmfile_tot = dm.fileDM(path_to_dmfile)
        additional_data = []
        for i in range(dmfile_tot.numObjects - dmfile_tot.thumbnail * 1):
            dmfile = dmfile_tot.getDataset(i)
            if dmfile['data'].ndim == 3:
                dmfile = dmfile_tot.getDataset(i)
                data = np.swapaxes(np.swapaxes(dmfile['data'], 0, 1), 1, 2)
                if not load_additional_data:
                    break
            elif load_additional_data:
                additional_data.append(dmfile_tot.getDataset(i))
            if i == dmfile_tot.numObjects - dmfile_tot.thumbnail * 1 - 1:
                if (len(additional_data) == i + 1) or not load_additional_data:
                    print("No spectral image detected")
                    dmfile = dmfile_tot.getDataset(0)
                    data = dmfile['data']

        ddeltaE = dmfile['pixelSize'][0]
        pixelsize = np.array(dmfile['pixelSize'][1:])
        energyUnit = dmfile['pixelUnit'][0]
        ddeltaE *= cls.get_prefix(energyUnit, 'eV')
        pixelUnit = dmfile['pixelUnit'][1]
        pixelsize *= cls.get_prefix(pixelUnit, 'm')

        image = cls(data, ddeltaE, pixelsize=pixelsize, name=path_to_dmfile[:-4])
        if load_additional_data:
            image.additional_data = additional_data
        return image

    @classmethod
    def load_spectral_image(cls, path_to_pickle):
        """
        Loads :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance from a pickled file.

        Parameters
        ----------
        path_to_pickle : str
            path to the pickled image file.

        Raises
        ------
        ValueError
            If path_to_pickle does not end on the desired format .pkl.
        FileNotFoundError
            If path_to_pickle does not exists.

        Returns
        -------
        SpectralImage
            :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object (i.e. including all attributes) loaded from pickle file.

        """
        if path_to_pickle[-4:] != '.pkl':
            raise ValueError("please provide a path to a pickle file containing a Spectral_image class object.")
        if not os.path.exists(path_to_pickle):
            raise FileNotFoundError('pickled file: ' + path_to_pickle + ' not found')
        with open(path_to_pickle, 'rb') as pickle_im:
            image = pickle.load(pickle_im)
        return image

    @classmethod
    def load_compressed_Spectral_image(cls, path_to_compressed_pickle):
        """
        Loads spectral image from a compressed pickled file. This will take longer than loading from non compressed pickle.

        Parameters
        ----------
        path_to_compressed_pickle : str
            path to the compressed pickle image file.

        Raises
        ------
        ValueError
            If path_to_compressed_pickle does not end on the desired format .pbz2.
        FileNotFoundError
            If path_to_compressed_pickle does not exists.

        Returns
        -------
        image : SpectralImage
             :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance loaded from the compressed pickle file.
        """
        if path_to_compressed_pickle[-5:] != '.pbz2':
            raise ValueError(
                "please provide a path to a compressed .pbz2 pickle file containing a Spectrall_image class object.")
        if not os.path.exists(path_to_compressed_pickle):
            raise FileNotFoundError('pickled file: ' + path_to_compressed_pickle + ' not found')

        image = cls.decompress_pickle(path_to_compressed_pickle)
        return image

    def set_n(self, n, n_background=None):
        """
        Sets value of refractive index for the image as attribute self.n. If unclusered, n will be an \
            array of length one, otherwise it is an array of len n_clusters. If n_background is defined, \
            the cluster with the lowest thickness (cluster 0) will be assumed to be the vacuum/background, \
            and gets the value of the background refractive index.
            
        If there are more specimen present in the image, it is wise to check by hand what cluster belongs \
            to what specimen, and set the values by running::

             image.n[cluster_i] = n_i

        Parameters
        ----------
        n : float
            refractive index of sample.
        n_background : float, optional
            if defined: the refractive index of the background/vacuum. This value will automatically be \
            assigned to pixels belonging to the thinnest cluster.
        """
        if type(n) == float or type(n) == int:
            self.n = np.ones(self.n_clusters) * n
            if n_background is not None:
                # assume thinnest cluster (=cluster 0) is background
                self.n[0] = n_background
        elif len(n) == self.n_clusters:
            self.n = n

    def determine_deltaE(self):
        """
        Determines the energy losses of the spectral image, based on the bin width of the energy loss.
        It shifts the ``self.deltaE`` attribute such that the zero point corresponds with the point of highest
        intensity.

        Returns
        -------
        deltaE: numpy.ndarray, shape=(M,)
            Array of :math:`\Delta E` values
        """
        data_avg = np.average(self.data, axis=(0, 1))
        ind_max = np.argmax(data_avg)
        deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l - ind_max - 1) * self.ddeltaE, self.l)
        return deltaE

    def calc_axes(self):
        """
        Determines the  x_axis and y_axis of the spectral image. Stores them in ``self.x_axis`` and ``self.y_axis`` respectively.
        """
        self.y_axis = np.linspace(0, self.image_shape[0] - 1, self.image_shape[0])
        self.x_axis = np.linspace(0, self.image_shape[1] - 1, self.image_shape[1])
        if self.pixelsize is not None:
            self.y_axis *= self.pixelsize[0]
            self.x_axis *= self.pixelsize[1]

    def get_pixel_signal(self, i, j, signal='EELS'):
        """
        Retrieves the spectrum at pixel (``i``, ``j``).

        Parameters
        ----------
        i: int
            x-coordinate of the pixel
        j: int
            y-coordinate of the pixel
        signal: str, optional
            The type of signal that is requested, should comply with the defined names. Set to `EELS` by default.

        Returns
        -------
        signal : numpy.ndarray, shape=(M,)
            Array with the requested signal from the requested pixel
        """

        if signal in self.EELS_NAMES:
            return np.copy(self.data[i, j, :])
        elif signal == "pooled":
            return np.copy(self.pooled[i, j, :])
        elif signal in self.DIELECTRIC_FUNCTION_NAMES:
            return np.copy(self.dielectric_function_im_avg[i, j, :])
        else:
            print("no such signal", signal, ", returned general EELS signal.")
            return np.copy(self.data[i, j, :])

    def get_image_signals(self, signal='EELS'):

        if signal in self.EELS_NAMES:
            return np.copy(self.data)
        elif signal == "pooled":
            return np.copy(self.pooled)
        elif signal in self.DIELECTRIC_FUNCTION_NAMES:
            return np.copy(self.dielectric_function_im_avg)
        else:
            print("no such signal", signal, ", returned general EELS data.")
            return np.copy(self.data)

    def get_cluster_spectra(self, conf_interval=1, clusters=None, signal="EELS"):
        """
        Returns a clustered spectral image.

        Parameters
        ----------
        conf_interval : float, optional
            The ratio of spectra returned. The spectra are selected based on the 
            based_on value. The default is 1.
        clusters : list of ints, optional
            list with all the cluster labels.
        signal: str, optional
            Description of signal, ``"EELS"`` by default.

        Returns
        -------
        cluster_data : numpy.ndarray, shape=(M,)
            An array with size equal to the number of clusters. Each entry is a 2D array that contains all the spectra within that cluster.
        """

        if clusters is None:
            clusters = range(self.n_clusters)

        integrated_int = np.sum(self.data, axis=2)
        cluster_data = np.zeros(len(clusters), dtype=object)

        j = 0
        for i in clusters:
            data_cluster = self.get_image_signals(signal)[self.clustered == i]
            if conf_interval < 1:
                intensities_cluster = integrated_int[self.clustered == i]
                arg_sort_int = np.argsort(intensities_cluster)
                ci_lim = round((1 - conf_interval) / 2 * intensities_cluster.size)
                data_cluster = data_cluster[arg_sort_int][ci_lim:-ci_lim]
            cluster_data[j] = data_cluster
            j += 1

        return cluster_data

    # %%METHODS ON SIGNAL

    def cut(self, E1=None, E2=None, in_ex="in"):
        """
        Cuts the spectral image at ``E1`` and ``E2`` and keeps only the part in between.

        Parameters
        ----------
        E1 : float, optional
            lower cut. The default is ``None``, which means no cut is applied.
        E2 : float, optional
            upper cut. The default is ``None``, which means no cut is applied.

        """
        if (E1 is None) and (E2 is None):
            raise ValueError("To cut energy spectra, please specify minimum energy E1 and/or maximum energy E2.")
        if E1 is None:
            E1 = self.deltaE.min() - 1
        if E2 is None:
            E2 = self.deltaE.max() + 1
        if in_ex == "in":
            select = ((self.deltaE >= E1) & (self.deltaE <= E2))
        else:
            select = ((self.deltaE > E1) & (self.deltaE < E2))
        self.data = self.data[:, :, select]
        self.deltaE = self.deltaE[select]

    def cut_image(self, range_width, range_height):
        """
        Cuts the spectral image

        Parameters
        ----------
        range_width: numpy.ndarray, shape=(2,)
            Contains the horizontal selection cut
        range_height: numpy.ndarray, shape=(2,)
            Contains the vertical selection cut
        """
        self.data = self.data[range_height[0]:range_height[1], range_width[0]:range_width[1]]
        self.y_axis = self.y_axis[range_height[0]:range_height[1]]
        self.x_axis = self.x_axis[range_width[0]:range_width[1]]

    def smooth(self, window_len=10, window='hanning', keep_original=False):
        """
        Smooth the data using a window length ``window_len``.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        
        Parameters
        ----------
        window_len: int, optional
            The dimension of the smoothing window; should be an odd integer.
        window: str, optional
            the type of window from ``"flat"``, ``"hanning"``,  ``"bartlett"``, ``"blackman"``.
            ``"flat"`` will produce a moving average smoothing.
        """

        window_len += (window_len + 1) % 2
        s = np.r_['-1', self.data[:, :, window_len - 1:0:-1], self.data, self.data[:, :, -2:-window_len - 1:-1]]

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        surplus_data = int((window_len - 1) * 0.5)
        if keep_original:
            self.data_smooth = np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=2, arr=s)[
                               :, :, surplus_data:-surplus_data]
        else:
            self.data = np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=2, arr=s)[:, :,
                        surplus_data:-surplus_data]

    def deconvolute(self, i, j, ZLP, signal='EELS'):
        """
        Removes the effect of plural scatterings from the spectrum at pixel (``i``, ``j``).

        Parameters
        ----------
        i: int
            x-coordinate of the pixel
        j: int
            y-coordinate of the pixel
        ZLP: numpy.ndarray, shape=(M,)
            zero-loss peak at pixel (``i``, ``j``) of length M
        signal: str, optional
            Type of signal, ''"EELS"'' by default.

        Returns
        -------
        output: numpy.ndarray, shape=(M,)
            deconvoluted spectrum.
        """
        y = self.get_pixel_signal(i, j, signal)
        r = 3  # Drude model, can also use estimation from exp. data
        A = y[-1]
        n_times_extra = 2
        sem_inf = next_fast_len(n_times_extra * self.l)

        y_extrp = np.zeros(sem_inf)
        y_ZLP_extrp = np.zeros(sem_inf)
        x_extrp = np.linspace(self.deltaE[0] - self.l * self.ddeltaE,
                              sem_inf * self.ddeltaE + self.deltaE[0] - self.l * self.ddeltaE, sem_inf)

        x_extrp = np.linspace(self.deltaE[0], sem_inf * self.ddeltaE + self.deltaE[0], sem_inf)

        y_ZLP_extrp[:self.l] = ZLP
        y_extrp[:self.l] = y
        x_extrp[:self.l] = self.deltaE[-self.l:]

        y_extrp[self.l:] = A * np.power(1 + x_extrp[self.l:] - x_extrp[self.l], -r)

        x = x_extrp
        y = y_extrp
        y_ZLP = y_ZLP_extrp

        z_nu = CFT(x, y_ZLP)
        i_nu = CFT(x, y)
        abs_i_nu = np.absolute(i_nu)
        N_ZLP = 1  # scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)

        s_nu = N_ZLP * np.log(i_nu / z_nu)
        j1_nu = z_nu * s_nu / N_ZLP
        S_E = np.real(iCFT(x, s_nu))
        s_nu_nc = s_nu
        s_nu_nc[500:-500] = 0
        S_E_nc = np.real(iCFT(x, s_nu_nc))
        J1_E = np.real(iCFT(x, j1_nu))

        return J1_E[:self.l]

    def pool(self, n_p):
        """
        Pools the spectral image using a squared window of size ``n_p''.

        Parameters
        ----------
        n_p: int
            pooling parameter
        """
        if n_p % 2 == 0:
            print("Unable to pool with even number " + str(n_p) + ", continuing with n_p=" + str(n_p + 1))
            n_p += 1
        pooled = np.zeros(self.shape)
        n_p_border = int(math.floor(n_p / 2))
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                min_x = max(0, i - n_p_border)
                max_x = min(self.image_shape[0], i + 1 + n_p_border)
                min_y = max(0, j - n_p_border)
                max_y = min(self.image_shape[1], j + 1 + n_p_border)
                pooled[i, j] = np.average(np.average(self.data[min_x:max_x, min_y:max_y, :], axis=1), axis=0)
        self.pooled = pooled

    # METHODS ON ZLP
    # CALCULATING ZLPs FROM PRETRAINDED MODELS

    def calc_zlps_matched(self, i, j, signal='EELS', **kwargs):
        """
        Returns the shape-(M, N) array of matched ZLP model predictions at pixel (``i``, ``j``) after training.
        M and N correspond to the number of model predictions and :math:`\Delta E` s respectively.

        Parameters
        ----------
        i: int
            horizontal pixel.
        j: int
            vertical pixel.
        signal: str, bool
            Description of signal type. Set to ``"EELS"`` by default.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        predictions: numpy.ndarray, shape=(M, N)
            The matched ZLP predictions at pixel (``i``, ``j``).
        """

        def matching(signal, gen_i_ZLP, dE1):
            """
            Apply the matching to the subtracted spectrum.

            Parameters
            ----------
            signal: numpy.ndarray, shape=(M,)
                Raw spectrum of length M
            gen_i_ZLP: numpy.ndarray, shape=(M,)
                Trained ZLP
            dE1: float
                Value of the hyperparameter :math:`\Delta E_I`

            Returns
            -------
            output: numpy.ndarray, shape=(M,)
                Matched ZLP model
            """
            dE0 = dE1 * 0.85
            dE2 = dE1 * 3
            delta = (dE1 - dE0) / 10

            factor_NN = 1 / (1 + np.exp(
                -(self.deltaE[(self.deltaE < dE1) & (self.deltaE >= dE0)] - (dE0 + dE1) / 2) / delta))
            factor_dm = 1 - factor_NN

            range_0 = signal[self.deltaE < dE0]
            range_1 = gen_i_ZLP[(self.deltaE < dE1) & (self.deltaE >= dE0)] * factor_NN + signal[
                (self.deltaE < dE1) & (self.deltaE >= dE0)] * factor_dm
            range_2 = gen_i_ZLP[(self.deltaE >= dE1) & (self.deltaE < 3 * dE2)]
            range_3 = gen_i_ZLP[(self.deltaE >= 3 * dE2)] * 0
            totalfile = np.concatenate((range_0, range_1, range_2, range_3), axis=0)

            totalfile = np.minimum(totalfile, signal)
            return totalfile

        ZLPs_gen = self.calc_zlps(i, j, signal, **kwargs)

        count = len(ZLPs_gen)
        ZLPs = np.zeros((count, self.l))  # np.zeros((count, len_data))

        signal = self.get_pixel_signal(i, j, signal)
        cluster = self.clustered[i, j]

        dE1 = self.dE1[1, int(cluster)]
        for k in range(count):
            predictions = ZLPs_gen[k]
            ZLPs[k, :] = matching(signal, predictions, dE1)  # matching(energies, np.exp(mean_k), data)
        return ZLPs

    def calc_zlps(self, i, j, signal='EELS', **kwargs):
        """
        Returns the shape-(M, N) array of ZLP model predictions at pixel (``i``, ``j``) after training, where
        M and N correspond to the number of model predictions and :math:`\Delta E` s respectively.

        Parameters
        ----------
        i: int
            horizontal pixel.
        j: int
            vertical pixel.
        signal: str, bool
            Description of signal type. Set to ``"EELS"`` by default.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        predictions: numpy.ndarray, shape=(M, N)
            The ZLP predictions at pixel (``i``, ``j``).

        """
        # Definition for the matching procedure
        signal = self.get_pixel_signal(i, j, signal)

        if self.ZLP_models is None:
            try:
                self.load_zlp_models(**kwargs)
            except:
                self.load_zlp_models()

        count = len(self.ZLP_models)

        predictions = np.zeros((count, self.l))  # np.zeros((count, len_data))

        if self.scale_var_deltaE is None:
            self.scale_var_deltaE = find_scale_var(self.deltaE)

        if self.scale_var_log_sum_I is None:
            all_spectra = self.data
            all_spectra[all_spectra < 1] = 1
            int_log_I = np.log(np.sum(all_spectra, axis=2)).flatten()
            self.scale_var_log_sum_I = find_scale_var(int_log_I)
            del all_spectra

        log_sum_I_pixel = np.log(np.sum(signal))
        predict_x_np = np.zeros((self.l, 2))
        predict_x_np[:, 0] = scale(self.deltaE, self.scale_var_deltaE)
        predict_x_np[:, 1] = scale(log_sum_I_pixel, self.scale_var_log_sum_I)

        predict_x = torch.from_numpy(predict_x_np)

        for k in range(count):
            model = self.ZLP_models[k]
            with torch.no_grad():
                predictions[k, :] = np.exp(model(predict_x.float()).flatten())

        return predictions

    def train_zlp(self, n_clusters=5, conf_interval=1, clusters=None, signal='EELS', **kwargs):
        """
        Train the ZLP on the spectral image.

        The spectral image is clustered in ``n_clusters`` clusters, according to e.g. the integrated intensity or thickness.
        A random spectrum is then taken from each cluster, which together defines one replica. The training is initiated
        by calling :py:meth:`train_zlp_scaled() <EELSFitter.core.training.train_zlp_scaled>`.

        Parameters
        ----------
        n_clusters: int, optional
            number of clusters
        conf_interval: int, optional
            Default is 1
        clusters
        signal: str, optional
            Type of spectrum. Set to EELS by default.
        **kwargs
            Additional keyword arguments that are passed to the method :py:meth:`train_zlp_scaled() <EELSFitter.core.training.train_zlp_scaled>` in the :py:mod:`training` module.
        """
        self.cluster(n_clusters)

        training_data = self.get_cluster_spectra(conf_interval=conf_interval, signal=signal)
        train_zlp_scaled(self, training_data, **kwargs)

    def calc_zlp_ntot(self, ntot):
        """
        Returns the shape-(M, N) array of zlp model predictions at the scaled log integrated intensity ``ntot``.
        M and N correspond to the number of model predictions and :math:`\Delta E` s respectively.

        Parameters
        ----------
        ntot: float
            Log integrated intensity (rescaled)
        """
        deltaE = np.linspace(0.1, 0.9, self.l)
        predict_x_np = np.zeros((self.l, 2))
        predict_x_np[:, 0] = deltaE
        predict_x_np[:, 1] = ntot

        predict_x = torch.from_numpy(predict_x_np)
        count = len(self.ZLP_models)
        ZLPs = np.zeros((count, self.l))

        for k in range(count):
            model = self.ZLP_models[k]
            with torch.no_grad():
                predictions = np.exp(model(predict_x.float()).flatten())
            ZLPs[k, :] = predictions

        return ZLPs

    def load_zlp_models(self, path_to_models, plot_chi2=True, idx=None, **kwargs):
        """
        Loads the trained ZLP models and stores them in ``self.ZLP_models``. Models that have a :math:`\chi^2 > \chi^2_{\mathrm{mean}} + 5\sigma` are
        discarded, where :math:`\sigma` denotes the 68% CI.

        Parameters
        ----------
        path_to_models: str
            Location where the model predictions have been stored after training.
        plot_chi2: bool, optional
            When set to `True`, plot and save the :math:`\chi^2` distribution.
        idx: int, optional
            When specified, only the zlp labelled by ``idx`` is loaded, instead of all model predictions.

        """

        if not os.path.exists(path_to_models):
            print(
                "No path " + path_to_models + " found. Please ensure spelling and that there are models trained.")
            return

        self.ZLP_models = []

        path_to_models += (path_to_models[-1] != '/') * '/'
        path_dE1 = "dE1.txt"
        model = MLP(num_inputs=2, num_outputs=1)
        self.dE1 = np.loadtxt(os.path.join(path_to_models, path_dE1))

        path_scale_var = 'scale_var.txt'
        self.scale_var_log_sum_I = np.loadtxt(os.path.join(path_to_models, path_scale_var))
        try:
            path_scale_var_deltaE = 'scale_var_deltaE.txt'
            self.scale_var_deltaE = np.loadtxt(os.path.join(path_to_models, path_scale_var_deltaE))
            print("found delta E vars")
        except:
            pass

        if self.clustered is not None:
            if self.n_clusters != self.dE1.shape[1]:
                print("image clustered in ", self.n_clusters, " clusters, but ZLP-models take ", self.dE1.shape[1],
                      " clusters, reclustering based on models.")
                self.cluster_on_cluster_values(self.dE1[0, :])
        else:
            self.cluster_on_cluster_values(self.dE1[0, :])

        if idx is not None:
            with torch.no_grad():
                model.load_state_dict(torch.load(os.path.join(path_to_models, "nn_rep_{}".format(idx))))
            self.ZLP_models.append(copy.deepcopy(model))
            return

        path_costs = "costs_test_"
        files_costs = [filename for filename in os.listdir(path_to_models) if filename.startswith(path_costs)]
        path_nn_reps = "nn_rep_"
        files_nn_reps = [filename for filename in os.listdir(path_to_models) if filename.startswith(path_nn_reps)]
        
        cost_files_nr = []
        for costs_name in files_costs:
            pattern = "\d+" 
            string = re.findall(pattern, costs_name)
            cost_files_nr.append(int(string[0]))
        cost_files_nr = np.array(cost_files_nr)
        cost_files_nr.sort()
        
        nn_reps_files_nr = []
        for nn_reps_name in files_nn_reps:
            pattern = "\d+" 
            string = re.findall(pattern, nn_reps_name)
            nn_reps_files_nr.append(int(string[0]))
        nn_reps_files_nr = np.array(nn_reps_files_nr)
        nn_reps_files_nr.sort()
        
        cost_tests = []
        cost_trains = []
        for i in cost_files_nr:
            path_tests = os.path.join(path_to_models, 'costs_test_{}.txt'.format(i))
            path_trains = os.path.join(path_to_models, 'costs_train_{}.txt'.format(i))
            
            with open(path_tests) as f:
                for line in f:
                    cost_tests.append(float(line.strip()))

            with open(path_trains) as f:
                for line in f:
                    cost_trains.append(float(line.strip()))

        cost_tests = np.array(cost_tests)
        cost_tests_mean = np.mean(cost_tests)
        cost_tests_std = np.percentile(cost_tests, 68)
        threshold_costs_tests = cost_tests_mean + 5 * cost_tests_std
        cost_tests = cost_tests[cost_tests < threshold_costs_tests]

        cost_trains = np.array(cost_trains)
        cost_trains_mean = np.mean(cost_trains)
        cost_trains_std = np.percentile(cost_trains, 68)
        threshold_costs_trains = cost_trains_mean + 5 * cost_trains_std
        
        nn_rep_idx = np.argwhere(cost_trains < threshold_costs_trains)
        cost_trains = cost_trains[cost_trains < threshold_costs_trains]

        # plot the chi2 distributions
        if plot_chi2:
            fig = plot_cost_dist(cost_trains, cost_tests, cost_tests_std, **kwargs)
            fig.savefig(os.path.join(self.output_path, 'chi2_dist.pdf'))
            print('chi2 plot saved at {}'.format(self.output_path))

        for idx in nn_rep_idx.flatten():
            path = os.path.join(path_to_models, 'nn_rep_{}'.format(nn_reps_files_nr[idx]))
            model.load_state_dict(torch.load(path))
            self.ZLP_models.append(copy.deepcopy(model))

    # METHODS ON DIELECTRIC FUNCTIONS
    def calc_thickness(self, spect, n, N_ZLP=1):
        """
        Calculates thickness from sample data, using Egerton [1]_

        Parameters
        ----------
        spect : numpy.ndarray, shape=(M,)
            spectral image
        n : float
            refraction index
        N_ZLP: float or int
            Set to 1 by default, for already normalized EELS spectra.

        Returns
        -------
        te: float
            thickness

        Notes
        -----
        Surface scatterings are not corrected for. If you wish to correct
        for surface scatterings, please extract the thickness ``t`` from :py:meth:`kramers_kronig_hs() <EELSFitter.core.spectral_image.SpectralImage.kramers_kronig_hs>`.


        .. [1] Ray Egerton, "Electron Energy-Loss Spectroscopy in the Electron
           Microscope", Springer-Verlag, 2011.

        """
        me = self.m_0
        e0 = self.e0
        beta = self.beta

        eaxis = self.deltaE[self.deltaE > 0]  # axis.axis.copy()
        y = spect[self.deltaE > 0]
        i0 = N_ZLP

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
        tgt = e0 * (2 * me + e0) / (me + e0)

        # Calculation of the ELF by normalization of the SSD
        # We start by the "angular corrections"
        Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / self.ddeltaE  # axis.scale

        K = np.sum(Im / eaxis) * self.ddeltaE
        K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
        te = (332.5 * K * ke / i0)

        return te

    def kramers_kronig_hs(self, I_EELS,
                          N_ZLP=None,
                          iterations=1,
                          n=None,
                          t=None,
                          delta=0.5, correct_S_s=False):
        r"""
        Computes the complex dielectric function from the single scattering
        distribution (SSD) ``I_EELS`` following the Kramers-Kronig relations. This
        is inspired by Hyperspy's :py:meth:`kramers_kronig_analysis <http://hyperspy.org/hyperspy-doc/v0.8/api/hyperspy._signals.html#hyperspy._signals.eels.EELSSpectrum.kramers_kronig_analysis>`.


        Parameters
        ----------
        I_EELS: numpy.ndarray, shape=(M,)
            SSD of length energy-loss (M)
        N_ZLP: float
            Total integrated intensity of the ZLP
        iterations: int
            Number of the iterations for the internal loop to remove the
            surface plasmon contribution. If 1 the surface plasmon contribution
            is not estimated and subtracted (the default is 1).
        n: float
            The medium refractive index. Used for normalization of the
            SSD to obtain the energy loss function. If given the thickness
            is estimated and returned. It is only required when `t` is None.
        t: float
            The sample thickness in nm.
        delta : float
            A small number (0.1-0.5 eV) added to the energy axis in
            specific steps of the calculation the surface loss correction to
            improve stability.
        correct_S_s: bool, optional
            Whether or not to correct for surface scatterings. Set to `False` by default

        Returns
        -------
        eps: numpy.ndarray
            The complex dielectric function,

                .. math::
                    \epsilon = \epsilon_1 + i*\epsilon_2,

        te: float
            local thickness
        Srfint: numpy.ndarray
            Surface losses correction

        Notes
        -----
        This method is based on Egerton's Matlab code [1]_, as also stated in the `hyperspy documentation <http://hyperspy.org/hyperspy-doc/v0.8/api/hyperspy._signals.html#hyperspy._signals.eels.EELSSpectrum.kramers_kronig_analysis>`_.

        References`
        ----------

        .. [1] Ray Egerton, "Electron Energy-Loss Spectroscopy in the Electron
           Microscope", Springer-Verlag, 2011.

        """

        output = {}
        # Constants and units
        me = 511.06

        e0 = self.e0
        beta = self.beta

        eaxis = self.deltaE[self.deltaE > 0]  # axis.axis.copy()
        S_E = I_EELS[self.deltaE > 0]
        y = I_EELS[self.deltaE > 0]
        l = len(eaxis)
        i0 = N_ZLP

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2  # m0 v**2
        tgt = e0 * (2 * me + e0) / (me + e0)
        rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)  # me c**2 / (hbar c) gamma sqrt(2Ekin /(me c**2))

        for io in range(iterations):
            # Calculation of the ELF by normalization of the SSD
            # We start by the "angular corrections"
            Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / self.ddeltaE  # axis.scale
            if n is None and t is None:
                raise ValueError("The thickness and the refractive index are "
                                 "not defined. Please provide one of them.")
            elif n is not None and t is not None:
                raise ValueError("Please provide the refractive index OR the "
                                 "thickness information, not both")
            elif n is not None:
                # normalize using the refractive index.
                K = np.sum(Im / eaxis) * self.ddeltaE
                K = K / (np.pi / 2) / (1 - 1. / n ** 2)
                te = (332.5 * K * ke / i0)

            Im = Im / K

            # Kramers Kronig Transform:
            # We calculate KKT(Im(-1/epsilon))=1+Re(1/epsilon) with FFT
            # Follows: D W Johnson 1975 J. Phys. A: Math. Gen. 8 490
            # Use an optimal FFT size to speed up the calculation, and
            # make it double the closest upper value to workaround the
            # wrap-around problem.
            esize = next_fast_len(2 * l)  # 2**math.floor(math.log2(l)+1)*4
            q = -2 * np.fft.fft(Im, esize).imag / esize

            q[:l] *= -1
            q = np.fft.fft(q)
            # Final touch, we have Re(1/eps)
            Re = q[:l].real + 1
            # Egerton does this to correct the wrap-around problem, but in our
            # case this is not necessary because we compute the fft on an
            # extended and padded spectrum to avoid this problem.
            # Re=real(q)
            # Tail correction
            # vm=Re[axis.size-1]
            # Re[:(axis.size-1)]=Re[:(axis.size-1)]+1-(0.5*vm*((axis.size-1) /
            #  (axis.size*2-arange(0,axis.size-1)))**2)
            # Re[axis.size:]=1+(0.5*vm*((axis.size-1) /
            #  (axis.size+arange(0,axis.size)))**2)

            # Epsilon appears:
            #  We calculate the real and imaginary parts of the CDF
            e1 = Re / (Re ** 2 + Im ** 2)
            e2 = Im / (Re ** 2 + Im ** 2)

            if iterations > 0 and N_ZLP is not None:
                # Surface losses correction:
                #  Calculates the surface ELF from a vaccumm border effect
                #  A simulated surface plasmon is subtracted from the ELF
                Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
                adep = (tgt / (eaxis + delta) *
                        np.arctan(beta * tgt / eaxis) -
                        beta / 1000. /
                        (beta ** 2 + eaxis ** 2. / tgt ** 2))
                Srfint = 2000 * K * adep * Srfelf / rk0 / te * self.ddeltaE  # axis.scale
                if correct_S_s == True:
                    print("correcting S_s")
                    Srfint[Srfint < 0] = 0
                    Srfint[Srfint > S_E] = S_E[Srfint > S_E]
                y = S_E - Srfint
                _logger.debug('Iteration number: %d / %d', io + 1, iterations)

        eps = (e1 + e2 * 1j)
        del y
        del I_EELS
        if 'thickness' in output:
            # As above,prevent errors if the signal is a single spectrum
            output['thickness'] = te

        return eps, te, Srfint

    def KK_pixel(self, i, j, signal='EELS', **kwargs):
        """
        Perform a Kramer-Krönig analysis on pixel (``i``, ``j``).

        Parameters
        ----------
        i : int
            x-coordinate of the pixel
        j : int
            y-coordinate of the pixel.

        Returns
        -------
        dielectric_functions : numpy.ndarray, shape=(M,)
            Collection dielectric-functions replicas at pixel (``i``, ``j``).
        ts : float
            Thickness.
        S_ss : array_like
            Surface scatterings.
        IEELSs : array_like
            Deconvoluted EELS spectrum.

        """

        ZLPs = self.calc_zlps_matched(i, j)

        dielectric_functions = (1 + 1j) * np.zeros(ZLPs[:, self.deltaE > 0].shape)
        S_ss = np.zeros(ZLPs[:, self.deltaE > 0].shape)
        ts = np.zeros(ZLPs.shape[0])
        IEELSs = np.zeros(ZLPs.shape)
        max_ieels = np.zeros(ZLPs.shape[0])
        n = self.n[self.clustered[i, j]]
        for k in range(ZLPs.shape[0]):
            ZLP_k = ZLPs[k, :]
            N_ZLP = np.sum(ZLP_k)
            IEELS = self.deconvolute(i, j, ZLP_k)
            IEELSs[k, :] = IEELS
            max_ieels[k] = self.deltaE[np.argmax(IEELS)]
            if signal in self.EELS_NAMES:
                dielectric_functions[k, :], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP=N_ZLP, n=n, **kwargs)
            else:
                ts[k] = self.calc_thickness(IEELS, n, N_ZLP)
        if signal in self.EELS_NAMES:
            return dielectric_functions, ts, S_ss, IEELSs, max_ieels

        IEELSs_OG = IEELSs
        ts_OG = ts
        max_OG = max_ieels

        ZLPs_signal = self.calc_zlps_matched(i, j, signal=signal)
        
        dielectric_functions = (1 + 1j) * np.zeros(ZLPs_signal[:, self.deltaE > 0].shape)
        S_ss = np.zeros(ZLPs_signal[:, self.deltaE > 0].shape)
        ts = np.zeros(ZLPs_signal.shape[0])
        IEELSs = np.zeros(ZLPs_signal.shape)
        max_ieels = np.zeros(ZLPs_signal.shape[0])

        for k in range(ZLPs_signal.shape[0]):
            ZLP_k = ZLPs_signal[k, :]
            N_ZLP = np.sum(ZLP_k)
            IEELS = self.deconvolute(i, j, ZLP_k, signal=signal)
            IEELSs[k] = IEELS
            max_ieels[k] = self.deltaE[np.argmax(IEELS)]
            dielectric_functions[k, :], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP=N_ZLP, n=n, **kwargs)

        return [ts_OG, IEELSs_OG, max_OG], [dielectric_functions, ts, S_ss, IEELSs, max_ieels]

    def im_dielectric_function(self, track_process=False, plot=False, save_index=None, save_path="KK_analysis"):
        """
        Computes the dielectric function by performing a Kramer-Krönig analysis at each pixel.

        Parameters
        ----------
        track_process: bool, optional
            default is `False`, if `True`,  outputs for each pixel the program that is busy with that pixel.
        plot: bool, optional
            default is `False`, if `True`, plots all calculated dielectric functions
        save_index: int, optional
            optional labelling to include in ``save_path``.
        save_path: str, optional
            location where the dielectric function, SSD and thickness are stored.
        """

        self.dielectric_function_im_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.dielectric_function_im_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.thickness_avg = np.zeros(self.image_shape)
        self.thickness_std = np.zeros(self.image_shape)
        self.IEELS_avg = np.zeros(self.data.shape)
        self.IEELS_std = np.zeros(self.data.shape)
        if plot:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if track_process: print("calculating dielectric function for pixel ", i, j)

                dielectric_functions, ts, S_ss, IEELSs = self.KK_pixel(i, j)
                self.dielectric_function_im_avg[i, j, :] = np.average(dielectric_functions, axis=0)
                self.dielectric_function_im_std[i, j, :] = np.std(dielectric_functions, axis=0)
                self.S_s_avg[i, j, :] = np.average(S_ss, axis=0)
                self.S_s_std[i, j, :] = np.std(S_ss, axis=0)
                self.thickness_avg[i, j] = np.average(ts)
                self.thickness_std[i, j] = np.std(ts)
                self.IEELS_avg[i, j, :] = np.average(IEELSs, axis=0)
                self.IEELS_std[i, j, :] = np.std(IEELSs, axis=0)
        if save_index is not None:
            save_path += (not save_path[0] == '/') * '/'
            with open(save_path + "diel_fun_" + str(save_index) + ".npy", 'wb') as f:
                np.save(f, self.dielectric_function_im_avg)
            with open(save_path + "S_s_" + str(save_index) + ".npy", 'wb') as f:
                np.save(f, self.S_s_avg)
            with open(save_path + "thickness_" + str(save_index) + ".npy", 'wb') as f:
                np.save(f, self.thickness_avg)

    def cluster(self, n_clusters=5, based_on="log", **kwargs):
        """
        Clusters the spectral image into clusters according to the (log) integrated intensity at each
        pixel. Cluster means are stored in the attribute ``self.clusters`` and the index to which each cluster belongs is
        stored in the attribute ``self.clustered``.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters, 5 by default
        based_on : str, optional
            One can cluster either on the sum of the intensities (pass ````sum````), the log of the sum (pass ````log````) or the thickness (pass ````thickness````).
            The default is ````log````.
        **kwargs : keyword arguments
            additional keyword arguments to pass to :py:meth:`k_means_clustering.k_means() <EELSFitter.clustering.k_means_clustering.k_means()>`.
        """

        if based_on == "sum":
            values = np.sum(self.data, axis=2).flatten()
        elif based_on == "log":
            values = np.log(np.sum(np.maximum(self.data, 1e-14), axis=2).flatten())
        elif based_on == "thickness":
            values = self.t[:, :, 0].flatten()
        elif type(based_on) == np.ndarray:
            values = based_on.flatten()
            if values.size != (self.image_shape[0] * self.image_shape[1]):
                raise IndexError("The size of values on which to cluster does not match the image size.")
        else:
            values = np.sum(self.data, axis=2).flatten()
        clusters_unsorted, r = k_means(values, n_clusters=n_clusters, **kwargs)
        self.clusters = np.sort(clusters_unsorted)[::-1]
        arg_sort_clusters = np.argsort(clusters_unsorted)[::-1]
        self.clustered = np.zeros(self.image_shape)
        for i in range(n_clusters):
            in_cluster_i = r[arg_sort_clusters[i]]
            self.clustered += ((np.reshape(in_cluster_i, self.image_shape)) * i)
        self.clustered = self.clustered.astype(int)

    def cluster_on_cluster_values(self, cluster_values):
        """
        If the image has been clustered before and the the cluster means are already known,
        one can use this function to reconstruct the original clustering of the image.

        Parameters
        ----------
        cluster_values: numpy.ndarray, shape=(M,)
            Array with the cluster means

        Notes
        -----
        Works only for images clustered on (log) integrated intensity."""
        self.clusters = cluster_values

        values = np.sum(self.data, axis=2)
        check_log = (np.nanpercentile(values, 5) > cluster_values.max())
        if check_log:
            values = np.log(values)
        valar = (values.transpose() * np.ones(np.append(self.image_shape, self.n_clusters)).transpose()).transpose()
        self.clustered = np.argmin(np.absolute(valar - cluster_values), axis=2)
        if len(np.unique(self.clustered)) < self.n_clusters:
            warnings.warn(
                "it seems like the clustered values of dE1 are not clustered on this image/on log or sum. Please check clustering.")


    def get_ticks(self, sig_ticks=2, npix_xtick=10, npix_ytick=10, scale_ticks=1, tick_int=False):
        """
        Sets the proper tick labels and tick positions for the heatmap plots.

        Parameters
        ----------
        sig_ticks : int, optional
            Set the amount of significant numbers displayed in the ticks. The default is 2.
        npix_xtick : float, optional
            Display a tick per n pixels in the x-axis. Note that this value can be a float. The default is 10.
        npix_ytick : float, optional
            Display a tick per n pixels in the y-axis. Note that this value can be a float. The default is 10.
        scale_ticks : float, optional
            Change the scaling of the numbers displayed in the ticks. Microns ([\u03BCm]) are assumed as standard scale, adjust scaling from there. The default is 1.
        tick_int : bool, optional
            Set whether you only want the ticks to display as integers instead of floats. The default is False.

        Returns
        -------
        xticks : numpy.ndarray, shape=(M,)
            Array of the xticks positions.
        yticks : numpy.ndarray, shape=(M,)
            Array of the yticks positions.
        xticks_labels : numpy.ndarray, shape=(M,)
            Array with strings of the xtick labels.
        yticks_labels : numpy.ndarray, shape=(M,)
            Array with strings of the ytick labels.
        """

        xticks = np.arange(0, self.x_axis.shape[0], npix_xtick)
        yticks = np.arange(0, self.y_axis.shape[0], npix_ytick)
        if tick_int == True:
            xticks_labels = (xticks * round_scientific(self.pixelsize[1] * scale_ticks, sig_ticks)).astype(int)
            yticks_labels = (yticks * round_scientific(self.pixelsize[0] * scale_ticks, sig_ticks)).astype(int)
        else:
            xticks_labels = trunc(xticks * round_scientific(self.pixelsize[1] * scale_ticks, sig_ticks), sig_ticks)
            yticks_labels = trunc(yticks * round_scientific(self.pixelsize[0] * scale_ticks, sig_ticks), sig_ticks)
        return xticks, yticks, xticks_labels, yticks_labels

    # GENERAL FUNCTIONS
    def get_key(self, key):
        if key.lower() in (string.lower() for string in self.EELS_NAMES):
            return 'data'
        elif key.lower() in (string.lower() for string in self.IEELS_NAMES):
            return 'ieels'
        elif key.lower() in (string.lower() for string in self.ZLP_NAMES):
            return 'zlp'
        elif key.lower() in (string.lower() for string in self.DIELECTRIC_FUNCTION_NAMES):
            return 'eps'
        elif key.lower() in (string.lower() for string in self.THICKNESS_NAMES):
            return 'thickness'
        else:
            return key

    # STATIC METHODS
    @staticmethod
    def get_prefix(unit, SIunit=None, numeric=True):
        """
        Method to convert units to their associated SI values.

        Parameters
        ----------
        unit: str,
            unit of which the prefix is requested
        SIunit: str, optional
            The SI unit of the unit
        numeric: bool, optional
            Default is `True`. If `True` the prefix is translated to the numeric value
            (e.g. :math:`10^3` for `k`)

        Returns
        ------
        prefix: str or int
            The character of the prefix or the numeric value of the prefix
        """
        if SIunit is not None:
            lenSI = len(SIunit)
            if unit[-lenSI:] == SIunit:
                prefix = unit[:-lenSI]
                if len(prefix) == 0:
                    if numeric:
                        return 1
                    else:
                        return prefix
            else:
                print("provided unit not same as target unit: " + unit + ", and " + SIunit)
                if numeric:
                    return 1
                else:
                    prefix = None
                    return prefix
        else:
            prefix = unit[0]
        if not numeric:
            return prefix

        if prefix == 'p':
            return 1E-12
        if prefix == 'n':
            return 1E-9
        if prefix in ['u', 'micron', 'µ']:
            return 1E-6
        if prefix == 'm':
            return 1E-3
        if prefix == 'k':
            return 1E3
        if prefix == 'M':
            return 1E6
        if prefix == 'G':
            return 1E9
        if prefix == 'T':
            return 1E12
        else:
            print("either no or unknown prefix in unit: " + unit + ", found prefix " + prefix + ", asuming no.")
        return 1

    def __getitem__(self, key):
        """ Determines behavior of `self[key]` """
        return self.data[key]

    def __getattr__(self, key):
        key = self.get_key(key)
        return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        key = self.get_key(key)
        self.__dict__[key] = value

    def __str__(self):
        if self.name is not None:
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return 'Spectral image: ' + name_str + ", image size:" + str(self.data.shape[0]) + 'x' + \
               str(self.data.shape[1]) + ', deltaE range: [' + str(round(self.deltaE[0], 3)) + ',' + \
               str(round(self.deltaE[-1], 3)) + '], deltadeltaE: ' + str(round(self.ddeltaE, 3))

    def __repr__(self):
        data_str = "data * np.ones(" + str(self.shape) + ")"
        if self.name is not None:
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return "Spectral_image(" + data_str + ", deltadeltaE=" + str(round(self.ddeltaE, 3)) + name_str + ")"

    def __len__(self):
        return self.l


# GENERAL DATA MODIFICATION FUNCTIONS  

def CFT(x, y):
    """

    Parameters
    ----------
    x: numpy.ndarray, shape=(M,)
    y: numpy.ndarray, shape=(M,)

    Returns
    -------
    F_k: numpy.ndarray, shape=(M,)

    """
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_max = np.max(x)
    delta_x = (x_max - x_0) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(2j * np.pi * N_0 * k / N) * delta_x  # np.exp(-1j*(x_0)*k*delta_omg)*delta_x
    F_k = cont_factor * np.fft.fft(y)
    return F_k


def iCFT(x, Y_k):
    """

    Parameters
    ----------
    x: numpy.ndarray, shape=(M,)
    Y_k: numpy.ndarray, shape=(M,)

    Returns
    -------
    f_n: numpy.ndarray, shape=(M,)

    """
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max - x_0) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(-2j * np.pi * N_0 * k / N)
    f_n = np.fft.ifft(cont_factor * Y_k) / delta_x
    return f_n

# MODELING CLASSES AND FUNCTIONS
def scale(inp, ab):
    """
    Rescale the input features by applying a linear map such that the range covers 0.1 to 0.9.

    Parameters
    ---------
    inp: numpy.ndarray, shape=(M,)
        Input feature array of length M
    ab: list
        Contains rescaling parameters

    Returns
    -------
    output: numpy.ndarray, shape=(M,)
        Rescaled features
    """
    return inp * ab[0] + ab[1]


def find_scale_var(inp, min_out=0.1, max_out=0.9):
    """
    Find rescaling parameters such that the input features lie between 0.1 and 0.9

    Parameters
    ----------
    inp: numpy.ndarray, shape=(M,)
        Input feature array of length M
    min_out: float, optional
        Minimum of input feature, set to 0.1 by default
    max_out: float, optional
        Maximum of input feature, set to 0.9 by default

    Returns
    -------
    output: list
        Rescaling parameters
    """
    a = (max_out - min_out) / (inp.max() - inp.min())
    b = min_out - a * inp.min()
    return [a, b]


def round_scientific(value, n_sig):
    """
    Round ``value`` off to ``n_sig`` digits.

    Parameters
    ----------
    value: float
    n_sig: int
        Number of signicant digits

    Returns
    -------
    output: float
        Rounded version of ``value``.
    """
    if value == 0:
        return 0
    if np.isnan(value):
        return 0
    scale = int(math.floor(math.log10(abs(value))))
    num = round(value, n_sig - scale - 1)
    return num


def trunc(values, decs=0):
    """
    Returns the truncated version of the input

    Parameters
    ----------
    values: numpy.ndarray, shape=(M,)
        Input array
    decs: int
        Number of decimals to keep
    Returns
    -------
    output: truncated version of the input
    """
    return np.trunc(values*10**decs)/(10**decs)
