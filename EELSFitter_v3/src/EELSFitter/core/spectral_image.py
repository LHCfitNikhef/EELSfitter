import math
import numpy as np
import logging
import os
import copy
import warnings
import torch
import bz2
import pickle
import _pickle as cPickle
import matplotlib.pyplot as plt
import ncempy.io as nio

from scipy.fft import next_fast_len, fft, ifft
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.mixture import GaussianMixture

from .training import MultilayerPerceptron, TrainZeroLossPeak
from ..plotting.training_perf import plot_cost_dist
from ..plotting.zlp import plot_zlp_cluster_predictions
from ..plotting.mva_perf import plot_pca_variance

# Where is this used?
plt.rcParams.update({'font.size': 16})  # pyplot style

# Where is this used?
_logger = logging.getLogger(__name__)


class ProcessSpectralImage:
    def __init__(self, image):
        r"""Load spectral image.

        ARGUMENTS
        ---------
            image (dm4): spectral image.

        EXAMPLE
        -------
        processed_SI = ProcessSpectralImage(image=img)
        processed_SI.save_spectra_in_3D_array()
        processed_SI.self.center_ZLP_at_0eV()
        """
        self.image = image
        self.x_signal = image.eaxis
        self.n_rows = image.shape[0]
        self.n_cols = image.shape[1]

    def save_3D_arrays_in_4D_array(self):
        A = np.zeros(shape=(self.n_rows, self.n_cols, self.n_aisles, 3))
        A[:, :, :, 0] = self.arr3D_deltaE
        A[:, :, :, 1] = self.arr3D_deltaE_shifted
        A[:, :, :, 2] = self.arr3D_signal
        return A

    def save_spectra_in_3D_array(self):
        self.n_aisles = self.x_signal.shape[0]
        # 3D array: (row, column, aisle)
        self.arr3D_zeroes = np.zeros(shape=(self.n_rows, self.n_cols, self.n_aisles))
        self.arr3D_deltaE = np.copy(self.arr3D_zeroes)
        self.arr3D_signal = np.copy(self.arr3D_zeroes)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # save the deltaE in aisle of 3D array
                self.arr3D_deltaE[i, j, :] = self.x_signal
                # save the signal in aisle of 3D array
                self.arr3D_signal[i, j, :] = self.image.get_pixel_signal(i=i, j=j)

    def center_ZLP_at_0eV(self):
        r"""Take maximum of ZLP, determine its median and shift ZLP to 0 eV."""
        # get index of max value for each aisle
        idx = self.arr3D_signal.argmax(axis=2)
        a1, a2 = np.indices(idx.shape)

        # get the shifted energy per aisle, i.e. per (row,col)
        self.arr2D_shifted_E = self.arr3D_deltaE[a1, a2, idx]

        # create 3D array of shifted energy
        arr3D_shifted_E = np.copy(self.arr3D_zeroes)
        for k in range(self.n_aisles):
            # fill each layer with the same 2D array containing shifted energy
            arr3D_shifted_E[:, :, k] = self.arr2D_shifted_E

        # subtract each aisle by the corresponding shifted energy
        self.arr3D_deltaE_shifted = self.arr3D_deltaE - arr3D_shifted_E

    def check_if_ZLP_centered_at_0eV_for(self, row=0):
        x0 = self.arr3D_deltaE
        xS = self.arr3D_deltaE_shifted
        y0 = self.arr3D_signal
        # plot
        fig, ax = plt.subplots()
        ax.set_xlim(-4, 4)
        row = 20
        for i in range(self.n_cols):
            # plot each spectrum
            ax.plot(x0[row, i, :], y0[row, i, :], 'r')
            # plot each spectrum (shifted)
            ax.plot(xS[row, i, :], y0[row, i, :], 'b')
        return fig

    def plot_spectral_image(self, arr, cmap='turbo_r', cbar_title=''):
        fig, ax = plt.subplots()
        p = ax.imshow(arr, cmap=cmap)
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_title(cbar_title)  # intensity lower for thicker sample
        return fig, ax


class SpectralImage:
    #  signal names
    DIELECTRIC_FUNCTION_NAMES = ['dielectric_function', 'dielectricfunction', 'dielec_func',
                                 'die_fun', 'df', 'epsilon']
    EELS_NAMES = ['electron_energy_loss_spectrum', 'electron_energy_loss',
                  'EELS', 'EEL', 'energy_loss', 'data']
    IEELS_NAMES = ['inelastic_scattering_energy_loss_spectrum', 'inelastic_scattering_energy_loss',
                   'inelastic_scattering', 'IEELS', 'IES', 'signal_scattering_distribution', 'ssd']
    ZLP_NAMES = ['zeros_loss_peak', 'zero_loss', 'ZLP', 'ZLPs', 'zlp', 'zlps']
    THICKNESS_NAMES = ['t', 'thickness', 'thick', 'thin']
    POOLED_ADDITION = ['pooled', 'pool', '_pooled', '_pool']
    PCA_ADDITION = ['pca', 'PCA', '_pca', '_PCA']
    NMF_ADDITION = ['nmf', 'NMF', '_nmf', '_NMF']

    # meta data names
    COLLECTION_ANGLE_NAMES = ["collection_angle", "col_angle", "beta"]
    CONVERGENCE_ANGLE_NAMES = ["convergence_angle", "con_angle", "alpha"]
    BEAM_ENERGY_NAMES = ["beam_energy", "beam_E", "E_beam", "E0", "E_0", "e0", "e_0"]

    # Global physical constants
    m_e = 5.1106E5  # [eV],     Electron rest mass
    e_c = 1.602176487E-19  # [C],      Electron charge
    a0 = 5.29E-11  # [m],      Bohr radius
    h_bar = 6.582119569E-16  # [eV*s],   Planck's constant
    c = 2.99792458E8  # [m/s],    Speed of light

    def __init__(self, data, deltaE=None, pixel_size=None, beam_energy=None,
                 collection_angle=None, convergence_angle=None, name=None, **kwargs):
        r"""
        The spectral image class that provides several tools to analyse spectral images with the zero-loss peak
        subtracted.

        Parameters
        ----------
        data: numpy.ndarray, shape=(M,N,L)
            Array containing the 3-D spectral image. The axes correspond to the x-axis (M), y-axis (N) and energy-loss (L).
        deltaE: float, optional
            bin width in energy loss spectrum in eV
        pixel_size: numpy.ndarray, shape=(2,), optional
            width of pixels in nm
        beam_energy: float, optional
            Energy of electron beam in KeV
        collection_angle: float, optional
            Collection angle of STEM in mrad
        convergence_angle: float, optional
            Convergence angle of STEM in mrad
        name: str, optional
            Name of the SI data file

        Examples
        --------
        An example how to train and analyse a spectral image::

            path_to_image = 'path to dm4 file'
            im = SpectralImage.load_data(path_to_image)
            im.train_zlp_models(n_clusters=8,
                                n_rep=100,
                                n_epochs=150000,
                                bs_rep_num=1,
                                path_to_models=path_to_models)
        """
        # Image properties
        self.data = data
        self.deltaE = deltaE  # eV, energy bin width
        self.set_eaxis()  # Sets the shifted energy axis (E = 0 corresponds to peak of ZLP)
        self.pixel_size = pixel_size  # nm, real world size of the pixel
        self.calc_axes()  # Calculates the x and y-axis based on the pixel size.
        self.beam_energy = beam_energy  # KeV, incoming beam energy
        self.collection_angle = collection_angle  # mrad, collection semi-angle
        self.convergence_angle = convergence_angle  # mrad, convergence semi-angle
        self.name = name  # SI name

        # ZLP models properties
        self.cluster_centroids = None  # Center values of the clusters
        self.cluster_labels = None  # 2d array where each pixel location has a corresponding cluster label
        self.cluster_signals = None  # All signals collected per cluster
        self.zlp_models = None
        self.scale_var_eaxis = find_scale_var(self.eaxis)  # scale variables for the energy axis
        self.scale_var_log_int_i = None  # scale variables for the log of the integrated intensities
        self.dE1 = None  # dE1 hyperparameter for each cluster
        self.dE2 = None  # dE2 hyperparameter for each cluster
        self.FWHM = None  # Full Width at Half Maximum for each cluster

        # Additional image properties for calculations and data modification
        self.data_pool = None  # SI data is enhanced by pooling a pixel with surrounding pixels
        self.data_pca = None  # SI data is enhanced by performing PCA on the data
        self.data_nmf = None  # SI data is enhanced by performing NMF on the data
        
        self.data_zlpsub = None
        self.data_zlpsub_pool = None
        self.data_zlpsub_pca = None
        self.data_zlpsub_nmf = None

        self.data_deconv = None
        self.data_deconv_pool = None
        self.data_deconv_pca = None
        self.data_deconv_nmf = None
        
        self.n = None  # Refractive index, set per cluster
        self.rho = None  # Mass density, set per cluster


        # Other
        self.output_path = os.getcwd()

    # PROPERTIES
    @property
    def shape(self):
        r"""
        Returns 3D-shape of
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object
        """

        return self.data.shape

    @property
    def n_spectra(self):
        r"""
        Returns the number of spectra present in
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object.

        """

        return self.shape[0]*self.shape[1]

    @property
    def n_clusters(self):
        r"""
        Returns the number of clusters in the
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object.

        """

        return len(self.cluster_centroids)

    # METHODS ON SAVING AND LOADING DATA
    @classmethod
    def load_dmfile(cls, path_to_dmfile, load_survey_data=False):
        r"""
        Load the .dm4 (or .dm3) data and return a
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance.

        Parameters
        ----------
        path_to_dmfile: str
            location of .dm4 file
        load_survey_data: bool, optional
            If there is HAADF survey data in your file, you can choose to also load that in. Default is `False`.

        Returns
        -------
        SpectralImage
            :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>`
            instance of the dm4 file

        """

        dmfile = nio.dm.fileDM(path_to_dmfile)

        dmobjects = dmfile.numObjects
        eels_dict = dmfile.getDataset(dmobjects - 2)
        eels_metadict = dmfile.getMetadata(dmobjects - 2)
        if eels_dict['data'].ndim == 3:
            eels_data = np.swapaxes(np.swapaxes(eels_dict['data'], 0, 1), 1, 2)
        elif eels_dict['data'].ndim == 2:
            eels_data = np.swapaxes(np.swapaxes(eels_dict['data'], 0, 1), 1, 2)
        elif eels_dict['data'].ndim == 1:
            # A single signal is constructed to be a 1 by 1 pixel spectral image.
            eels_data = np.zeros(shape=(1, 1, len(eels_dict['data'])))
            eels_data[0, 0, :] = eels_dict['data']
        else:
            print("Dimensions of the data set could not be determined, please check your dataset")

        # eels data
        if eels_dict['pixelSize']:
            try:
                deltaE = eels_dict['pixelSize'][0]
            except (IndexError, ValueError):
                print("deltaE is not found, assuming to be in 1. Otherwise adjust manually")
                deltaE = 1
            try:
                pixel_size = np.array(eels_dict['pixelSize'][1:])
                if len(pixel_size) == 0:
                    pixel_size = np.array([1., 1.])
            except (IndexError, ValueError):
                print("pixel_size is not found, assuming to be 1. Otherwise adjust manually")
                pixel_size = np.array([1, 1])
        else:
            print("The dm4 metadata does not contain 'pixelSize'. deltaE and pixel_size needs to be manually set.")

        if eels_dict['pixelUnit']:
            try:
                energy_unit = eels_dict['pixelUnit'][0]
            except IndexError:
                print("energy_unit is not found, assuming to be in eV. Otherwise adjust manually")
                deltaE = 'eV'
            try:
                pixel_unit = eels_dict['pixelUnit'][1]
            except IndexError:
                print("pixel_unit is not found, assuming to be in nanometer. Otherwise adjust manually")
                pixel_unit = 'nm'
        else:
            print("The dm4 metadata does not contain 'pixelUnit'. energy_unit and pixel_unit needs to be manually set")

        if (energy_unit is not None) and (deltaE is not None):
            deltaE *= cls.get_prefix(energy_unit, 'eV')

        if (pixel_size is not None) and (pixel_unit is not None):
            pixel_size *= cls.get_prefix(pixel_unit, 'm') * 1E9

        # eels metadata
        try:
            beam_energy = eels_metadict['Microscope Info Voltage'] / 1E3  # convert the beam energy to KeV
        except KeyError:
            beam_energy = None
            print("The dm4 metadata does not contain 'Microscope Info Voltage'. beam_energy needs to be manually set.")

        try:
            collection_angle = eels_metadict['EELS Experimental Conditions Collection semi-angle (mrad)']
        except KeyError:
            collection_angle = None
            print("The dm4 metadata does not contain 'EELS Experimental Conditions Collection semi-angle (mrad)'. "
                  "collection_angle needs to be manually set.")

        try:
            convergence_angle = eels_metadict['EELS Experimental Conditions Convergence semi-angle (mrad)']
        except KeyError:
            convergence_angle = None
            print("The dm4 metadata does not contain 'EELS Experimental Conditions Convergence semi-angle (mrad)'. "
                  "convergence_angle needs to be manually set.")

        image = cls(data=eels_data, deltaE=deltaE, pixel_size=pixel_size, beam_energy=beam_energy,
                    collection_angle=collection_angle, convergence_angle=convergence_angle,
                    name=path_to_dmfile[:-4])

        # load in survey data if available
        if dmobjects > 2 and load_survey_data is True:
            image.survey_data = dmfile.getDataset(0)
            image.survey_metadata = dmfile.getMetadata(0)
        return image

    @classmethod
    def load_spectral_image(cls, path_to_pickle):
        r"""
        Loads :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance
        from a pickled file.

        Parameters
        ----------
        path_to_pickle : str
            path to the pickled image file.

        Raises
        ------
        ValueError
            If path_to_pickle does not end on the desired format .pkl.
        FileNotFoundError
            If path_to_pickle does not exist.

        Returns
        -------
        SpectralImage
            :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object
            (i.e. including all attributes) loaded from pickle file.

        """

        if path_to_pickle[-4:] != '.pkl':
            raise ValueError(
                "please provide a path to a pickle file containing a Spectral_image class object.")
        if not os.path.exists(path_to_pickle):
            raise FileNotFoundError(
                "pickled file: " + path_to_pickle + " not found")
        with open(path_to_pickle, 'rb') as pickle_im:
            image = pickle.load(pickle_im)
        return image

    @classmethod
    def load_compressed_Spectral_image(cls, path_to_compressed_pickle):
        r"""
        Loads spectral image from a compressed pickled file. This will take longer than loading from non-compressed
        pickle.

        Parameters
        ----------
        path_to_compressed_pickle : str
            path to the compressed pickle image file.

        Raises
        ------
        ValueError
            If path_to_compressed_pickle does not end on the desired format .pbz2.
        FileNotFoundError
            If path_to_compressed_pickle does not exist.

        Returns
        -------
        image : SpectralImage
             :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` instance
             loaded from the compressed pickle file.

        """

        if path_to_compressed_pickle[-5:] != '.pbz2':
            raise ValueError(
                "please provide a path to a compressed .pbz2 pickle file containing a Spectrall_image class object.")
        if not os.path.exists(path_to_compressed_pickle):
            raise FileNotFoundError(
                'pickled file: ' + path_to_compressed_pickle + ' not found')

        image = cls.decompress_pickle(path_to_compressed_pickle)
        return image

    def save_image(self, filename):
        r"""
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
        r"""
        Function to save image, including all attributes, in compressed pickle (.pbz2) format. Image will \
            be saved at location ``filename``.
            Advantage over :py:meth:`save_image() <EELSFitter.core.spectral_image.SpectralImage.save_image>` is that \
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
    def compressed_pickle(title, data):
        r"""
        Saves ``data`` at location ``title`` as compressed pickle.

        """

        with bz2.BZ2File(title, 'w') as f:
            cPickle.dump(data, f)

    @staticmethod
    def decompress_pickle(file):
        r"""
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

    @staticmethod
    def get_prefix(unit, SIunit=None, numeric=True):
        r"""
        Method to convert units to their associated SI values.

        Parameters
        ----------
        unit: str,
            unit of which the prefix is requested
        SIunit: str, optional
            The SI unit of the unit
        numeric: bool, optional
            Default is `True`. If `True` the prefix is translated to the
            numeric value (e.g. :math:`10^3` for `k`)

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
                print("provided unit not same as target unit: " +
                      unit + ", and " + SIunit)
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
        if prefix == 'K':
            return 1E3
        if prefix == 'M':
            return 1E6
        if prefix == 'G':
            return 1E9
        if prefix == 'T':
            return 1E12
        else:
            print("either no or unknown prefix in unit: " + unit +
                  ", found prefix " + prefix + ", asuming no.")
        return 1

    # METHODS ON IMAGE
    def set_eaxis(self):
        r"""
        Determines the energy losses of the spectral image, based on the bin
        width of the energy loss. It shifts the ``self.eaxis`` attribute such
        that the zero point corresponds with the point of the highest intensity.

        It also set the extrapolated eaxis for calculations that require extrapolation.

        Returns
        -------
        eaxis: numpy.ndarray, shape=(M,)
            Array of :math:`\Delta E` values

        """

        data_avg = np.average(self.data, axis=(0, 1))
        ind_max = np.argmax(data_avg)
        self.eaxis = np.linspace(-ind_max * self.deltaE, (self.shape[2] - ind_max - 1) * self.deltaE, self.shape[2])
        if self.eaxis[-1] <= 200:
            esize = math.ceil((-1 * self.eaxis[0] + 200) / self.deltaE)
            esize = next_fast_len(esize)
        else:
            esize = next_fast_len(len(self.eaxis))
        self.eaxis_extrp = np.linspace(self.eaxis[0], esize * self.deltaE + self.eaxis[0], esize)

    def calc_axes(self):
        r"""
        Determines the x_axis and y_axis of the spectral image. Stores them in
        ``self.x_axis`` and ``self.y_axis`` respectively.
        """

        self.y_axis = np.linspace(0, self.shape[0], self.shape[0] + 1)
        self.x_axis = np.linspace(0, self.shape[1], self.shape[1] + 1)
        if self.pixel_size is not None:
            self.y_axis *= self.pixel_size[0]
            self.x_axis *= self.pixel_size[1]

    def get_pixel_signal(self, i, j, signal_type='EELS', data_type='EELS', **kwargs):
        r"""
        Retrieves the spectrum at pixel (``i``, ``j``).

        Parameters
        ----------
        i: int
            y-coordinate of the pixel
        j: int
            x-coordinate of the pixel
        signal_type: str, optional
            The type of signal that is requested, should comply with the defined
            names. Set to `EELS` by default.
        subtract_zlps: bool, optional
            If `True`, subtract the zlps from the signal, only possible if zlp models are available. Note that the
            median will be taken from the zlps. Set to `False` by default.

        Returns
        -------
        signal : numpy.ndarray, shape=(M,)
            Array with the requested signal from the requested pixel
        """

        if data_type == 'EELS':
            if signal_type in self.POOLED_ADDITION:
                if self.data_pool is not None:
                    signal = np.copy(self.data_pool[i, j, :])
                else:
                    signal = self.pool_pixel(data=self.data, i=i, j=j, **kwargs)
            elif signal_type in self.PCA_ADDITION:
                if self.data_pca is not None:
                    signal = np.copy(self.data_pca[i, j, :])
                else:
                    signal = self.pca_pixel(data=self.data, i=i, j=j, **kwargs)
            elif signal_type in self.NMF_ADDITION:
                if self.data_nmf is not None:
                    signal = np.copy(self.data_nmf[i, j, :])
                else:
                    signal = self.nmf_pixel(data=self.data, i=i, j=j, **kwargs)
            else:
                signal = np.copy(self.data[i, j, :])

        elif data_type == 'deconv' and self.data_deconv is not None:
            if signal_type in self.POOLED_ADDITION:
                if self.data_deconv_pool is not None and isinstance(self.data_deconv_pool[i, j], np.ndarray):
                    signal = np.copy(self.data_deconv_pool[i, j, :])
                else:
                    signal = self.pool_pixel(data=self.data_deconv, i=i, j=j, **kwargs)
            elif signal_type in self.PCA_ADDITION:
                if self.data_deconv_pca is not None and isinstance(self.data_deconv_pca[i, j], np.ndarray):
                    signal = np.copy(self.data_deconv_pca[i, j, :])
                else:
                    signal = self.pca_pixel(data=self.data_deconv, i=i, j=j, **kwargs)
            elif signal_type in self.NMF_ADDITION:
                if self.data_deconv_nmf is not None and isinstance(self.data_deconv_nmf[i, j], np.ndarray):
                    signal = np.copy(self.data_deconv_nmf[i, j, :])
                else:
                    signal = self.nmf_pixel(data=self.data_deconv, i=i, j=j, **kwargs)
            else:
                signal = np.copy(self.data_deconv[i, j, :])

        elif data_type == 'subtract' and self.data_zlpsub is not None:
            if signal_type in self.POOLED_ADDITION:
                if self.data_zlpsub_pool is not None and isinstance(self.data_zlpsub_pool[i, j], np.ndarray):
                    signal = np.copy(self.data_zlpsub_pool[i, j, :])
                else:
                    signal = self.pool_pixel(data=self.data_zlpsub, i=i, j=j, **kwargs)
            elif signal_type in self.PCA_ADDITION:
                if self.data_zlpsub_pca is not None and isinstance(self.data_zlpsub_pca[i, j], np.ndarray):
                    signal = np.copy(self.data_zlpsub_pca[i, j, :])
                else:
                    signal = self.pca_pixel(data=self.data_zlpsub, i=i, j=j, **kwargs)
            elif signal_type in self.NMF_ADDITION:
                if self.data_zlpsub_nmf is not None and isinstance(self.data_zlpsub_nmf[i, j], np.ndarray):
                    signal = np.copy(self.data_zlpsub_nmf[i, j, :])
                else:
                    signal = self.nmf_pixel(data=self.data_zlpsub, i=i, j=j, **kwargs)
            else:
                signal = np.copy(self.data_zlpsub[i, j, :])

        else:
            print("Either no such signal or the signal shape has not been calculated yet, "
                  "returned general EELS signal.")
            signal = np.copy(self.data[i, j, :])
        return signal

    def get_image_signals(self, signal_type='EELS', data_type='EELS', **kwargs):
        r"""
        Get all the signals of the image.

        Parameters
        ----------
        signal_type: str, optional
            Description of signal, ``'EELS'`` by default.
        kwargs

        Returns
        -------
        image_signals : numpy.ndarray, shape=(M,N)
        """

        if data_type == 'EELS':
            if signal_type in self.POOLED_ADDITION:
                if self.data_pool is not None:
                    image_signals = np.copy(self.data_pool)
                else:
                    image_signals = self.pool_image(data=self.data, **kwargs)
            elif signal_type in self.PCA_ADDITION:
                if self.data_pca is not None:
                    image_signals = np.copy(self.data_pca)
                else:
                    image_signals = self.pca_image(data=self.data, **kwargs)
            elif signal_type in self.NMF_ADDITION:
                if self.data_nmf is not None:
                    image_signals = np.copy(self.data_nmf)
                else:
                    image_signals = self.nmf_image(data=self.data, **kwargs)
            else:
                image_signals = np.copy(self.data)

        elif data_type == 'deconv':
            if self.data_deconv is None:
                print("Deconvoluted image not available, calculating now, this might take a while")
                self.deconv_image()
                print("Deconvolution done!")
            if signal_type in self.POOLED_ADDITION:
                if self.data_deconv_pool is not None:
                    image_signals = np.copy(self.data_deconv_pool)
                else:
                    image_signals = self.pool_image(data=self.data_deconv, **kwargs)
            elif signal_type in self.PCA_ADDITION:
                if self.data_deconv_pca is not None:
                    image_signals = np.copy(self.data_deconv_pca)
                else:
                    image_signals = self.pca_image(data=self.data_deconv, **kwargs)
            elif signal_type in self.NMF_ADDITION:
                if self.data_deconv_nmf is not None:
                    image_signals = np.copy(self.data_deconv_nmf)
                else:
                    image_signals = self.nmf_image(data=self.data_deconv, **kwargs)
            else:
                image_signals = np.copy(self.data_deconv)

        elif data_type == 'subtract':
            if self.data_deconv is None:
                print("Subtracted image not available, calculating now, this might take a while")
                self.subtract_image()
                print("Subtraction done!")
            if signal_type in self.POOLED_ADDITION:
                if self.data_zlpsub_pool is not None:
                    image_signals = np.copy(self.data_zlpsub_pool)
                else:
                    image_signals = self.pool_image(data=self.data_zlpsub, **kwargs)
            elif signal_type in self.PCA_ADDITION:
                if self.data_zlpsub_pca is not None:
                    image_signals = np.copy(self.data_zlpsub_pca)
                else:
                    image_signals = self.pca_image(data=self.data_zlpsub, **kwargs)
            elif signal_type in self.NMF_ADDITION:
                if self.data_zlpsub_nmf is not None:
                    image_signals = np.copy(self.data_zlpsub_nmf)
                else:
                    image_signals = self.nmf_image(data=self.data_zlpsub, **kwargs)
            else:
                image_signals = np.copy(self.data_zlpsub)

        else:
            print("Either no such signal or the signal shape has not been calculated yet, "
                  "returned general EELS signal.")
            image_signals = np.copy(self.data)
        return image_signals

    def get_cluster_signals(self, conf_interval=1, signal_type='EELS', data_type='EELS', data=None, **kwargs):
        r"""
        Get the signals ordered per cluster. Cluster signals are stored in attribute ``self.cluster_signals``.
        Note that the pixel location information is lost.

        Parameters
        ----------
        conf_interval : float, optional
            The ratio of spectra returned. The spectra are selected based on the 
            based_on value. The default is 1.
        signal_type: str, optional
            Description of signal, ``'EELS'`` by default.

        Returns
        -------
        cluster_signals : numpy.ndarray, shape=(M,)
            An array with size equal to the number of clusters. Each entry is a
            2D array that contains all the spectra within that cluster.
        """

        integrated_int = np.sum(self.data, axis=2)
        cluster_signals = np.zeros(self.n_clusters, dtype=object)
        if data is None:
            data = self.get_image_signals(signal_type=signal_type, data_type=data_type, **kwargs)

        j = 0
        for i in range(self.n_clusters):
            cluster_signal = data[self.cluster_labels == i]
            if conf_interval < 1:
                intensities_cluster = integrated_int[self.cluster_labels == i]
                arg_sort_int = np.argsort(intensities_cluster)
                ci_lim = round((1 - conf_interval) / 2 *
                               intensities_cluster.size)
                cluster_signal = cluster_signal[arg_sort_int][ci_lim:-ci_lim]
            cluster_signals[j] = cluster_signal
            j += 1
        self.cluster_signals = cluster_signals
        return cluster_signals

    def cut_image_energy(self, e1=None, e2=None, include_edge=True):
        r"""
        Cuts the spectral image at ``E1`` and ``E2`` and keeps only the part in between.

        Parameters
        ----------
        e1 : float, optional
            lower cut. The default is ``None``, which means no cut is applied.
        e2 : float, optional
            upper cut. The default is ``None``, which means no cut is applied.
        include_edge : Boolean, optional
            If True, the edge values given in ``E1`` and ``E2`` are included in the cut result. Default is True
        """

        if (e1 is None) and (e2 is None):
            raise ValueError(
                "To cut energy spectra, please specify minimum energy E1 and/or\
                    maximum energy E2.")
        if e1 is None:
            e1 = self.eaxis.min() - 1
        if e2 is None:
            e2 = self.eaxis.max() + 1
        if include_edge is True:
            select = ((self.eaxis >= e1) & (self.eaxis <= e2))
        else:
            select = ((self.eaxis > e1) & (self.eaxis < e2))
        self.data = self.data[:, :, select]
        self.eaxis = self.eaxis[select]

    def cut_image_pixels(self, range_width, range_height):
        r"""
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

    def deconv_image(self, save_data=True, signal_type='EELS', **kwargs):
        r"""
        Deconvolute the signals of all pixels. Only the median value of the ZLP models is taken.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        data_deconv = np.zeros(self.shape)
        print("Start deconvolution")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                signal = self.get_pixel_signal(i, j, signal_type=signal_type)
                zlp_median = np.nanpercentile(
                    self.get_pixel_matched_zlp_models(i, j, signal_type=signal_type, signal=signal, **kwargs),
                    50, axis=0)
                data_deconv[i, j, :] = self.deconvolution(signal, zlp_median)
            print("row ", i,"done")
        self.data_deconv = data_deconv
        print("Finished deconvolution")
        if save_data is True:
            filename = self.output_path + 'data_deconv.npy'
            np.save(filename, data_deconv)

    def subtract_image(self, save_data=True, signal_type='EELS', **kwargs):
        r"""
        Subtract the ZLP from the signals of all pixels. Only the median value of the ZLP models is taken.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        data_zlpsub = np.zeros(self.shape)
        print("Start subtraction")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                signal = self.get_pixel_signal(i, j, signal_type=signal_type)
                zlp_median = np.nanpercentile(
                    self.get_pixel_matched_zlp_models(i, j, signal_type=signal_type, signal=signal, **kwargs),
                    50, axis=0)
                data_zlpsub[i, j, :] = self.subtract_zlp(signal, zlp_median)
            print("row ", i,"done")
        self.data_zlpsub = data_zlpsub
        print("Finished subtraction")
        if save_data is True:
            filename = self.output_path + 'data_zlpsub.npy'
            np.save(filename, data_zlpsub)

    # METHODS ON DATA MODIFICATION
    #TO DO remove pca_pixel, nmf_pixel, pca_cluster and nmf_cluster, these are not a correct application of the technique
    def pool_image(self, data, area=9, **kwargs):
        r"""
        Pools spectral image using a squared window of size ``area`` around each pixel

        Parameters
        ----------
        data: numpy.ndarray (M,N)
            2D data set.
        area: int
            Pooling parameter: area around the pixel, must be an odd number
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------

        """

        if area % 2 == 0:
            print("Unable to pool with even number " + str(area) + ", continuing with n_p=" + str(area + 1))
            area += 1
        if area > data.shape[0] or area > data.shape[1]:
            raise ValueError("Your pooling area is too large for one or both of the image axes, "
                             "please choose a number smaller than these values")
        print("Pooling of data started")
        data_pool = np.zeros(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_pool[i, j] = self.pool_pixel(data=data, i=i, j=j, area=area, **kwargs)
        if self.data_zlpsub is not None:
            self.data_pool_zlpsub = data_pool
        else:
            self.data_pool = data_pool
        print("Pooling of data complete")
        return data_pool

    def pool_pixel(self, data, i, j, area=9, gaussian=True, **kwargs):
        r"""
        Pools the data of a squared window of size ``area`` around pixel (``i``, ``j``).

        Parameters
        ----------
        i: int
            y-coordinate of the pixel
        j: int
            x-coordinate of the pixel
        area: int
            Pooling parameter: area around the pixel used for pooling, must be an odd number
        gaussian: boolean
            If true the pooling weights will use a gaussian distribution
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        output: numpy.ndarray, shape=(M,)
            Pooled spectrum of the pixel
        """

        if area % 2 == 0:
            print("Unable to pool with even number " + str(area) + ", continuing with n_p=" + str(area + 1))
            area += 1
        if area > data.shape[0] or area > data.shape[1]:
            raise ValueError("Your pooling area is too large for one or both of the image axes, "
                             "please choose a number smaller than these values")

        if gaussian is True:
            # Set up the pooling weights, basically a 1d gaussian to be used twice
            x = np.linspace(-1 * math.floor(area / 2), math.floor(area / 2), area)
            z = gauss1d(x, **kwargs)
            weights = z / np.max(z)
        else:
            x = np.ones(area)
            weights = np.ones(area)

        n_p_border = int(math.floor(area / 2))
        y_min = i - n_p_border
        y_max = i + n_p_border + 1
        x_min = j - n_p_border
        x_max = j + n_p_border + 1
        # Check if the center pixel is on the border
        if x_min < 0:
            z = gauss1d(x, mx=x_min, **kwargs)
            weights_x = z / np.max(z)
            x_max += np.abs(x_min)
            x_min = 0
        elif x_max > data.shape[1]:
            z = gauss1d(x, mx=(x_max - data.shape[1]), **kwargs)
            weights_x = z / np.max(z)
            x_min -= (x_max - data.shape[1])
            x_max = data.shape[1]
        else:
            weights_x = weights
        if y_min < 0:
            z = gauss1d(x, mx=y_min, **kwargs)
            weights_y = z / np.max(z)
            y_max += np.abs(y_min)
            y_min = 0
        elif y_max > data.shape[0]:
            z = gauss1d(x, mx=(y_max - data.shape[0]), **kwargs)
            weights_y = z / np.max(z)
            y_min -= (y_max - data.shape[0])
            y_max = data.shape[0]
        else:
            weights_y = weights
        if len(data.shape) == 2:
            pooled_pixel = np.average(np.average(data[y_min:y_max, x_min:x_max], axis=1, weights=weights_x),
                                      axis=0, weights=weights_y)
        else:
            pooled_pixel = np.average(np.average(data[y_min:y_max, x_min:x_max, :], axis=1, weights=weights_x),
                                      axis=0, weights=weights_y)
        return pooled_pixel

    def pca_image(self, data, area_type='segment', n_components=30, segments_x=1, segments_y=1, norm_poisson=True,
                  plot_variance=False, **kwargs):
        r"""
        Use principal component analysis on the spectral image.

        Parameters
        ----------
        area_type: str
            type of area used for principal component analysis. Usage types as follows:
                - ``'segment'``, the image is segmented and pca is only done per segmented areas.
                - ``'cluster'``, the data per cluster is used for pca within that cluster.
                - ``'pixel'``, only the data used around a pixel is used for pca of that pixel.
        n_components: float,
            number components to calculate. If between 0 and 1 the amount of components will be determined based on the
            sum of the variance of the components below the given value. Default is 0.9.
        segments_x: int
            For ``'segment'`` option, number of segments the x-axis is divided upon. Default is 1.
        segments_y: int
            For ``'segment'`` option, number of segments the y-axis is divided upon. Default is 1.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------

        """
        data = np.copy(data)
        if area_type == 'segment':
            print("PCA of segmented areas started")
            segsize_x = data.shape[1] / segments_x
            segsize_y = data.shape[0] / segments_y
            model_data = np.zeros(data.shape)
            for segment_y in np.arange(0, segments_y):
                for segment_x in np.arange(0, segments_x):
                    x_min = round(segsize_x * segment_x)
                    x_max = round(segsize_x * (segment_x + 1))
                    y_min = round(segsize_y * segment_y)
                    y_max = round(segsize_y * (segment_y + 1))
                    seg_data = data[y_min:y_max, x_min:x_max, :]
                    seg_data[seg_data < 0] = 0
                    X = seg_data.reshape(np.prod(seg_data.shape[:2]), seg_data.shape[2])
                    if norm_poisson:
                        # Normalize for poissonian noise
                        aG = X.sum(axis=1).squeeze()
                        bH = X.sum(axis=0).squeeze()
                        root_aG = np.sqrt(aG)[:, np.newaxis]
                        root_bH = np.sqrt(bH)[np.newaxis, :]
                        X /= root_aG * root_bH
                        X = np.nan_to_num(X)

                    model = PCA(n_components=n_components, svd_solver='full')
                    loadings = model.fit_transform(X)
                    factors = model.components_.T
                    if norm_poisson:
                        # rescale back the loadings and factors
                        loadings[:] *= root_aG
                        factors[:] *= root_bH.T
                    if segments_x == 1 and segments_y == 1:
                        # loadings_maps = loadings.reshape(seg_data_shape[:2] + (model.n_components_,))
                        if plot_variance:
                            fig = plot_pca_variance(components=model.n_components,
                                                    eigenvalues_ratio=model.explained_variance_ratio_, figsize=(8, 4))
                            fig.savefig(os.path.join(self.output_path, 'Scree_plot.pdf'))
                    X_model = factors @ loadings.T + model.mean_[:, np.newaxis]
                    model_data[y_min:y_max, x_min:x_max, :] = X_model.T.reshape(seg_data.shape)

        elif area_type == 'cluster':
            print("PCA of data per cluster started")
            model_data = np.zeros(data.shape)
            for cluster_idx in np.arange(0, len(self.cluster_centroids)):
                model_data[self.cluster_labels == cluster_idx] = self.pca_cluster(data=data, cluster=cluster_idx,
                                                                                  n_components=n_components,
                                                                                  norm_poisson=norm_poisson, **kwargs)
        elif area_type == 'pixel':
            print("PCA of data per pixel started")
            model_data = np.zeros(data.shape)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    model_data[i, j] = self.pca_pixel(data=data, i=i, j=j, n_components=n_components,
                                                      norm_poisson=norm_poisson, **kwargs)
        else:
            print("please pick a valid area type")
        print("PCA of data complete")
        return model_data

    def pca_pixel(self, data, i, j, area=9, n_components=30, norm_poisson=True, **kwargs):
        r"""
        Use principal component analysis on the spectral image, using the data of a squared window of size ``n_p``
        around pixel (``i``, ``j``).

        Parameters
        ----------
        i: int
            y-coordinate of the pixel
        j: int
            x-coordinate of the pixel
        area: int
            PCA area parameter. Area around the pixel used for principal component analysis, must be an odd number
        n_components: float,
            number components to calculate. If between 0 and 1 the amount of components will be determined based on the
            sum of the variance of the components below the given value. Default is 0.9.
        deconv
        zlp_num

        Returns
        -------
        output: numpy.ndarray, shape=(M,)
            PCA spectrum of the pixel
        """

        if area % 2 == 0:
            print("Unable to PCA with even number " + str(area) + ", continuing with n_area=" + str(area + 1))
            area += 1
        if area > data.shape[0] or area > data.shape[1]:
            raise ValueError("Your pixel area is too large for one or both of the image axes, "
                             "please choose a number smaller than these values")

        # Check pixel segment for borders
        n_area_border = int(math.floor(area / 2))
        x_min = j - n_area_border
        x_max = j + n_area_border + 1
        y_min = i - n_area_border
        y_max = i + n_area_border + 1
        if x_min < 0:
            x_max += np.abs(x_min)
            x_min = 0
        if x_max > data.shape[1]:
            x_min -= (x_max - data.shape[1])
            x_max = data.shape[1]
        if y_min < 0:
            y_max += np.abs(y_min)
            y_min = 0
        if y_max > data.shape[0]:
            y_min -= (y_max - data.shape[0])
            y_max = data.shape[0]
        seg_data = data[y_min:y_max, x_min:x_max, :]
        seg_data[seg_data < 0] = 0
        X = seg_data.reshape(np.prod(seg_data.shape[:2]), seg_data.shape[2])
        if norm_poisson:
            # Normalize for poissonian noise
            aG = X.sum(axis=1).squeeze()
            bH = X.sum(axis=0).squeeze()
            root_aG = np.sqrt(aG)[:, np.newaxis]
            root_bH = np.sqrt(bH)[np.newaxis, :]
            X /= root_aG * root_bH
            X = np.nan_to_num(X)

        model = PCA(n_components=n_components, svd_solver='full')
        loadings = model.fit_transform(X)
        factors = model.components_.T
        if norm_poisson:
            # rescale back the loadings and factors
            loadings[:] *= root_aG
            factors[:] *= root_bH.T
        X_model = factors @ loadings.T + model.mean_[:, np.newaxis]
        model_data = X_model.T.reshape(seg_data.shape)
        model_pixel = model_data[int(i - y_min), int(j - x_min)]
        return model_pixel

    def pca_cluster(self, data, cluster, n_components=30, norm_poisson=True, **kwargs):
        r"""
        Use principal component analysis on a cluster of the spectral image. The signals of the cluster are already
        in reduced format (pixel location is lost).

        Parameters
        ----------
        cluster : numpy.ndarray, shape=(M,)
            An array with size equal to the number of clusters. Each entry is a 2D array that contains all the spectra
            within that cluster.
        n_components: float,
            number components to calculate. If between 0 and 1 the amount of components will be determined based on the
            sum of the variance of the components below the given value. Default is 0.9.

        Returns
        -------

        """

        cluster_data = self.get_cluster_signals(data=data, **kwargs)[cluster]
        cluster_data[cluster_data < 0] = 0
        X = np.copy(cluster_data)
        if norm_poisson:
            # Normalize for poissonian noise
            aG = X.sum(axis=1).squeeze()
            bH = X.sum(axis=0).squeeze()
            root_aG = np.sqrt(aG)[:, np.newaxis]
            root_bH = np.sqrt(bH)[np.newaxis, :]
            X /= root_aG * root_bH
            X = np.nan_to_num(X)

        model = PCA(n_components=n_components, svd_solver='full')
        loadings = model.fit_transform(X)
        factors = model.components_.T
        if norm_poisson:
            # rescale back the loadings and factors
            loadings[:] *= root_aG
            factors[:] *= root_bH.T
        X_model = factors @ loadings.T + model.mean_[:, np.newaxis]
        model_data = X_model
        return model_data

    def nmf_image(self, data, area_type='segment', n_components=30, max_iter=100000, segments_x=1, segments_y=1,
                  norm_poisson=True, **kwargs):
        r"""
        Use non-negative matrix factorization on the spectral image.

        Parameters
        ----------
        area_type: str
            type of area used for principal component analysis. Usage types as follows:
                - ``'segment'``, the image is segmented and nmf is only done per segmented areas.
                - ``'cluster'``, the data per cluster is used for nmf within that cluster.
                - ``'pixel'``, only the data used around a pixel is used for nmf of that pixel.
        n_components: float,
            number components to calculate. If between 0 and 1 the amount of components will be determined based on the
            sum of the variance of the components below the given value by first running PCA. Default is 0.9.
        max_iter: int,
            Default is 100000
        segments_x: int
            For ``'segment'`` option, number of segments the x-axis is divided upon. Default is 1.
        segments_y: int
            For ``'segment'`` option, number of segments the y-axis is divided upon. Default is 1.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------

        """
        data = np.copy(data)
        if area_type == 'segment':
            print("NMF of segmented areas started")
            segsize_x = data.shape[1] / segments_x
            segsize_y = data.shape[0] / segments_y
            model_data = np.zeros(data.shape)
            for segment_y in np.arange(0, segments_y):
                for segment_x in np.arange(0, segments_x):
                    x_min = round(segsize_x * segment_x)
                    x_max = round(segsize_x * (segment_x + 1))
                    y_min = round(segsize_y * segment_y)
                    y_max = round(segsize_y * (segment_y + 1))
                    seg_data = data[y_min:y_max, x_min:x_max, :]
                    seg_data[seg_data < 0] = 0
                    X = seg_data.reshape(np.prod(seg_data.shape[:2]), seg_data.shape[2])
                    if norm_poisson:
                        # Normalize for poissonian noise
                        aG = X.sum(axis=1).squeeze()
                        bH = X.sum(axis=0).squeeze()
                        root_aG = np.sqrt(aG)[:, np.newaxis]
                        root_bH = np.sqrt(bH)[np.newaxis, :]
                        X /= root_aG * root_bH
                        X = np.nan_to_num(X)
                    if n_components < 1:
                        model_pca = PCA(n_components=n_components, svd_solver='full').fit(X)
                        if model_pca.n_components_ < 2:
                            n_components = 2
                    model = NMF(n_components=n_components, max_iter=max_iter)
                    loadings = model.fit_transform(X)
                    factors = model.components_.T
                    if norm_poisson:
                        # rescale back the loadings and factors
                        loadings[:] *= root_aG
                        factors[:] *= root_bH.T
                    # if segments_x == 1 and segments_y == 1:
                    #     loadings_maps = loadings.reshape(seg_data_shape[:2] + (model.n_components_,))
                    X_model = factors @ loadings.T
                    model_data[y_min:y_max, x_min:x_max, :] = X_model.T.reshape(seg_data.shape)

        elif area_type == 'cluster':
            print("NMF of data per cluster started")
            model_data = np.zeros(data.shape)
            for cluster_idx in np.arange(0, len(self.cluster_centroids)):
                model_data[self.cluster_labels == cluster_idx] = self.nmf_cluster(data=data, cluster=cluster_idx,
                                                                                  n_components=n_components,
                                                                                  max_iter=max_iter, **kwargs)
        elif area_type == 'pixel':
            print("NMF of data per pixel started")
            model_data = np.zeros(data.shape)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    model_data[i, j] = self.nmf_pixel(data=data, i=i, j=j, n_components=n_components, **kwargs)
        else:
            print("please pick a valid area type")
        print("NMF of data complete")
        return model_data

    def nmf_cluster(self, data, cluster, n_components=30, max_iter=100000, norm_poisson=True, **kwargs):
        r"""
        Use non-negative matrix factorization on a cluster of the spectral image.
        The signals of the cluster are already in reduced format (pixel location is lost).

        Parameters
        ----------
        cluster : numpy.ndarray, shape=(M,)
            An array with size equal to the number of clusters. Each entry is a
            2D array that contains all the spectra within that cluster.
        n_components: float,
            number components to calculate. If between 0 and 1 the amount of components will be determined based on the
            sum of the variance of the components below the given value by PCA first. Default is 0.9.
        max_iter: int,
            Default is 100000

        Returns
        -------

        """

        cluster_data = self.get_cluster_signals(data=data, **kwargs)[cluster]
        cluster_data[cluster_data < 0] = 0
        X = np.copy(cluster_data)
        if norm_poisson:
            # Normalize for poissonian noise
            aG = X.sum(axis=1).squeeze()
            bH = X.sum(axis=0).squeeze()
            root_aG = np.sqrt(aG)[:, np.newaxis]
            root_bH = np.sqrt(bH)[np.newaxis, :]
            X /= root_aG * root_bH
            X = np.nan_to_num(X)
        if n_components < 1:
            model_pca = PCA(n_components=n_components, svd_solver='full').fit(X)
            if model_pca.n_components_ < 2:
                n_components = 2
        model = NMF(n_components=n_components, max_iter=max_iter)
        loadings = model.fit_transform(X)
        factors = model.components_.T
        if norm_poisson:
            # rescale back the loadings and factors
            loadings[:] *= root_aG
            factors[:] *= root_bH.T
        X_model = factors @ loadings.T
        model_data = X_model
        return model_data

    def nmf_pixel(self, data, i, j, area=9, n_components=30, max_iter=100000, norm_poisson=True, **kwargs):
        r"""
        Use principal component analysis on the spectral image, using the data of a squared window of size ``n_p``
        around pixel (``i``, ``j``).

        Parameters
        ----------
        i: int
            y-coordinate of the pixel
        j: int
            x-coordinate of the pixel
        area: int
            PCA area parameter. Area around the pixel used for principal component analysis, must be an odd number
        n_components: float,
            number components to calculate. If between 0 and 1 the amount of components will be determined based on the
            sum of the variance of the components below the given value by PCA first. Default is 0.9.
        max_iter: int,
            Default is 100000

        Returns
        -------
        output: numpy.ndarray, shape=(M,)
            PCA spectrum of the pixel
        """

        if area % 2 == 0:
            print("Unable to NMF with even number " + str(area) + ", continuing with n_area=" + str(area + 1))
            area += 1
        if area > data.shape[0] or area > data.shape[1]:
            raise ValueError("Your pixel area is too large for one or both of the image axes, "
                             "please choose a number smaller than these values")

        # Check pixel area
        n_area_border = int(math.floor(area / 2))
        x_min = j - n_area_border
        x_max = j + n_area_border + 1
        y_min = i - n_area_border
        y_max = i + n_area_border + 1
        if x_min < 0:
            x_max += np.abs(x_min)
            x_min = 0
        if x_max > data.shape[1]:
            x_min -= (x_max - data.shape[1])
            x_max = data.shape[1]
        if y_min < 0:
            y_max += np.abs(y_min)
            y_min = 0
        if y_max > data.shape[0]:
            y_min -= (y_max - data.shape[0])
            y_max = data.shape[0]
        seg_data = data[y_min:y_max, x_min:x_max, :]
        seg_data[seg_data < 0] = 0
        X = seg_data.reshape(np.prod(seg_data.shape[:2]), seg_data.shape[2])
        if norm_poisson:
            # Normalize for poissonian noise
            aG = X.sum(axis=1).squeeze()
            bH = X.sum(axis=0).squeeze()
            root_aG = np.sqrt(aG)[:, np.newaxis]
            root_bH = np.sqrt(bH)[np.newaxis, :]
            X /= root_aG * root_bH
            X = np.nan_to_num(X)

        if n_components < 1:
            model_pca = PCA(n_components=n_components, svd_solver='full').fit(X)
            if model_pca.n_components_ < 2:
                n_components = 2
        model = NMF(n_components=n_components, max_iter=max_iter)
        loadings = model.fit_transform(X)
        factors = model.components_.T

        if norm_poisson:
            # rescale back the loadings and factors
            loadings[:] *= root_aG
            factors[:] *= root_bH.T
        X_model = factors @ loadings.T
        model_data = X_model.T.reshape(seg_data.shape)
        model_pixel = model_data[int(i - y_min), int(j - x_min)]
        return model_pixel

    # METHODS ON SIGNAL
    @staticmethod
    def smooth_signal(signal, window_length=51, window_type='hanning', beta=14.0, polyorder=2, **kwargs):
        r"""
        Smooth a signal using a window length ``window_length`` and a window type ``window_type``.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.

        Parameters
        ----------
        signal: numpy.ndarray, shape=(M,)
            Signal of length M
        window_length: int, optional
            The dimension of the smoothing window; should be an odd integer. Default is 51.
        window_type: str, optional
            the type of window from ``'flat'``, ``'hanning'``, ``'hamming'``, ``'bartlett'``,
            ``'blackman'`` and ``'kaiser'``. ``'flat'`` window will produce a moving average smoothing.
            Default is ``'hanning'``.
        beta: float, optional
            If using the kaiser window, beta determines the shape parameter for window.

        Returns
        -------
        signal_smooth: numpy.ndarray, shape=(M,)
            The smoothed signal
        """

        # Set window length to uneven number
        window_length += (window_length + 1) % 2

        # extend the signal with the window length on both ends
        signal_padded = np.r_['-1', signal[window_length - 1:0:-1], signal, signal[-2:-window_length - 1:-1]]

        if window_type == 'savgol':
            signal_smooth = savgol_filter(x=signal, window_length=window_length, polyorder=polyorder)
        else:
            # Pick the window type
            if window_type == 'flat':  # moving average
                window = np.ones(window_length, 'd')
            elif window_type == 'kaiser':
                (window) = np.kaiser(M=window_length, beta=beta)
            else:
                window = eval('np.' + window_type + '(window_length)')
            # Determine the smoothed signal and throw away the padded ends
            surplus_data = int((window_length - 1) * 0.5)
            signal_smooth = np.convolve(signal_padded, window / window.sum(), mode='valid')[
                                surplus_data:-surplus_data]
        return signal_smooth

    def deconvolution(self, signal, zlp, correction=True):
        r"""
        Perform deconvolution on a given signal with a given Zero Loss Peak. This removes both the ZLP and plural
        scattering. Based on the Fourier Log Method :cite:p:`Johnson1974, Egerton2011`

        Parameters
        ----------
        signal: numpy.ndarray, shape=(M,)
            Raw signal of length M
        zlp: numpy.ndarray, shape=(M,)
            zero-loss peak of length M
        correction: Bool
            Sometimes a decreasing linear slope occurs on the place of the ZLP after deconvolution. This correction fits
            a linear function and subtracts that from the signal. Default is True.

        Returns
        -------
        output: numpy.ndarray, shape=(M,)
            deconvoluted spectrum.

        """

        # Extrapolate data
        x = self.eaxis_extrp
        y_zlp = np.zeros(len(x))
        y_zlp[:self.shape[2]] = zlp
        if len(signal) < len(self.eaxis_extrp):
            y_signal = self.extrp_signal(signal=signal)
        else:
            y_signal = signal

        # Fourier log method with Zero-loss modifier. See Egerton Chapter 4 for details
        z_nu = cft(x, y_zlp)
        j_nu = cft(x, y_signal)
        j1_nu = z_nu * np.log(j_nu / z_nu)
        J1_E = np.real(icft(x, j1_nu))

        if correction is True:
            # Correct for linear increase in ZLP and gain region.
            zlp_peak = np.max(signal)
            fwhm_idx1 = np.argwhere((signal >= 0.5 * zlp_peak))[-1] + 1
            fwhm_idx2 = np.argwhere((signal >= 0.5 * zlp_peak))[0] - 1
            fwhm = (self.eaxis[fwhm_idx1] - self.eaxis[fwhm_idx2])

            dydx1_idx = int(np.argwhere(x >= -1 * fwhm)[0])
            dydx2_idx = int(np.argwhere(x < fwhm)[-1])

            x_fit = np.array(x[dydx1_idx:dydx2_idx], dtype='float64')
            y_fit = np.array(J1_E[dydx1_idx:dydx2_idx], dtype='float64')
            popt, pcov = curve_fit(f=linear_fit, xdata=x_fit, ydata=y_fit, bounds=([-np.inf, 0], [0, np.inf]))

            deconv_corr = linear_fit(x, popt[0], popt[1])
            deconv_corr[deconv_corr < 0] = 0

            # Apply correction
            J1_E = J1_E - deconv_corr

        J1_E[J1_E < 0] = 0
        J1_E[y_zlp == y_signal] = 0

        return J1_E[:self.shape[2]].flatten()

    def rl_deconvolution(self, signal, zlp, iterations=15):
        """
        Richardson-lucy deconvolution
        Parameters
        ----------
        signal
        zlp
        iterations

        Returns
        -------

        """
        zlp_ch = zlp.shape[0]
        max_idx = zlp.argmax()
        signal_rl = np.array(signal).copy()
        mimax_idx = zlp_ch - 1 - max_idx
        for _ in range(iterations):
            first = np.convolve(zlp, signal_rl)[max_idx: max_idx + zlp_ch]
            signal_rl *= np.convolve(zlp[::-1], signal / first)[mimax_idx: mimax_idx + zlp_ch]
        return signal_rl

    def subtract_zlp(self, signal, zlp):
        r"""
        Subtract the Zero Loss Peak from the signal

        Parameters
        ----------
        signal: numpy.ndarray, shape=(M,)
            Raw signal of length M
        zlp: numpy.ndarray, shape=(M,)
            zero-loss peak of length M

        Returns
        -------

        """
        return signal - zlp

    def get_extrp_param(self, signal, range_perc=0.1):
        r"""
        Retrieve the extrapolation parameter from the last 10% of a given signal

        Parameters
        ----------
        signal: numpy.ndarray, shape=(M,)
        range_perc: float, optional

        Returns
        -------
        r: float.
            extrapolation parameter

        """
        idx_last = int((1 - range_perc) * len(signal))
        x_fit = np.array(self.eaxis[idx_last:], dtype='float64')
        y_fit = np.array(signal[idx_last:], dtype='float64')
        try:
            popt, pcov = curve_fit(power_fit, x_fit, y_fit, bounds=([0, -np.inf], [np.inf, -1]))
            r = popt[1]
        except:
            r = -1
        return r

    def extrp_signal(self, signal, r=None):
        r"""
        Extrapolate your signal. Extrapolation model, generate vanishing to zero data after the real exp. data
        See Egerton paragraph 4.2.2 for details. extrapolation of the form A*E^-r is used.

        Parameters
        ----------
        signal : numpy.ndarray, shape=(M,)
            spectrum
        r : float, optional
            extrapolation parameter

        Returns
        -------

        """

        x = self.eaxis_extrp
        y = np.zeros(len(x))
        y[:self.shape[2]] = signal
        a = signal[-1]  # Starting amplitude for extrapolated data, endpoint of the exp. data
        if r is None:
            r = self.get_extrp_param(signal=signal)
        y[self.shape[2]:] = power_fit(1 + x[self.shape[2]:] - x[self.shape[2]], a, r)
        return y

    # METHODS ON ZLP
    def get_pixel_matched_zlp_models(self, i, j, signal_type='EELS', signal=None, **kwargs):
        r"""
        Returns the shape-(M, N) array of matched ZLP model predictions at pixel
        (``i``, ``j``) after training. M and N correspond to the number of model
        predictions and :math:`\Delta E` s respectively.

        Parameters
        ----------
        i: int
            y-coordinate of the pixel.
        j: int
            x-coordinate of the pixel.
        signal_type: str, bool
            Description of signal type. Set to ``'EELS'`` by default.
        signal: array, bool,
            signal you want to match the zlps to. Important to do if you did not do any pooling, pca or nmf on the whole
            image, otherwise it will calculate the denoised signal twice.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        predictions: numpy.ndarray, shape=(M, N)
            The matched ZLP predictions at pixel (``i``, ``j``).
        """

        # get pixel information
        if signal is None:
            signal = self.get_pixel_signal(i=i, j=j, signal_type=signal_type, **kwargs)
        cluster = self.cluster_labels[i, j]
        de1 = self.dE1[int(cluster)]
        de2 = self.dE2[int(cluster)]
        fwhm = self.FWHM[int(cluster)]

        # get zlp predictions
        max_idx = np.argmax(signal)
        int_i = np.sum(signal[max_idx - 1:max_idx + 2])
        predictions = self.get_zlp_models(int_i=int_i, **kwargs)

        # match the predictions
        predictions_matched = np.zeros((len(predictions), self.shape[2]))
        for m, prediction in enumerate(predictions):
            predictions_matched[m, :] = self.match_zlp_to_signal(signal, prediction, de1, de2, fwhm)
        return predictions_matched

    def get_zlp_models(self, int_i, **kwargs):
        r"""
        Returns the shape-(M, N) array of zlp model predictions at the
        integrated intensity ``int_i``. The logarithm of the integrated intensity is taken,
        as the data is always trained to log of the signal.

        Parameters
        ----------
        int_i: float
            Integrated intensity
        """

        if self.zlp_models is None:
            try:
                self.load_zlp_models(**kwargs)
            except:
                self.load_zlp_models()

        if self.scale_var_eaxis is None:
            self.scale_var_eaxis = find_scale_var(self.eaxis)

        if self.scale_var_log_int_i is None:
            all_spectra = self.data
            all_spectra[all_spectra < 1] = 1
            log_int_i = np.log(np.sum(all_spectra, axis=2)).flatten()
            self.scale_var_log_int_i = find_scale_var(log_int_i)
            del all_spectra

        # Prepare the neural network input features (scaled eaxis and scaled log of the integrated intensity)
        log_int_i = np.log(int_i)
        predict_x_np = np.zeros((self.shape[2], 2))
        predict_x_np[:, 0] = scale(self.eaxis, self.scale_var_eaxis)
        predict_x_np[:, 1] = scale(log_int_i, self.scale_var_log_int_i)
        predict_x = torch.from_numpy(predict_x_np)

        # Get the model predictions based on the neural network input features
        predictions = np.zeros((len(self.zlp_models), self.shape[2]))
        for i, model in enumerate(self.zlp_models):
            with torch.no_grad():
                predictions[i, :] = np.exp(model(predict_x.float()).flatten())[:self.shape[2]]

        # Post-fit selection, if the prediction goes upward after the zlp peak, it is filtered out.
        # predictions_filtered = []
        # for j, prediction in enumerate(predictions):
        #     pred_lim = prediction[np.argwhere(self.eaxis > self.eaxis[np.argmax(prediction) + 1]).flatten()]
        #     if not any(np.diff(pred_lim) > 0.01):
        #         predictions_filtered.append(prediction)
        # predictions_filtered = np.array(predictions_filtered)

        return predictions # predictions_filtered

    def match_zlp_to_signal(self, signal, zlp, de1, de2, fwhm=None):
        r"""
        Apply the matching to the subtracted spectrum.

        Parameters
        ----------
        signal: numpy.ndarray, shape=(M,)
            Signal to be matched
        zlp: numpy.ndarray, shape=(M,)
            ZLP model to be matched, must match length of Signal.
        de1: float
            Value of the hyperparameter :math:`\Delta E_{I}`
        de2: float
            Value of the hyperparameter :math:`\Delta E_{II}`
        fwhm: float
            Value of the hyperparameter :math:`FWHM`. If none is given, a fwhm is determined from the signal.
            Default is None

        Returns
        -------
        output: numpy.ndarray, shape=(M,)
            Matched ZLP model
        """

        peak = np.max(signal)
        fwhm_window = np.argwhere(signal > 0.5*peak)
        fwhm_idx1 = fwhm_window.flatten()[0] - 1
        fwhm_idx2 = fwhm_window.flatten()[-1] + 1
        fwhm = self.eaxis[fwhm_idx2] - self.eaxis[fwhm_idx1]

        de0 = (fwhm + de1) / 2
        de0_m = -1 * de0
        de1_m = -1 * de1

        # Right side ZLP (loss spectrum)
        e_window = self.eaxis[(self.eaxis < de1) & (self.eaxis >= de0)]
        # delta = (de1 - de0) / 3.
        # factor_nn = np.exp(- (e_window - de1)**2 / delta**2)
        # factor_zlp = 1 - factor_model
        delta = (de1 - de0) / 10.
        factor_nn = 1 / (1 + np.exp(-(e_window - (de0 + de1) / 2) / delta))
        factor_zlp = 1 - factor_nn

        # Left side ZLP (gain spectrum)
        e_window_m = self.eaxis[(self.eaxis > de1_m) & (self.eaxis <= de0_m)]
        # delta_m = (de1_m + de0_m) / 3.
        # factor_nn_m = np.exp(- (e_window_m - de1_m)**2 / delta_m**2)
        # factor_zlp_m = 1 - factor_model
        delta_m = (de1_m + de0_m) / 10.
        factor_nn_m = 1 / (1 + np.exp(-(e_window_m - (de0_m + de1_m) / 2) / delta_m))
        factor_zlp_m = 1 - factor_nn_m

        # Match the ZLP to signal using the factors
        range_m2 = zlp[self.eaxis <= de1_m]
        range_m1 = zlp[(self.eaxis > de1_m) & (self.eaxis <= de0_m)] * factor_nn_m + \
                   signal[(self.eaxis > de1_m) & (self.eaxis <= de0_m)] * factor_zlp_m
        range_0 = signal[(self.eaxis > de0_m) & (self.eaxis < de0)]     # ZLP
        range_1 = zlp[(self.eaxis < de1) & (self.eaxis >= de0)] * factor_nn + \
                  signal[(self.eaxis < de1) & (self.eaxis >= de0)] * factor_zlp
        range_2 = zlp[(self.eaxis >= de1) & (self.eaxis <= de2)]
        range_3 = zlp[(self.eaxis > de2)] * 0   # Rest of the spectrum

        matched_zlp = np.minimum(
            np.concatenate((range_m2, range_m1, range_0, range_1, range_2, range_3),
                           axis=0), signal)
        return matched_zlp

    def train_zlp_models(self, conf_interval=1, lr=1e-3, signal_type='EELS', **kwargs):
        r"""
        Train the ZLP on the spectral image.

        The spectral image is clustered in ``n_clusters`` clusters, according to
        e.g. the integrated intensity or thickness. A random spectrum is then
        taken from each cluster, which together defines one replica. The
        training is initiated by calling
        :py:meth:`train_zlp_models_scaled() <EELSFitter.core.training.train_zlp_models_scaled>`.

        Parameters
        ----------
        conf_interval: int, optional
            Default is 1
        lr: float, optional
            Default is 1
        signal_type: str, optional
            Type of spectrum. Set to EELS by default.
        **kwargs
            Additional keyword arguments that are passed to the method
            :py:meth:`train_zlp_models_scaled() <EELSFitter.core.training.train_zlp_models_scaled>`
            in the :py:mod:`training` module.
        """

        self.cluster(**kwargs)
        self.training_data = self.get_cluster_signals(conf_interval=conf_interval, signal_type=signal_type)
        self.train_zlps = TrainZeroLossPeak(spectra=self.training_data, eaxis=self.eaxis,
                                            cluster_centroids=self.cluster_centroids, **kwargs)
        self.train_zlps.train_zlp_models_scaled(lr=lr)


    def load_zlp_models(self, path_to_models, plot_chi2=False, plot_pred=False, idx=None, **kwargs):
        r"""
        Loads the trained ZLP models and stores them in ``self.zlp_models``.
        Models that have a :math:`\chi^2 > \chi^2_{\mathrm{mean}} + 5\sigma` are
        discarded, where :math:`\sigma` denotes the 68% CI.

        Parameters
        ----------
        path_to_models: str
            Location where the model predictions have been stored after training.
        plot_chi2: bool, optional
            When set to `True`, plot and save the :math:`\chi^2` distribution.
        plot_pred: bool, optional
            When set to `True`, plot and save the ZLP predictions per cluster.
        idx: int, optional
            When specified, only the zlp labelled by ``idx`` is loaded, instead
            of all model predictions.

        """

        if not os.path.exists(path_to_models):
            print(
                "No path " + path_to_models + " found. Please ensure spelling \
                    and that there are models trained.")
            return
        path_to_models += (path_to_models[-1] != '/') * '/'

        # Load in the hyperparameters
        path_hp = 'hyperparameters.txt'
        hypar = np.loadtxt(os.path.join(path_to_models, path_hp), ndmin=2)
        self.dE1 = hypar[1, :]
        self.dE2 = hypar[2, :]
        self.FWHM = hypar[3, :]
        print("Loading hyper-parameters complete")

        # Load in the scale variables for scaling on the log of the integrated intensity of the spectra.
        path_scale_var = 'scale_var.txt'
        self.scale_var_log_int_i = np.loadtxt(os.path.join(path_to_models, path_scale_var))
        print("Loading scale variables for zlp models complete")

        # Cluster image based on the models centroids
        self.cluster_on_centroids(hypar[0, :], **kwargs)
        print("Clustering based on cluster centroids complete")

        # Load in the models
        self.zlp_models = []
        model = MultilayerPerceptron(num_inputs=2, num_outputs=1)
        if idx is not None:
            with torch.no_grad():
                model.load_state_dict(torch.load(os.path.join(
                    path_to_models, 'nn_replicas'))[f'model_{idx + 1}'])
            self.zlp_models.append(copy.deepcopy(model))
            return

        filename_test = os.path.join(path_to_models, 'costs_test.txt')
        cost_tests = np.loadtxt(filename_test)
        cost_tests_mean = np.mean(cost_tests)
        cost_tests_std = np.percentile(cost_tests, 68)
        threshold_costs_tests = cost_tests_mean + 5 * cost_tests_std
        cost_tests = cost_tests[cost_tests < threshold_costs_tests]

        filename_train = os.path.join(path_to_models, 'costs_train.txt')
        cost_trains = np.loadtxt(filename_train)
        cost_trains_mean = np.mean(cost_trains)
        cost_trains_std = np.percentile(cost_trains, 68)
        threshold_costs_trains = cost_trains_mean + 5 * cost_trains_std

        nn_rep_idx = np.argwhere(cost_trains < threshold_costs_trains)
        cost_trains = cost_trains[cost_trains < threshold_costs_trains]

        nn_replicas_path = os.path.join(path_to_models, 'nn_replicas')
        checkpoint = torch.load(nn_replicas_path)
        for idx in nn_rep_idx.flatten():
            model.load_state_dict(checkpoint[f'model_{idx + 1}'])
            self.zlp_models.append(copy.deepcopy(model))

        print("Loading models complete")

        # plot the chi2 distributions
        if plot_chi2:
            fig = plot_cost_dist(cost_trains, cost_tests, cost_tests_std)
            fig.savefig(os.path.join(self.output_path, 'chi2_dist.pdf'))
            print("Chi2 plot saved at {}".format(self.output_path))

        # plot the zlp predictions for each cluster
        if plot_pred:
            fig = plot_zlp_cluster_predictions(image=self, dpi=500, xlim=[self.eaxis[0], np.median(self.dE2)], x=1,
                                               yscale='log', xlabel=r"$\rm{Energy\;loss\;[eV]}$",
                                               title=r"$\rm{Cluster\;predictions\;}$")
            fig.savefig(os.path.join(self.output_path, 'Cluster_predictions.pdf'))
            print("Cluster predictions plot saved at {}".format(self.output_path))

    # METHODS ON QUANTITATIVE ANALYSIS
    def set_refractive_index(self, n=None, n_background=None):
        r"""
        Sets value of refractive index for the image as attribute self.n. If not clustered, n will be an
        array of length one, otherwise it is an array of length n_clusters. If n_background is defined,
        the cluster with the lowest thickness (cluster 0) will be assumed to be the vacuum/background,
        and gets the value of the background refractive index.

        If there are more specimen present in the image, it is wise to check by hand what cluster belongs
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

        if n is None:
            self.n = None
        elif type(n) == float or type(n) == int:
            self.n = np.ones(self.n_clusters) * n
            if n_background is not None:
                # assume cluster 0 is background
                self.n[0] = n_background
        elif len(n) == self.n_clusters:
            self.n = n

    def set_mass_density(self, rho=None, rho_background=None):
        r"""
        Sets value of mass density for the image as attribute self.rho. If not clustered, rho will be an
        array of length one, otherwise it is an array of length n_clusters. If rho_background is defined,
        the cluster with the lowest thickness (cluster 0) will be assumed to be the vacuum/background,
        and gets the value of the background mass density.

        If there are more specimen present in the image, it is wise to check by hand what cluster belongs
        to what specimen, and set the values by running::

             image.n[cluster_i] = n_i

        Parameters
        ----------
        rho
        rho_background

        Returns
        -------

        """

        if rho is None:
            self.rho = None
        elif type(rho) == float or type(rho) == int:
            self.rho = np.ones(self.n_clusters) * rho
            if rho_background is not None:
                self.rho[0] = rho_background
        elif len(rho) == self.n_clusters:
            self.rho = rho

    def calc_thickness(self, signal, n=None, rho=None, n_zlp=None):
        r"""
        Calculates thickness from sample data by one of two methods:
            - Kramer-Kronig sum rule using the refractive index :cite:p:`Egerton1987`
            - Log ratio using mass density :cite:p:`Iakoubovskii2008a`

        Parameters
        ----------
        signal : numpy.ndarray, shape=(M,)
            spectrum
        n : float
            refraction index
        rho : float
            mass density
        n_zlp: float or int
            Set to 1 by default, for already normalized EELS spectra.

        Returns
        -------
        t: float
            thickness

        Notes
        -----
        If using the refractive index, surface scatterings are not corrected for.
        If you wish to correct for surface scatterings, please extract the thickness ``t`` from
        :py:meth:`kramers_kronig_analysis() <EELSFitter.core.spectral_image.SpectralImage.kramers_kronig_analysis>`.
        """

        # Microscope data
        e0 = self.beam_energy  # Electron gun voltage, KeV
        beta = self.collection_angle  # Collection semi-angle, mrad
        alpha = self.convergence_angle  # Convergence semi-angle, mrad

        # Constants & kinetic definitions (Egerton 2011 appendix E)
        m_e = self.m_e / 1E3  # Electron mass, KeV
        a0 = self.a0 * 1E9  # Bohr radius, nm
        gamma = 1 + (e0 / m_e)  # Relativistic factor
        T_eff = e0 * ((1 + gamma) / (2 * gamma ** 2))  # Effective kinetic energy, KeV

        # Check if data is extrapolated
        x = self.eaxis_extrp
        if len(signal) < len(x):
            y = self.extrp_signal(signal=signal)
        else:
            y = signal

        if n is None and rho is None:
            raise ValueError("The mass density and the refractive index are "
                             "not defined. Please provide one of them.")
        elif n is not None and rho is not None:
            raise ValueError("Please provide the refractive index OR the "
                             "mass density information, not both")
        elif n is not None:
            # Prepare single scattering distribution, Only take values from E = 0 onward
            x_spliced = copy.deepcopy(x[x > 0])
            y_spliced = copy.deepcopy(y[x > 0])

            # Calculation of the ELF by normalization of the SSD
            # First perform Angular corrections (Egerton 2011 section 4.2.1)
            theta_e = x_spliced / (2 * gamma * T_eff)  # Characteristic scattering angle (per energy loss), mrad
            ang_cor = np.log(1 + (beta / theta_e) ** 2)  # Angular/Aperture correction
            Im = y_spliced / ang_cor / self.deltaE

            # Thickness calculation
            Im_sum = np.sum(Im / x_spliced) * self.deltaE
            K = (Im_sum / (np.pi / 2) / (1 - 1. / n ** 2))  # proportionality constant (Egerton 2011 section 4.2.2)
            t = (2 * np.pi * a0 * K * T_eff * 1E3) / n_zlp
        elif rho is not None:
            # Inelastic mean free path, based on Iakoubovskii et al. (2008) with gamma scaling suggestion from Egerton
            theta_c = 20  # Effective cutoff angle, mrad
            if theta_c > beta:
                theta_c = beta
            theta_e = 5.5 * rho ** 0.3 / (gamma * T_eff)  # Characteristic scattering angle (from mass density), mrad

            # Calculate lamba, the mean free path
            upper_term = alpha ** 2 + beta ** 2 + 2 * theta_e ** 2 + np.abs(alpha ** 2 - beta ** 2)
            lower_term = alpha ** 2 + beta ** 2 + 2 * theta_c ** 2 + np.abs(alpha ** 2 - beta ** 2)
            lmbda = (100 / theta_e) / np.log((upper_term / lower_term) * (theta_c ** 2 / theta_e ** 2))

            # Thickness calculation
            t = lmbda * np.log(np.sum(y) / n_zlp)

        return t

    def kramers_kronig_analysis(self, signal_ssd, n_zlp=None, iterations=1, n=None, t=None, delta=0.5):
        r"""
        Computes the complex dielectric function from the single scattering distribution (SSD) ``signal_ssd`` following
        the Kramers-Krönig relations. This code is based on Egerton's MATlab code :cite:p:`Egerton2011`.

        Parameters
        ----------
        signal_ssd: numpy.ndarray, shape=(M,)
            SSD of length energy-loss (M)
        n_zlp: float
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
        delta: float
            Factor added to aid stability for calculating surface losses

        Returns
        -------
        eps: numpy.ndarray
            The complex dielectric function,

                .. math::
                    \epsilon = \epsilon_1 + i*\epsilon_2,

        te: float
            local thickness
        srf_int: numpy.ndarray
            Surface losses correction

        Notes
        -----
        - Relativistic effects are not considered when correcting surface scattering.
        - The value of delta depends on the thickness of your sample and is qualitatively determined by how realistic
          the output is.

        """

        # Microscope data
        e0 = self.beam_energy * 1E3  # Electron gun voltage, converted to eV
        beta = self.collection_angle / 1E3  # Collection semi-angle, converted to rad

        # Check if data is extrapolated
        x_extrp = self.eaxis_extrp
        if len(signal_ssd) < len(x_extrp):
            y_extrp = self.extrp_signal(signal=signal_ssd)
        else:
            y_extrp = signal_ssd

        # Prepare spectrum
        x = copy.deepcopy(x_extrp[x_extrp > 0])
        y_spliced = copy.deepcopy(y_extrp[x_extrp > 0])
        y = y_spliced
        y_srf = np.empty(x.shape)
        bins = len(x)

        # Constants & kinetic definitions (Egerton 2011 appendix E)
        m_e = self.m_e  # Electron mass, eV
        a0 = self.a0  # Bohr radius, m
        hbar_c = self.h_bar * self.c  # Planck's constant * speed of light, eV*m
        gamma = 1 + (e0 / m_e)  # Relativistic factor
        T_eff = e0 * ((1 + gamma) / (2 * gamma ** 2))  # Effective kinetic energy, eV
        k0 = gamma * np.sqrt(2 * T_eff * m_e) / hbar_c  # Wavenumber, m^-1

        # select refractive or thickness loop
        if n is None and t is None:
            raise ValueError("The refractive index and thickness are not defined. Please provide ONLY one of them.")
        elif n is not None and t is not None:
            raise ValueError("Please provide the refractive index OR the thickness, not both")
        elif n is not None:
            refractive_loop = True
        elif t is not None:
            refractive_loop = False
            t = t / 1E9  # Thickness, converted to m
            if n_zlp is None:
                raise ValueError("Please provide the ZLP  when thickness is used for normalization.")

        for _ in range(iterations):
            # Calculation of the ELF by normalization of the Single Scattering Distribution
            # First perform Angular corrections (Egerton 2011 section 4.2.1)
            theta_e = x / (2 * gamma * T_eff)  # Characteristic scattering angle (per energy loss), rad
            ang_cor = np.log(1 + (beta / theta_e) ** 2)  # Angular/Aperture correction
            Im = y / ang_cor / self.deltaE  # Im[-1/eps]

            if refractive_loop:
                # normalize using the refractive index.
                Im_sum = np.sum(Im / x) * self.deltaE
                K = Im_sum / (np.pi / 2) / (1 - 1. / n ** 2)  # proportionality constant (Egerton 2011 section 4.2.2)
                if n_zlp is not None:
                    te = (2 * np.pi * a0 * K * T_eff) / n_zlp   # Calculate thickness
            else:
                # normalize using the thickness
                K = t * n_zlp / (2 * np.pi * a0 * T_eff)    # proportionality constant from thickness
                te = t
            Im = Im / K

            # Kramers Kronig Transform:
            #  We calculate KKT(Im(-1/epsilon))=1+Re(1/epsilon) with FFT
            #  Follows: D W Johnson 1975 J. Phys. A: Math. Gen. 8 490
            #  Use an optimal FFT size to speed up the calculation, and make it double the closest upper value to
            #  work around the wrap-around problem.
            optimal_fft_bins = next_fast_len(2 * bins)
            q = -2 * fft(Im, optimal_fft_bins).imag / optimal_fft_bins
            q[:bins] *= -1
            q = fft(q)
            Re = q[:bins].real + 1  # Final touch, we have Re[1/eps]

            # Epsilon appears:
            #  We calculate the real and imaginary parts of the CDF
            eps_1 = Re / (Re ** 2 + Im ** 2)
            eps_2 = Im / (Re ** 2 + Im ** 2)
            if iterations > 1 and n_zlp is not None:
                # See Egerton 2011 section 4.2.4, Ritchie (1957) and Heather (1967)
                # Calculates the surface ELF from a vacuum border effect and subtracts from the ELF.
                # delta = 0.1 for thick samples (>100nm).
                theta_delta_e = delta / (2 * gamma * T_eff)

                Im_srf = 4 * eps_2 / ((eps_1 + 1) ** 2 + eps_2 ** 2) - Im
                dPsdE = (np.arctan(beta / theta_e) / (theta_e + theta_delta_e)
                         - beta / (beta ** 2 + theta_e ** 2)) / (np.pi * a0 * k0 * T_eff)
                y_srf = (n_zlp * dPsdE * Im_srf) * self.deltaE

                y = y_spliced - y_srf
        eps = (eps_1 + eps_2 * 1j)
        return eps[x <= self.eaxis[-1]], te*1E9, y_srf[x <= self.eaxis[-1]]

    def KK_pixel(self, i, j, signal_type='EELS', iterations=1, mat_prop='n', **kwargs):
        r"""
        Perform a Kramer-Krönig analysis on pixel (``i``, ``j``).

        Parameters
        ----------
        i : int
            y-coordinate of the pixel
        j : int
            x-coordinate of the pixel.
        signal_type: str, optional
            Type of spectrum. Set to 'EELS' by default.
        iterations: int
            Number of the iterations for the internal loop to remove the
            surface plasmon contribution. If 1 the surface plasmon contribution
            is not estimated and subtracted (the default is 1).
        mat_prop: str, optional
            Material property to be used for calculations, 'n' refers to refractive index, 'rho' refers to mass density.
            Set to 'n' by default

        Returns
        -------
        dielectric_functions : numpy.ndarray, shape=(M,)
            Collection dielectric-functions replicas at pixel (``i``, ``j``).
        ts : float
            Thickness.
        S_ss : array_like
            Surface scatterings.
        signal_ssds : array_like
            Deconvoluted EELS spectrum.

        """

        if mat_prop == 'n':
            n = self.n[self.cluster_labels[i, j]]
        elif mat_prop == 'rho':
            rho = self.rho[self.cluster_labels[i, j]]
        else:
            raise ValueError("Please select either the refractive index by setting mat_prop = 'n' OR "
                             "the mass density by setting mat_prop = 'rho'")

        # Get your signal ready
        signal = self.get_pixel_signal(i, j, signal_type=signal_type, **kwargs)
        zlps = self.get_pixel_matched_zlp_models(i, j, signal_type=signal_type, signal=signal)
        signal_extrp = self.extrp_signal(signal=signal)

        # Prepare arrays
        epss = (1 + 1j) * np.zeros(zlps[:, self.eaxis > 0].shape)
        S_ss = np.zeros(zlps[:, self.eaxis > 0].shape)
        ts = np.zeros(zlps.shape[0])
        signal_ssds = np.zeros(zlps.shape)
        max_signal_ssds = np.zeros(zlps.shape[0])

        # Loop through all models
        r = None
        for k in range(zlps.shape[0]):
            zlp_k = zlps[k, :]
            n_zlp = float(np.sum(zlp_k))
            signal_ssds[k] = self.deconvolution(signal=signal_extrp, zlp=zlp_k)
            max_signal_ssds[k] = self.eaxis[np.argmax(signal_ssds[k])]
            if r is None:
                r = self.get_extrp_param(signal=signal_ssds[k])
                signal_ssd_extrp = self.extrp_signal(signal=signal_ssds[k], r=r)
            else:
                signal_ssd_extrp = self.extrp_signal(signal=signal_ssds[k], r=r)
            if mat_prop == 'rho':
                ts[k] = self.calc_thickness(signal=signal_extrp, rho=rho, n_zlp=n_zlp)
                epss[k, :], ts[k], S_ss[k] = self.kramers_kronig_analysis(signal_ssd=signal_ssd_extrp, n_zlp=n_zlp,
                                                                          t=ts[k], iterations=iterations, **kwargs)
            if mat_prop == 'n':
                epss[k, :], ts[k], S_ss[k] = self.kramers_kronig_analysis(signal_ssd=signal_ssd_extrp, n_zlp=n_zlp,
                                                                          n=n, iterations=iterations, **kwargs)
        return epss, ts, S_ss, signal_ssds, max_signal_ssds

    # METHODS ON CLUSTERING
    def cluster(self, n_clusters=3, based_on='log_zlp', init='k-means++', n_times=10, max_iter=300, seed=None,
                save_seed=False, algorithm='lloyd', **kwargs):
        r"""
        Clusters the spectral image into clusters according to the (log)
        integrated intensity at each pixel. Cluster means are stored in the
        attribute ``self.cluster_centroids``. This is then passed on to the
        cluster_on_centroids function where the index to which each cluster belongs
        is stored in the attribute ``self.cluster_labels``.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters, 5 by default
        based_on : str, optional
            One can cluster either on the sum of the intensities (pass ``'sum'``),
            the log of the sum (pass ``'log_sum'``),
            the log of the ZLP peak value (pass ``'log_peak'``),
            the log of the ZLP peak value + the two bins next to the peak value (pass ``'log_zlp'``),
            the log of the sum of the bulk spectrum (no zlp) (pass ``'log_bulk'``),
            the thickness (pass ``'thickness'``).
            The default is ``'log_zlp'``.
        init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        n_times : int, default=10
            Number of time the k-means algorithm will be run with different
            centroid seeds. The final results will be the best output of
            n_init consecutive runs in terms of inertia.
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm for a
            single run.
        seed : int or None, default=None
            Determines random number generation for centroid initialization. Use
            an int to make the randomness deterministic.
        save_seed : bool, default=False
            save the seed with corresponding settings to get to the same result
        algorithm : {'lloyd', 'elkan'}, default='lloyd'
            K-means algorithm to use. The classical EM-style algorithm is ``'lloyd'``.
            The ``'elkan'`` variation can be more efficient on some datasets with
            well-defined clusters, by using the triangle inequality. However, it's
            more memory intensive due to the allocation of an extra array of shape
            `(n_samples, n_clusters)`.
        kwargs

        Returns
        -------

        """

        if self.n_spectra == 1:
            print("Not much to cluster on with a single spectrum")
            n_clusters = 1
            seed = 12345678
            max_iter = 1

        image_data = np.copy(self.get_image_signals(**kwargs))

        if based_on == 'sum':
            intensities = image_data.sum(axis=2)
        elif based_on == 'log_sum':
            intensities = np.log(np.maximum(image_data, 1e-14).sum(axis=2))
        elif based_on == 'log_peak':
            intensities = np.log(image_data[:, :, np.argwhere((self.eaxis > -1) & (self.eaxis < 1)).flatten()].max(axis=2))
        elif based_on == 'log_zlp':
            intensities = np.zeros([self.shape[0], self.shape[1]])
            max_idx = np.argmax(image_data, axis=2)
            for i in np.arange(self.shape[0]):
                for j in np.arange(self.shape[1]):
                    intensities[i, j] = np.log(image_data[i, j, max_idx[i, j] - 1:max_idx[i, j] + 2].sum())
        elif based_on == 'log_bulk':
            intensities = np.log(np.maximum(image_data[:, :, np.argwhere(self.eaxis > 5).flatten()], 1e-14).sum(axis=2))
        elif based_on == 'thickness':
            intensities = self.t[:, :, 0]
        elif type(based_on) == np.ndarray:
            intensities = based_on
            if intensities.size != self.n_spectra:
                raise IndexError("The size of values on which to cluster does not match the image size.")
        else:
            intensities = np.sum(image_data, axis=2)
            print("provide either sum, log or thickness as clustering base, reverting back to sum")

        if seed is not None:
            seed_init = seed
            n_times = 1

        cost_min = np.inf
        X = intensities.reshape(-1, 1)
        for _ in range(n_times):
            if seed is None:
                seed_init = get_seed()
            kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter,
                            random_state=seed_init, algorithm=algorithm).fit(X)
            if cost_min > kmeans.inertia_:
                cost_min = kmeans.inertia_
                min_cluster_centroids = kmeans.cluster_centers_.flatten()
                min_seed = seed_init
                min_iter = kmeans.n_iter_
            print("Seed: " + str(seed_init) + " finished after " + str(
                kmeans.n_iter_) + " iterations and has cost: " + str(kmeans.inertia_))
        print("Seed: " + str(min_seed) + " has the lowest cost")
        self.cluster_centroids = np.sort(min_cluster_centroids)[::-1]
        print("cluster centroids are", self.cluster_centroids)
        self.cluster_on_centroids(self.cluster_centroids, based_on=based_on)

        if save_seed is True:
            print("Saving seed.txt parameters, sit tight!")
            print("seed = " + str(seed) + " clusters = " + str(n_clusters) +
                  " iterations = " + str(self.k_means_cluster.n_iter))
            path_seed = os.path.join(self.output_path, 'seed.txt')
            np.savetxt(path_seed, np.vstack((seed, n_clusters, min_iter)))
            print("Saved seed.txt!")

    def cluster_on_centroids(self, cluster_centroids, based_on='log_zlp', **kwargs):
        r"""
        If the image has been clustered before and the cluster centroids are
        already known, one can use this function to reconstruct the original
        clustering of the image.

        Parameters
        ----------
        cluster_centroids : numpy.ndarray, shape=(M,)
            Array with the cluster centroids
        based_on : str, optional
            One can cluster either on the sum of the intensities (pass ``'sum'``),
            the log of the sum (pass ``'log_sum'``),
            the log of the ZLP peak value (pass ``'log_peak'``),
            the log of the ZLP peak value + the two bins next to the peak value (pass ``'log_zlp'``),
            the log of the sum of the bulk spectrum (no zlp) (pass ``'log_bulk'``),
            the thickness (pass ``'thickness'``).
            The default is ``'log_zlp'``.
        kwargs

        """

        self.cluster_centroids = cluster_centroids
        image_data = self.get_image_signals(**kwargs)

        if based_on == 'sum':
            values = image_data.sum(axis=2)
        elif based_on == 'log_sum':
            values = np.log(np.maximum(image_data, 1e-14).sum(axis=2))
        elif based_on == 'log_peak':
            values = np.log(image_data[:, :, np.argwhere((self.eaxis > -1) & (self.eaxis < 1)).flatten()].max(axis=2))
        elif based_on == 'log_zlp':
            values = np.zeros([self.shape[0], self.shape[1]])
            max_idx = np.argmax(image_data, axis=2)
            for i in np.arange(self.shape[0]):
                for j in np.arange(self.shape[1]):
                    values[i, j] = np.log(image_data[i, j, max_idx[i, j] - 1:max_idx[i, j] + 2].sum())
        elif based_on == 'log_bulk':
            values = np.log(np.maximum(image_data[:, :, np.argwhere(self.eaxis > 5).flatten()], 1e-14).sum(axis=2))
        elif based_on == 'thickness':
            values = self.t[:, :, 0]
        elif type(based_on) == np.ndarray:
            values = based_on
            if values.size != self.n_spectra:
                raise IndexError("The size of values on which to cluster does not match the image size.")
        else:
            values = np.sum(image_data, axis=2)
            print("provide either sum, log or thickness as value base, reverting back to sum")

        valar = (values.transpose() * np.ones(np.append(self.shape[:2], self.n_clusters)).transpose()).transpose()
        self.cluster_labels = np.argmin(np.absolute(valar - cluster_centroids), axis=2)
        print("# of spectra per cluster is", np.bincount(self.cluster_labels.flatten()))
        if len(np.unique(self.cluster_labels)) < self.n_clusters:
            warnings.warn(
                "it seems like the clustered values of dE1 are not clustered on \
                    this image/on log or sum. Please check clustering.")

    def find_optimal_amount_of_clusters(self, n_clusters=3, bins=100, based_on='log_zlp', **kwargs):
        r"""
        Find the optimal amount of clusters by performing a Gaussian Mixture model on the specific data intensities.
        The user will need to judge what is best

        Parameters
        ----------
        n_clusters
        bins
        based_on
        kwargs

        Returns
        -------
        fig
        """

        image_data = self.get_image_signals(**kwargs)

        if based_on == 'sum':
            intensities = image_data.sum(axis=2)
        elif based_on == 'log_sum':
            intensities = np.log(np.maximum(image_data, 1e-14).sum(axis=2))
        elif based_on == 'log_peak':
            intensities = np.log(
                image_data[:, :, np.argwhere((self.eaxis > -1) & (self.eaxis < 1)).flatten()].max(axis=2))
        elif based_on == 'log_zlp':
            intensities = np.zeros([self.shape[0], self.shape[1]])
            max_idx = np.argmax(image_data, axis=2)
            for i in np.arange(self.shape[0]):
                for j in np.arange(self.shape[1]):
                    intensities[i, j] = np.log(image_data[i, j, max_idx[i, j] - 1:max_idx[i, j] + 2].sum())
        elif based_on == 'log_bulk':
            intensities = np.log(np.maximum(image_data[:, :, np.argwhere(self.eaxis > 5).flatten()], 1e-14).sum(axis=2))
        elif based_on == 'thickness':
            intensities = self.t[:, :, 0]
        elif type(based_on) == np.ndarray:
            intensities = based_on
            if intensities.size != self.n_spectra:
                raise IndexError("The size of values on which to cluster does not match the image size.")
        else:
            intensities = np.sum(image_data, axis=2)
            print("provide either sum, log or thickness as clustering base, reverting back to sum")

        X = intensities.reshape(-1, 1)
        gm = GaussianMixture(n_components=n_clusters).fit(X)

        # Evaluate GMM
        gm_x = np.linspace(np.min(intensities), np.max(intensities), 256)
        gm_y = np.exp(gm.score_samples(gm_x.reshape(-1, 1)))

        fig, ax = plt.subplots()
        ax.hist(X.flatten(), density=True, bins=bins, color='tab:blue')
        ax.plot(gm_x, gm_y, color='tab:red', lw=2, label="GMM")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Pixel Intensity")
        plt.legend(frameon=False)

        return fig

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

    def __getitem__(self, key):
        r"""
        Determines behavior of `self[key]`
        """

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
        return "Spectral image: " + name_str + ", image size:" + str(self.data.shape[0]) + "x" + \
            str(self.data.shape[1]) + ", eaxis range: [" + str(round(self.eaxis[0], 3)) + "," + \
            str(round(self.eaxis[-1], 3)) + \
            "], deltaE: " + str(round(self.deltaE, 3))

    def __repr__(self):
        data_str = "data * np.ones(" + str(self.shape) + ")"
        if self.name is not None:
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return "Spectral_image(" + data_str + ", deltaE=" + str(round(self.deltaE, 3)) + name_str + ")"

    def __len__(self):
        return self.shape[2]


# STATIC FUNCTIONS
def cft(x, y):
    r"""
    Fourier Transformation

    Parameters
    ----------
    x: numpy.ndarray, shape=(M,)
    y: numpy.ndarray, shape=(M,)

    Returns
    -------
    F_k: numpy.ndarray, shape=(M,)

    """

    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    delta_x = (x_max - x_min) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(2j * np.pi * N_0 * k / N)
    F_k = fft(y) * cont_factor * delta_x
    return F_k


def icft(x, Y_k):
    r"""
    Inverse Fourier Transformation

    Parameters
    ----------
    x: numpy.ndarray, shape=(M,)
    Y_k: numpy.ndarray, shape=(M,)

    Returns
    -------
    f_n: numpy.ndarray, shape=(M,)

    """

    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    delta_x = (x_max - x_min) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(-2j * np.pi * N_0 * k / N)
    f_n = ifft(Y_k * cont_factor) / delta_x
    return f_n


def scale(inp, ab):
    r"""
    Rescale the input features by applying a linear map such that the range
    covers 0.1 to 0.9.

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
    r"""
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
    r"""
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
    r"""
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

    return np.trunc(values * 10 ** decs) / (10 ** decs)


def get_seed(n_min=1e7, n_max=1e8):
    return np.random.randint(n_min, n_max)


def linear_fit(x, a, b):
    return a * x + b


def power_fit(x, a, r):
    return a * x ** r


def gauss1d(x=0, mx=0, sx=2, **kwargs):
    r"""

    Parameters
    ----------
    x
    mx
    sx

    Returns
    -------

    """

    return 1. / (np.sqrt(2 * np.pi) * sx) * np.exp(
        -(x - mx) ** 2 / (2 * sx ** 2))


def gauss2d(x=0, y=0, mx=0, my=0, sx=2, sy=2, **kwargs):
    r"""
    Returns values for a 2d gaussian distribution

    Parameters
    ----------
    x
    y
    mx
    my
    sx
    sy

    Returns
    -------

    """

    return 1. / (2. * np.pi * sx * sy) * np.exp(
        -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))
