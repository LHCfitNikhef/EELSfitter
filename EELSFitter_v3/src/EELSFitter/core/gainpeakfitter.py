# @AUTHOR:        H.LA, S.VANDERLIPPE
# @STATUS:        PRODUCTION
# @DATE_CREATED:  25-09-2022
# @DESCRIPTION:   This script generates intensity heatmaps of gain peaks to
#                 investigate its spatial distribution. The GainPeakFitter
#                 fits the ZLP to a Gaussian based on the ZLP's FWHM. The ZLP is
#                 then subtractred from the EEL spectrum (also referred to as
#                 'the signal'). The subtracted signal contains a peak which
#                 is then fitted to a Lorentzian. From the Lorentzian, the energy
#                 of the gain peak is determined.


from re import X
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import scipy.special

plt.rcParams.update({'font.size': 16})  # pyplot style


class GainPeakFitter:
    """This class extracts gain peaks from an EEL spectrum. The GainPeakFitter
    fits the ZLP to a Gaussian based on the ZLP's FWHM. The ZLP is then 
    subtractred from the EEL spectrum (also referred to as 'the signal'). The 
    subtracted signal contains a peak which is then fitted to a Lorentzian. 
    From the Lorentzian, the energy of the gain peak is determined.

    INPUT
    -----
    image : spectral image
        4D data from a .dm4 file

    EXAMPLE
    -------

    To fit the gain/loss peaks:

        lrtz = GainPeakFitter(x_signal, y_signal, image_shape)

        lrtz.generate_best_fits()

        lrtz.fit_gain_peak()

        lrtz.fit_loss_peak()

    To plot the gain/loss peaks:

        lrtz.create_new_plot()

        lrtz.plot_all_results()

        lrtz.ax.set_ylim(0, 1e4)

        lrtz.fig

        lrtz.plot_signal()

        lrtz.plot_model()

        lrtz.plot_subtracted()

        lrtz.plot_gain_peak()

        lrtz.plot_loss_peak()

        lrtz.print_results()
    """

    def __init__(self, x_signal, y_signal, image_shape, image=None):
        """Load processed signal.
        
        ARGUMENTS
        ---------
        x_A : float, optional)
            Gain peak left bound. Defaults to -3.2.
        x_B : float, optional
            Gain peak right bound. Defaults to -0.4.
        x_C : float, optional
            Loss peak left bound. Defaults to 0.4.
        x_D : float, optional
            Loss peak right bound. Defaults to 3.2.
        """
        self.x_signal = x_signal
        self.y_signal = y_signal
        self.n_rows = image_shape[0]
        self.n_cols = image_shape[1]
        self.x_A = -2.5
        self.x_B = -0.4
        self.x_C = 0.4
        self.x_D = 2.5
        self.xm = np.arange(-4, 4, 1e-3)  # energy loss (eV)
        self.image = image
        self.zlp_models_all = None
        self.y_signals_subtracted = None

    def generate_best_fits(self, function='Gaussian', **kwargs):
        self.get_signal_subtracted(function, **kwargs)

        # get x,y coordinates for model fit of the ZLP
        self.generate_model(function, x=self.x_signal, **kwargs)

    def get_signal_subtracted(self, function='Gaussian', **kwargs):
        # best fit values from model fit of the ZLP
        self.popt_zlp, self.pcov_zlp = self.model_fit_between(
                                                        function=function,
                                                        **kwargs)
        self.signal_subtracted(function=function, **kwargs)

    def set_area_under_zlp(self, function='Gaussian', **kwargs):
        self.ym_model = self.model(self.xm, function, **kwargs)
        self.total_count_zlp = self.ym_model.sum()

    def ratio_between_gain_peak_and_zlp_fwhm(self):
        i1, i2 = self.fwhm_idx(y=self.ym_gain)
        return self.ym_gain[i1:i2].sum() / self.ym_model[i1:i2].sum()

    def ratio_between_loss_peak_and_zlp_fwhm(self):
        i1, i2 = self.fwhm_idx(y=self.ym_loss)
        return self.ym_loss[i1:i2].sum() / self.ym_model[i1:i2].sum()

    def ratio_between_gain_peak_and_zlp(self):
        return self.total_count_gain_peak / self.total_count_zlp

    def ratio_between_loss_peak_and_zlp(self):
        return self.total_count_loss_peak / self.total_count_zlp

    def fit_gain_peak(self, L_bound=-3.5, R_bound=-.5):
        try:
            self.popt_gain, self.pcov_gain = self.curve_fit_between(
                L_bound, R_bound)  # mirror of loss peak
            self.ym_gain = self.lorentzian(self.xm, *self.popt_gain)
            self.set_total_count_gain_peak()
        except Exception as e:
            print("==> Gain peak fit FAILED!!", e)

    def fit_loss_peak(self, L_bound=.5, R_bound=1.5):
        try:
            self.popt_loss, self.pcov_loss = self.curve_fit_between(
                L_bound, R_bound)  # mirror of gain peak
            self.ym_loss = self.lorentzian(self.xm, *self.popt_loss)
            self.set_total_count_loss_peak()
        except Exception as e:
            print("==> Loss peak fit FAILED!!", e)

    def fit_loss_peak_background(self, L_bound=.5, R_bound=3.5):
        try:
            self.popt_loss, self.pcov_loss = self.curve_fit_between_background(
                L_bound, R_bound)  # mirror of loss peak
            self.ym_loss = self.lorentzian(self.xm, *self.popt_loss[:3])
            self.background_signal = self.background(self.x_signal, *self.popt_loss[3:])
            self.total_fit = self.lorentzian_background(self.x_signal, *self.popt_loss)
            self.set_total_count_loss_peak()
        except Exception as e:
            print("==> Gain peak fit FAILED!!", e)

    def inspect_spectrum(self, row=0, col=0, function='Gaussian', method='fit', **kwargs):
        """Fit chosen ``function`` model to the ZLP and obtain subtracted spectrum.

        Parameters
        ----------
        function : str, {'Gaussian', 'Split Gausian', 'Lorentzian', 'Pearson VII', 'Split Pearson VII', 'Pseudo-Voigt', 'Split Pseudo-Voigt', 'Generalised Peak', 'Kaiser window'}
            Model choice for the ZLP.
        method : str, {``'fit'``, ``'FWHM'``}
            Method to use to extract model fit parameters. ``'FWHM'`` is only supported for Gaussian and
            Lorentzian ZLP models.
        kwargs: dict, optional
            Additional keyword arguments.        
        """
        self.y_signal = self.image.get_pixel_signal(i=row, j=col, **kwargs)

        print(f"Pixel {(row, col)}.")
        self.row = row  # pixel row
        self.col = col  # pixel column

        if method == 'FWHM':
            self.popt_zlp = self.determine_parameters(function)
        elif method == 'fit':
            self.popt_zlp, self.pcov_zlp = self.model_fit_between(
                function, **kwargs)

        self.signal_subtracted(function, **kwargs)
        self.generate_model(function, **kwargs)

    def fwhm(self, row=0, col=0, do_fit=True, **kwargs):
        """Calculates the FWHM of the ZLP in a particular pixel (``row``, ``col`` ). Optionally
        calculate FWHM through fitting a Gaussian to the ZLP and extracting the FWHM.

        Parameters
        ----------
        do_fit : bool, default=True
            Option to obtain FWHM through fitting a Gaussian.

        Returns
        -------
        fwhm : float
            FWHM of the ZLP in chosen pixel.
        fit : float
            FWHM obtained through fitting a Gaussian to the ZLP.
        """
        if self.y_signal is None:
            self.y_signal = self.image.get_pixel_signal(i=row, j=col, **kwargs)
        height = np.max(self.y_signal)
        # Find locations where data is larger than half the peak height.
        # The left edge of the FWMM is the first index, the right edge the last index.
        # We subtract and add 1 respectively to get a better estimate of the FWHM.
        fwhm_idx1 = np.argwhere(self.y_signal >= height/2)[0] - 1
        fwhm_idx2 = np.argwhere(self.y_signal >= height/2)[-1] + 1
        fwhm = self.x_signal[fwhm_idx2] - self.x_signal[fwhm_idx1]
        if do_fit:
            self.popt_zlp, self.pcov_zlp = self.model_fit_between()
            fit = abs(self.popt_zlp[1]) * (2 * np.sqrt(2 * np.log(2)))
        else:
            fit = None
        return fwhm, fit

    def fit_models(self, n_rep=500, n_clusters=5, function='Gaussian', conf_interval=1, signal_type='EELS', **kwargs):
        """Use the Monte Carlo replica method to fit chosen ``function`` model to the ZLP. In this method
        it is assumed that in each cluster the ZLP is sampled from the same underlying distribution in that 
        particular cluster. This methods samples the underlying distribution in order to obtain the median, 
        low, and high predictions for the ZLP at teach loss value.

        The model predictions are stored in self.zlp_models_all, where the median, low, and high values are the first, 
        second, and third element respectively.

        Parameters
        ----------
        n_rep : int, default=500
            Number of Monte Carlo replicas to use.
        n_clusters : int, default=5
            Number of clusters to use.
        function : str, {'Gaussian', 'Split Gausian', 'Lorentzian', 'Pearson VII', 'Split Pearson VII', 'Pseudo-Voigt', 'Split Pseudo-Voigt', 'Generalised Peak', 'Kaiser window'}
            Model choice for the ZLP.
        conf_interval : float, optional
            The ratio of spectra returned. The spectra are selected based on the 
            based_on value. The default is 1.
        signal_type: str, optional
            Description of signal, ``'EELS'`` by default.
        kwargs: dict, optional
            Additional keyword arguments. 
        """
        self.n_rep = n_rep
        if self.image.cluster_labels is None:
            print('Spectral image has not been clustered yet. Clustering with default parameters.')
            self.image.cluster(n_clusters=n_clusters, **kwargs)
        # If the number of unique values is not equal to the number of clusters,
        # then the clustering used a different number of values and needs to be redone.
        elif len(np.unique(self.image.cluster_labels)) != n_clusters:
            self.image.cluster(n_clusters=n_clusters, **kwargs)
        spectra = self.image.get_cluster_signals(conf_interval=conf_interval, signal_type=signal_type)

        rng = np.random.default_rng()

        # Each of the clusters will have n_rep models for the ZLP, each fitted on a different spectrum.
        self.zlp_models_all = np.zeros((n_clusters, n_rep, len(self.x_signal)))
        for i in range(n_rep):
            for cluster in range(n_clusters):
                # Number of spectra in the cluster
                n_spectra = len(spectra[cluster])
                # Select a random spectrum in the cluster
                idx = rng.integers(n_spectra)
                self.y_signal = spectra[cluster][idx]

                # Best fit values from model fit of the ZLP
                self.popt_zlp, self.pcov_zlp = self.model_fit_between(
                    function=function, **kwargs)

                # Get x, y coordinates for model fit of the ZLP
                self.generate_model(function=function, x=self.image.eaxis, **kwargs)
                self.zlp_models_all[cluster, i] = np.copy(self.ym_model)
        
        # # Array that holds the median, low and high of each cluster ZLP prediction for each energy loss
        # self.zlp_models = np.zeros((n_clusters, 3, len(self.x_signal)))
        # for cluster in range(n_clusters):
        #     self.zlp_models[cluster] = summary_distribution(self.zlp_models_all[cluster])

    def fit_gain_peak_mc(self, i, j, L_bound=-3.5, R_bound=-.5, return_all=False, return_conf_interval=False, **kwargs):
        """Use the Monte Carlo replica method to fit a Lorentzian model to the subtracted spectrum
        in a specified energy interval [``L_bound, R_bound``].

        Parameters
        ----------
        i : int
            y-coordinate of the pixel.
        j : int
            x-coordinate of the pixel.
        L_bound : float, optional
            Left bound of the interval in which to fit to the subtracted spectrum.
        R_bound : float, optional
            Right bound of the interval in which to fit to the subtracted spectrum.
        return_all : bool, optional
            Option to return the subtracted spectra for all replicas corresponding
            to this pixel.
        return_conf_interval : bool, optional
            Option to specify if the upper and lower bounds of the confidence 
            interval must be returned.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        gain : numpy.ndarray
            Array with the median Lorentzian fit to the subtracted spectrum.
        gain_low : numpy.ndarray, optional
            Lower bound of the Lorentzian fit.
        gain_high : numpy.ndarray, optional
            Upper bound of the Lorentzian fit.
        """
        y_signal_subtracted = self.get_subtracted_spectrum(i, j, return_all=True, **kwargs)
        y_gain_all = np.zeros((self.n_rep, len(self.image.eaxis)))
        for i in range(self.n_rep):
            try:
                self.y_signal_subtracted = y_signal_subtracted[i]
                self.popt_gain, self.pcov_gain = self.curve_fit_between(
                    L_bound, R_bound)  # mirror of loss peak
                self.y_gain = self.lorentzian(self.image.eaxis, *self.popt_gain)
                y_gain_all[i] = np.copy(self.y_gain)
            except Exception as e:
                print("==> Gain peak fit FAILED!!", e)
                y_gain_all[i] = np.nan
        if return_all:
            return y_gain_all
        if return_conf_interval:
            return summary_distribution(y_gain_all)
        else:
            return summary_distribution(y_gain_all)[0]

    def get_subtracted_spectrum(self, i, j, signal_type='EELS', return_all=False, return_conf_interval=False, **kwargs):
        """Retrieves the subtracted spectrum at pixel (``i``, ``j``).

        Parameters
        ----------
        i : int
            y-coordinate of the pixel.
        j : int
            x-coordinate of the pixel.
        signal_type: str, optional
            The type of signal that is requested, should comply with the defined
            names. Set to ``'EELS'`` by default.
        return_all : bool, optional
            Option to return the subtracted spectra for all replicas corresponding
            to this pixel.
        return_conf_interval : bool, optional
            Option to specify if the upper and lower bounds of the confidence 
            interval must be returned.
        kwargs: dict, optional
            Additional keyword arguments. 

        Returns
        -------
        signal : numpy.ndarray
            Array with the median subtracted spectrum from the requested pixel.
        signal_low : numpy.ndarray, optional
            Lower bound of the confidence interval of the subtracted spectrum.
        signal_high : numpy.ndarray, optional
            Upper bound of the confidence interval of the subtracted spectrum.
        """
        if self.image.cluster_labels is None:
            print('Spectral image has not been clustered yet. Clustering with default parameters.')
            self.image.cluster(**kwargs)
        if self.zlp_models_all is None:
            print('ZLP models have not been fitted yet. Fitting with default parameters.')
            self.fit_models(signal_type=signal_type, **kwargs)
        signal = self.image.get_pixel_signal(i=i, j=j, signal_type=signal_type, **kwargs)
        cluster = self.image.cluster_labels[i,j]
        y_signal_subtracted = signal - self.zlp_models_all[cluster]
        # Intensity is a positive definite quantity
        y_signal_subtracted[y_signal_subtracted < 0] = 0
        if return_all:
            return y_signal_subtracted
        if return_conf_interval:
            return summary_distribution(y_signal_subtracted)
        else:
            return summary_distribution(y_signal_subtracted)[0]
    
    def get_model(self, i, j, signal_type='EELS', return_all=False, return_conf_interval=False, **kwargs):
        """Retrieves the model fit to the ZLP at pixel (``i``, ``j``).

        Parameters
        ----------
        i : int
            y-coordinate of the pixel.
        j : int
            x-coordinate of the pixel.
        signal_type: str, optional
            The type of signal that is requested, should comply with the defined
            names. Set to ``'EELS'`` by default.
        return_all : bool, optional
            Option to return all models.
        return_conf_interval : bool, optional
            Option to specify if the upper and lower bounds of the confidence 
            interval must be returned.
        kwargs: dict, optional
            Additional keyword arguments. 

        Returns
        -------
        model : numpy.ndarray
            Array with the median model fit to the ZLP from the requested pixel.
        model_low : numpy.ndarray, optional
            Lower bound of the confidence interval of the model fit to the ZLP.
        model_high : numpy.ndarray, optional
            Upper bound of the confidence interval of the model fit to the ZLP.
        """
        if self.image.cluster_labels is None:
            print('Spectral image has not been clustered yet. Clustering with default parameters.')
            self.image.cluster(**kwargs)
        cluster = self.image.cluster_labels[i, j]
        if self.zlp_models_all is None:
            print('ZLP models have not been fitted yet. Fitting with default parameters.')
            self.fit_models(signal_type=signal_type, **kwargs)
        if return_all:
            return self.zlp_models_all[cluster]
        if return_conf_interval:
            return summary_distribution(self.zlp_models_all[cluster])
        else:
            return summary_distribution(self.zlp_models_all[cluster])[0]
    
    def plot_single_spectrum(self, ymin=0, ymax=1e5, xmin=-4, xmax=4):
        fig, ax1 = plt.subplots()
        ax1.plot(self.x_signal, self.y_signal, 'r')
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlim(xmin, xmax)

    def perr_gain(self):
        return np.sqrt(np.diag(self.popt_gain))

    def gaussian(self, x, a, sigma, x0=0):
        """Gaussian centered around ``x`` = ``x0``.

        Parameters
        ----------
        x : numpy.ndrray
            1D array of the energy loss.
        x0 : float
            Energy loss at the center of the Gaussian.
        a : float
            Height of the Gaussian peak.
        sigma : float
            Standard deviation of the Gaussian.

        Returns
        -------
        numpy.ndarray
            Gaussian.
        """
        return a * np.exp(-(x-x0)**2 / (2*sigma**2))

    def split_gaussian(self, x, a, sigma_left, sigma_right, x0=0):
        """Gaussian centered around ``x`` = ``x0`` with a different standard deviation for ``x`` < ``x0``
        and ``x`` > ``x0``.

        Parameters
        ----------
        x : numpy.ndrray
            1D array of the energy loss.
        x0 : float
            Energy loss at the center of the Gaussian.
        a : float
            Height of the Gaussian peak.
        sigma_left : float
            Standard deviation of the left half of the Gaussian.
        sigma_right : float
            Standard deviation of the right half of the Gaussian.

        Returns
        -------
        numpy.ndarray
            Split Gaussian.
        """
        return np.where(x < x0, a * np.exp(-(x-x0)**2 / (2*sigma_left**2)), a * np.exp(-(x-x0)**2 / (2*sigma_right**2)))

    def lorentzian(self, x, x0, a, gam):
        """Lorentzian centered around ``x`` = ``x0``.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        x0 : float
            Energy loss at the center of the Lorentzian.
        a : float
            Height of the Lorentzian peak.
        gam : float
            2 * ``gamma`` is full width at half maximum (FWHM).

        Returns
        -------
        numpy.ndarray
            Lorentzian.
        """
        return a * gam**2 / (gam**2 + (x - x0)**2)  # Lorentzian
    
    # def lorentzian_background(self, x, x0, a, gam, b):
    def lorentzian_background(self, x, x0, a, gam, E_0, b):
    # def lorentzian_background(self, x, x0, a, gam, E_0, b, c):
    # def lorentzian_background(self, x, x0, a, gam, E_0, a_0, b, c):
        """Lorentzian centered around ``x`` = ``x0``.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        x0 : float
            Energy loss at the center of the Lorentzian.
        a : float
            Height of the Lorentzian peak.
        gam : float
            2 * ``gamma`` is full width at half maximum (FWHM).

        Returns
        -------
        numpy.ndarray
            Lorentzian.
        """
        # return a * gam**2 / (gam**2 + (x - x0)**2) + self.background(x, b)
        return a * gam**2 / (gam**2 + (x - x0)**2) + self.background(x, E_0, b)
        # return a * gam**2 / (gam**2 + (x - x0)**2) + self.background(x, E_0, b, c)
        # return a * gam**2 / (gam**2 + (x - x0)**2) + self.background(x, E_0, a_0, b, c)
    
    # def background(self, x, b):
    def background(self, x, E_0, b):
    # def background(self, x, E_0, b, c):
    # def background(self, x, E_0, a_0, b, c):
        # return np.where(x > E_0, b*np.sqrt(x) + c*x**(3/2), 0)
        # return np.where(x > E_0, a_0 + b*x + c*x**2, 0)
        # return np.where(x > E_0, b * x**c, 0)
        # return np.where(x > E_0, b * (x - E_0)**c, 0)
        return np.where(x > E_0, b * np.sqrt(x), 0)
        # return np.where(x > E_0, b * np.sqrt(x - E_0), 0)
        # return np.where(x > 0, b * np.sqrt(x), 0)
        # return np.where(x > E_0, b / np.sqrt(x), 0)
        # return np.where(x > E_0, b / np.sqrt(x - E_0), 0)
        # return np.where(x > E_0, b*x**(3/2), 0)
        # return np.where(x > E_0, b*(x - E_0)**(3/2), 0)
        # return np.where(x > E_0, b / (x)**(3/2), 0)

    def split_lorentzian(self, x, x0, a, gam_left, gam_right):
        """Lorentzian centered around ``x`` = ``x0`` with a different FWHM for ``x`` < ``x0``
        and ``x`` > ``x0``.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        x0 : float
            Energy loss at the center of the Lorentzian.
        a : float
            Height of the Lorentzian peak.
        gam_left : float
            2 * ``gam_left`` is full width at half maximum (FWHM) of the left half of the Lorentzian.
        gam_right : float
            2 * ``gam_right`` is full width at half maximum (FWHM) of the right half of the Lorentzian.

        Returns
        -------
        numpy.ndarray
            Split Lorentzian.
        """
        return np.where(x < x0, a * gam_left**2 / (gam_left**2 + (x - x0)**2), a * gam_right**2 / (gam_right**2 + (x - x0)**2))  # Lorentzian

    def pearson(self, x, I_max, x_0, w, m):
        r"""Pearson VII function calculated as 

        .. math::
            I(x, I_{\mathrm{max}}, x_0, w, m) = I_{\mathrm{max}} \frac{w^{2m}}{\left[w^2 + \left(2^\frac{1}{m} - 1 \right) \left(x - x_0 \right)^2 \right]^2}.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        I_max : float
            Height of the peak.
        x_0 : float
            Energy loss at the center of the peak.
        w : float
            Parameter related to the width of the peak.
        m : float
            Parameter chosen to suit a particular peak shape.

        Returns
        -------
        numpy.ndarray
            Pearson VII function.
        """
        return I_max * w ** (2*m) / (w**2 + (2**(1/m) - 1) * (x - x_0)**2)**m

    def split_pearson(self, x, I_max, x_0, w_left, w_right, m):
        """Pearson VII function with a different ``w`` for ``x`` < ``x_0`` 
        and ``x`` > ``x_0``.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        I_max : float
            Height of the peak.
        x_0 : float
            Energy loss at the center of the peak.
        w : float
            Parameter related to the width of the peak.
        m : float
            Parameter chosen to suit a particular peak shape.

        Returns
        -------
        numpy.ndarray
            Split Pearson VII function.
        """
        return np.where(x < x_0, I_max * w_left ** (2*m) / (w_left**2 + (2**(1/m) - 1) * (x - x_0)**2)**m, I_max * w_right ** (2*m) / (w_right**2 + (2**(1/m) - 1) * (x - x_0)**2)**m)

    def pseudo_voigt(self, x, I_max, x_0, f, eta):
        """Linear combination of a Gaussian and Lorentzian function, both described by
        the same FWHM ``f``.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        I_max : float
            Height of the peak.
        x_0 : float
            Energy loss at the center of the peak.
        f : float
            FWHM.
        eta : float
            Mixing parameter.

        Returns
        -------
        numpy.ndarray
            Pseudo-Voigt function.
        """
        return eta * self.gaussian(x, I_max, f / (2 * np.sqrt(2 * np.log(2))), x_0) + (1 - eta) * self.lorentzian(x, x_0, I_max, f / 2)

    def split_pseudo_voigt(self, x, I_max, x_0, f_left, f_right, eta):
        """Pseudo-Voigt function with a different FWHM for ``x`` < ``x_0`` 
        and ``x`` > ``x_0``.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        I_max : float
            Height of the peak.
        x_0 : float
            Energy loss at the center of the peak.
        f_left : float
            FWHM for the left half of the function.
        f_right : float
            FWHM for the right half of the function.
        eta : float
            Mixing parameter.

        Returns
        -------
        numpy.ndarray
            Split Pseudo-Voigt function.
        """
        return eta * self.split_gaussian(x, I_max, f_left / (2 * np.sqrt(2 * np.log(2))), f_right / (2 * np.sqrt(2 * np.log(2))), x_0) + (1 - eta) * self.split_lorentzian(x, x_0, I_max, f_left / 2, f_right / 2)

    def generalised_peak(self, x, x_0, delta, nu):
        r"""Generalised Peak function calculated as 

        .. math::
            \frac{2}{\pi \delta} \Bigg|\frac{\Gamma\left[\frac{\nu}{2} + i \gamma_\nu \left(\frac{4 x_s^2}{\pi^2 \delta^2} \right)^2 \right]}{\Gamma\left[\frac{\nu}{2}] \right]}\Bigg|^2,

        where :math:`\gamma_\nu = \sqrt{\pi} \frac{\Gamma\left[\frac{\nu + 1}{2} \right]}{\Gamma\left[\nu + \frac{1}{2} \right]}`.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        x_0 : float
            Energy offset.
        delta : float
            Parameter describing the peak width.
        nu : float
            Parameter describing the peak shape.

        Returns
        -------
        numpy.ndarray
            Generalised Peak function.
        """
        gamma_v = np.sqrt(np.pi) * scipy.special.gamma((nu +
                                                        1) / 2) / scipy.special.gamma(nu + 1/2)
        q_s = x - x_0
        return 2 / (np.pi * delta) * np.abs(scipy.special.gamma(nu/2 + 1j*gamma_v * (4 * q_s / (np.pi * delta))**4) / scipy.special.gamma(nu / 2))**2

    def kaiser(self, x, L, m):
        r"""Kaiser window function, calculated as

        .. math::
            w_0(x) \triangleq \begin{array}{cl} \frac{1}{L} \frac{I_0\left[m \sqrt{1-(2 x / L)^2}\right]}{I_0[m]},
            & |x| \leq L / 2 \\ 0, & |x|>L / 2 \end{array},

        where :math:`I_0` is the zeroth-order modified Bessel function of the first kind.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        L : float
            Window duration.
        m : float
            Parameter determining the window shape.

        Returns
        -------
        numpy.ndarray
            Kaiser window function.
        """
        return np.where(np.abs(x) <= L/2, scipy.special.iv(0, m * np.emath.sqrt(1 - (2*x / L)**2).real) / (L * scipy.special.iv(0, m).real), 0)

    def model(self, x, function, **kwargs):
        """Calculates the ZLP model using the optimal fit parameters.

        Parameters
        ----------
        x : numpy.ndarray
            1D array of energy loss.
        function : str, {'Gaussian', 'Split Gausian', 'Lorentzian', 'Pearson VII', 'Split Pearson VII', 'Pseudo-Voigt', 'Split Pseudo-Voigt', 'Generalised Peak', 'Kaiser window'}
            Model choice for the ZLP.
        kwargs: dict, optional
            Additional keyword arguments.       

        Returns
        -------
        numpy.ndarray
            ZLP model fit.
        """
        if function == 'Gaussian':
            return self.gaussian(x, *self.popt_zlp)
        elif function == 'Split Gaussian':
            return self.split_gaussian(x, *self.popt_zlp)
        elif function == 'Lorentzian':
            return self.lorentzian(x, *self.popt_zlp)
        elif function == 'Pearson VII':
            return self.pearson(x, *self.popt_zlp, **kwargs)
        elif function == 'Split Pearson VII':
            return self.split_pearson(x, *self.popt_zlp, **kwargs)
        elif function == 'Pseudo-Voigt':
            return self.pseudo_voigt(x, *self.popt_zlp)
        elif function == 'Split Pseudo-Voigt':
            return self.split_pseudo_voigt(x, *self.popt_zlp)
        elif function == 'Generalised peak':
            return self.generalised_peak(x, *self.popt_zlp)
        elif function == 'Kaiser window':
            return self.kaiser(x, *self.popt_zlp, **kwargs)

    def curve_fit_between(self, x_left=-1.6, x_right=-0.6):
        """Curve fit between chosen eloss fitting range.

        Args:
            x (np.array): eloss [in eV]
            y (np.array): signal, intensity, electron count [in a.u.]
            x_left (float): left margin of eloss fitting range
            x_right (float): right margin of eloss fitting range
        """
        idx_L, _ = find_nearest(self.x_signal, x_left)
        idx_R, _ = find_nearest(self.x_signal, x_right)
        x_range = self.x_signal[idx_L:idx_R]
        y_range = self.y_signal_subtracted[idx_L:idx_R]
        try:
            return curve_fit(self.lorentzian, x_range, y_range)
        except Exception:
            print(f"==> Lorentzian fit failed!")

    def curve_fit_between_background(self, x_left=-1.6, x_right=-0.6):
        """Curve fit between chosen eloss fitting range.

        Args:
            x (np.array): eloss [in eV]
            y (np.array): signal, intensity, electron count [in a.u.]
            x_left (float): left margin of eloss fitting range
            x_right (float): right margin of eloss fitting range
        """
        idx_L, _ = find_nearest(self.x_signal, x_left)
        idx_R, _ = find_nearest(self.x_signal, x_right)
        x_range = self.x_signal[idx_L:idx_R]
        y_range = self.y_signal_subtracted[idx_L:idx_R]
        return curve_fit(self.lorentzian_background, x_range, y_range)

    def model_fit_between(self, function='Gaussian', **kwargs):
        """Model fit of the ZLP. Delete x coordinates that contain information about gain/loss peaks.

        Parameters
        ----------
        function : str, {'Gaussian', 'Split Gausian', 'Lorentzian', 'Pearson VII', 'Split Pearson VII', 'Pseudo-Voigt', 'Split Pseudo-Voigt', 'Generalised Peak', 'Kaiser window'}
            Model choice for the ZLP.
        kwargs: dict, optional
            Additional keyword arguments.  

        Returns
        -------
        popt: numpy.ndarray
            Optimal values for the parameters so that the sum of the squared residuals of
            f(xdata, *popt) - ydata is minimized.
        pcov : numpy.ndarray
            The estimated covariance of popt. The diagonals provide the  variance of the parameter estimates.
            To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
        """

        # Define which indices in the array of the signal should be ignored
        idx_A, _ = find_nearest(self.x_signal, self.x_A)
        idx_B, _ = find_nearest(self.x_signal, self.x_B)
        idx_C, _ = find_nearest(self.x_signal, self.x_C)
        idx_D, _ = find_nearest(self.x_signal, self.x_D)
        idx_delete_from_A_to_B = np.arange(idx_A, idx_B, dtype=np.int64)
        idx_delete_from_C_to_D = np.arange(idx_C, idx_D, dtype=np.int64)
        idx_delete = np.append(idx_delete_from_A_to_B, idx_delete_from_C_to_D)

        # Delete all indices between idx_A & idx_B; keep relevant ranges.
        x_new = np.delete(self.x_signal, idx_delete)
        y_new = np.delete(self.y_signal, idx_delete)
        x_0 = 0
        try:
            if function == 'Gaussian':
                fwhm_est = self.fwhm(do_fit=False)[0][0]
                # return curve_fit(lambda x, a, sigma: self.gaussian(x, a, sigma, 0), x_new, y_new)
                return curve_fit(self.gaussian, x_new, y_new, p0=[max(self.y_signal), fwhm_est / (2 * np.sqrt(2 * np.log(2))), x_0])
            elif function == 'Split Gaussian':
                return curve_fit(self.split_gaussian, x_new, y_new)
            elif function == 'Lorentzian':
                fwhm_est = self.fwhm(do_fit=False)[0][0]
                return curve_fit(self.lorentzian, x_new, y_new, p0=[x_0, max(self.y_signal), fwhm_est])
            elif function == 'Pearson VII':
                m = kwargs['m']
                fwhm_est = self.fwhm(do_fit=False)[0][0]
                return curve_fit(lambda x, I_max, x_0, w: self.pearson(x, I_max, x_0, w, m), x_new, y_new, p0=[max(self.y_signal), x_0, fwhm_est])
                # return curve_fit(self.pearson, x_new, y_new)
            elif function == 'Split Pearson VII':
                m = kwargs['m']
                fwhm_est = self.fwhm(do_fit=False)[0][0]
                return curve_fit(lambda x, I_max, x0, w_left, w_right: self.split_pearson(x, I_max, x0, w_left, w_right, m), x_new, y_new, p0=[max(self.y_signal), x_0, fwhm_est, fwhm_est])
                # return curve_fit(self.pearson, x_new, y_new)
            elif function == 'Pseudo-Voigt':
                fwhm_est = self.fwhm(do_fit=False)[0][0]
                return curve_fit(self.pseudo_voigt, x_new, y_new, p0=[max(self.y_signal), x_0, fwhm_est, 0.5])
            elif function == 'Split Pseudo-Voigt':
                fwhm_est = self.fwhm(do_fit=False)[0][0]
                return curve_fit(self.split_pseudo_voigt, x_new, y_new, p0=[max(self.y_signal), x_0, fwhm_est, fwhm_est, 0.5])
            elif function == 'Generalised peak':
                return curve_fit(self.generalised_peak, x_new, y_new)
            elif function == 'Kaiser window':
                return curve_fit(self.kaiser, x_new, y_new, p0=[0.5, 20])
            else:
                print('Model not recognised. Use one of the possible options.')
        except Exception as e:
            print(e)
            print(f"==> Model ZLP-fit failed!")

    def determine_parameters(self, function):
        """ Determines the parameters specifying a Gaussian or Lorentzian function using
        the signal's peak height and the FWHM of the peak.

        Parameters
        ----------
        function : str, {'Gaussian', 'Lorentzian'}
            Model choice for the ZLP.
        """
        index = np.argmax(self.y_signal)
        height = self.y_signal[index]
        x0 = self.x_signal[index]

        # Find locations where data is larger than half the peak height.
        # The left edge of the FWMM is the first index, the right edge the last index.
        # We subtract and add 1 respectively to get a better estimate of the FWHM.
        fwhm_idx1 = np.argwhere(self.y_signal >= height/2)[0] - 1
        fwhm_idx2 = np.argwhere(self.y_signal >= height/2)[-1] + 1
        fwhm = self.x_signal[fwhm_idx2] - self.x_signal[fwhm_idx1]
        if function == 'Gaussian':
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            return height, sigma, x0
        elif function == 'Lorentzian':
            return x0, height, fwhm/2

    def signal_subtracted(self, function='Gaussian',**kwargs):
        """Generate signal spectrum minus model fitted ZLP.

        Parameters
        ----------
        function : str, {'Gaussian', 'Split Gausian', 'Lorentzian', 'Pearson VII', 'Split Pearson VII', 'Pseudo-Voigt', 'Split Pseudo-Voigt', 'Generalised Peak', 'Kaiser window'}
            Model choice for the ZLP.
        kwargs: dict, optional
            Additional keyword arguments.  

        Returns
        -------
        numpy.ndarray
            1D array of signal spectrum minus model fitted ZLP. 
        """
        # SUBTRACT OBTAINED MODEL FROM THE SIGNAL
        self.y_signal_subtracted = \
        self.y_signal - self.model(self.x_signal, function, **kwargs)
        # Intensity is a positive definite quantity
        self.y_signal_subtracted[self.y_signal_subtracted < 0] = 0

    def create_new_plot(self):
        """Create pyplot.subplots figure with axes."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-4, 4)

    def plot_all_results(self, i=None, j=None, monte_carlo=False):
        """Plot all components: original signal, Model-fit ZLP, subtracted
        spectrum, Lorentzian-fit gain peak."""
        self.plot_signal(i=i, j=j, monte_carlo=monte_carlo)
        self.plot_model(i=i, j=j, monte_carlo=monte_carlo)
        self.plot_subtracted(i=i, j=j, monte_carlo=monte_carlo)
        self.plot_gain_peak()
        # self.plot_loss_peak()
        self.ax.legend()

    def set_ax_title(self, title="Title here."):
        self.ax.set_title(title)

    def plot_signal(self, clr='b', label='spX', linewidth=2,
                    i=None, j=None, monte_carlo=False, **kwargs):
        if monte_carlo:
            self.y_signal = self.image.get_pixel_signal(i=i, j=j, **kwargs)
        self.ax.plot(self.x_signal, self.y_signal,
                     c=clr, lw=linewidth, label=label)

    def generate_model(self, function='Gaussian', x=np.arange(-4, 4, 1e-3), **kwargs):
        # Get coordinates for Gaussian fit of the ZLP
        # self.xm = np.arange(-4, 4, 1e-3)  # energy loss (eV)
        self.ym_model = self.model(x, function, **kwargs)

    def plot_model(self, clr='b', label='spX ZLP fit',
                   linestyle='dashed', linewidth=2,
                   i=None, j=None, monte_carlo=False,
                   plot_intervals=True):
        if monte_carlo:
            ym_model = self.get_model(i=i, j=j)
            if plot_intervals:
                self.ax.fill_between(self.x_signal, ym_model[1], ym_model[2], 
                                alpha=0.2, color=clr)
            self.ax.plot(self.x_signal, ym_model[0],
                        c=clr, ls=linestyle, label=label, lw=linewidth)
        else:
            self.generate_model()
            self.ax.plot(self.xm, self.ym_model,
                     c=clr, ls=linestyle, label=label, lw=linewidth)

    def plot_subtracted(self, clr='b', label='subtracted',
                        linestyle='dotted', linewidth=2,
                        i=None, j=None, monte_carlo=False,
                        plot_intervals=True):
        if monte_carlo:
            subtracted_signal = self.get_subtracted_spectrum(i=i, j=j, return_conf_interval=plot_intervals)
            if plot_intervals:
                self.ax.fill_between(self.x_signal, subtracted_signal[1], subtracted_signal[2], 
                                alpha=0.2, color=clr)
            self.ax.plot(self.x_signal, subtracted_signal[0],
                        c=clr, ls=linestyle, label=label, lw=linewidth)
        else:
            self.ax.plot(self.x_signal, self.y_signal_subtracted,
                     c=clr, ls=linestyle, label=label, lw=linewidth)

    def perr(self, pcov):
        return np.sqrt(np.diag(pcov))

    def fwhm_idx(self, y):
        i1 = np.argwhere(y >= np.amax(y)/2)[0][0] - 1
        i2 = np.argwhere(y >= np.amax(y)/2)[-1][0] + 1
        return i1, i2

    def fwhm_area(self, y):
        i1, i2 = self.fwhm_idx(y)
        return y[i1:i2].sum()

    def plot_gain_peak(self, clr='g', linewidth=2, label='mode #1 gain'):
        try:
            self.ym_gain
        except AttributeError:
            self.fit_gain_peak()
        self.ax.plot(self.xm, self.ym_gain, c=clr, lw=linewidth, label=label)
        i1, i2 = self.fwhm_idx(y=self.ym_gain)
        self.ax.fill_between(self.xm[i1:i2], self.ym_gain[i1:i2], color=clr, alpha=0.2)
        print(
            f"gain : x0 = {self.popt_gain[0]:.2f} +/- {self.perr(self.pcov_gain)[0]:.2f}")
        self.ax.axvline(self.popt_gain[0], 0, 1e9, color='k',
                        linestyle='dashdot', linewidth=2)

    def plot_loss_peak(self, clr='r', linewidth=2, label='mode #1 loss'):
        try:
            self.ym_loss
        except AttributeError:
            self.fit_loss_peak()
        self.ax.plot(self.xm, self.ym_loss, c=clr, label=label, lw=linewidth)
        self.ax.fill_between(self.xm, self.ym_loss, color=clr, alpha=0.2)
        print(
            f"loss : x0 = {self.popt_loss[0]:.2f} +/- {self.perr(self.pcov_loss)[0]:.2f}")
        self.ax.axvline(self.popt_loss[0], 0, 1e9, color='k',
                        linestyle='dashdot', linewidth=2)

    def plot_spectral_image(self, array, figsize=(4, 4), cmap='turbo_r'):
        fig0, ax0 = plt.subplots(figsize=figsize)
        ax0.imshow(array, cmap=cmap)
        return fig0, ax0

    def set_total_count_gain_peak(self):
        self.total_count_gain_peak = self.ym_gain.sum()

    def set_total_count_loss_peak(self):
        self.total_count_loss_peak = self.ym_loss.sum()

    def print_results(self):
        print(f"count={self.total_count_gain_peak:.1e}, popt={self.popt_gain}")
        # print(f"count={self.total_count_loss_peak:.1e}, popt={self.popt_loss}")

    def array2D_of_intensity_at_energy_loss(self, eloss=-1.1):
        """Generate 2D array of intensities for each SI-pixel at a chosen energy 
        loss.
        """
        y = self.store_pooled_spectrum_from_each_pixel_in_one_array()
        arr = np.zeros(shape=(self.n_rows, self.n_cols))
        arr[:] = np.nan
        for i in range(y.shape[1]):
            m = int(y[0, i])
            n = int(y[1, i])
            # intensity at eloss -2.1eV
            arr[m][n] = np.interp(eloss, self.x_signal, y[2:, i])
        return arr

    def plot_array2D_of_intensity_at_energy_loss(self, eloss=-1.1,
                                                 cmap='turbo',
                                                 cbar_title='intensity',
                                                 dm4_filename=''):
        """Example heatmap plot: SI with only intensity at energy loss -1.1eV.

        Args:
            eloss (float, optional): energy loss value. Defaults to -1.1.
            cmap (str, optional): color map. Defaults to 'turbo'.
            cbar_title (str, optional): color bar title. Defaults to 'intensity'.
            dm4_filename (str, optional): file name. Defaults to ''.

        Returns:
            fig: pyplot figure
            ax:  pyplot axis
        """
        fig, ax = plt.subplots()
        arr = self.array2D_of_intensity_at_energy_loss(eloss)  # [eV]
        p = ax.imshow(arr, cmap=cmap)
        ax.set_title(f"{dm4_filename}\n@ {eloss:.1f} eV")
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_title(cbar_title)  # intensity lower for thicker sample
        return fig, ax


class RectangleInSpectralImage:
    """
    Create rectangle in spectral image.
    """

    def __init__(self, image) -> None:
        """Load spectral image.

        Args:
            image (dm4): spectral image.
        """
        self.img = image
        self.intensity = np.sum(image.data, axis=2)
        self.row_i = 1                                 # vertical index top
        self.row_f = int(self.intensity.shape[0]/2)    # vertical index bottom
        self.col_i = 1                                 # horizontal index left
        self.col_f = int(self.intensity.shape[0]/2)    # horizontal index right

    def get_rectangle(self):
        width = self.col_f - self.col_i
        height = self.row_f - self.row_i
        self.rectangle = patches.Rectangle((self.col_i, self.row_i),
                                           width, height,
                                           linewidth=1,
                                           edgecolor='r',
                                           linestyle='dashed',
                                           facecolor='none',
                                           )

    def plot_spectral_image(self, figsize=(4, 4), cmap='turbo_r'):
        self.fig0, self.ax0 = plt.subplots(figsize=figsize)
        self.ax0.imshow(self.intensity, cmap=cmap)

    def plot_highlight_single_pixel(self, row=0, col=0, color_pixel='w', marker='*'):
        # Choose spectrum and mark its pixel in the Spectral Image.
        self.plot_spectral_image()
        self.ax0.scatter(col, row, color=color_pixel, marker=marker)

    def plot_rectangle(self):
        self.plot_spectral_image()
        self.get_rectangle()
        self.ax0.add_patch(self.rectangle)  # rectangle

    def plot_spectra_within_rectangle(self,
                                      figsize=(4, 4),
                                      xmin=-2,
                                      xmax=2,
                                      ymin=0,
                                      ymax=300000):
        self.fig, self.ax1 = plt.subplots(figsize=figsize)
        self.ax1.set_xlim(xmin, xmax)
        self.ax1.set_ylim(ymin, ymax)
        for row in range(self.row_i, self.row_f):
            for col in range(self.col_i, self.col_f):
                # plot each spectrum within the rectangle
                signal = self.img.get_pixel_signal(i=row, j=col)
                self.ax1.plot(self.img.eaxis, signal)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def save_figure(fig, name, plot_type, dpi=300, pad_inches=0):
    path_to_fig = f"./figs/{name}_{plot_type}.pdf"
    fig.savefig(path_to_fig, dpi=dpi, bbox_inches='tight',
                pad_inches=pad_inches)
    print(f"Saved under {path_to_fig}")


def summary_distribution(data, mean=50, lower=16, upper=84):
    median = np.nanpercentile(data, mean, axis=0)
    low = np.nanpercentile(data, lower, axis=0)
    high = np.nanpercentile(data, upper, axis=0)
    return [median, low, high]

