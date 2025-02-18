import numpy as np
import random
import os
import fnmatch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import copy

from sklearn.model_selection import train_test_split
from kneed import KneeLocator

from ..plotting.hyperparameters import plot_hp


class TrainZeroLossPeak:
    def __init__(self,
                 spectra,
                 eaxis,
                 cluster_centroids=None,
                 display_step=1000,
                 training_report_step=1,
                 n_batch_of_replica=1,
                 n_batches=1,
                 n_replica=100,
                 n_epochs=1000,
                 shift_de1=1.,
                 shift_de2=1.,
                 regularisation_constant=10.,
                 path_to_models='./models/',
                 remove_temp_files=True,
                 **kwargs
                 ):
        r"""
        The TrainZeroLossPeak class provides the tools to train the ZLP models for the spectra.

        Parameters
        ----------
        spectra
        eaxis
        cluster_centroids
        path_to_models
        display_step
        training_report_step
        n_batch_of_replica
        n_batches
        n_replica
        n_epochs
        shift_de1
        shift_de2
        regularisation_constant : float
            Constant that weighs the fit accuracy to the regularisation term
        remove_temp_files
        kwargs

        """

        self.spectra = spectra
        self.eaxis = eaxis
        self.deltaE = np.abs(self.eaxis[1] - self.eaxis[0])
        if cluster_centroids is None:
            self.cluster_centroids = np.zeros(1)
        else:
            self.cluster_centroids = cluster_centroids
        self.n_ebins = len(self.eaxis)
        self.n_clusters = len(self.spectra)
        self.path_to_models = path_to_models
        self.display_step = display_step
        self.training_report_step = training_report_step
        self.n_batch_of_replica = n_batch_of_replica
        self.n_replica = n_replica
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        if type(shift_de1) == float or type(shift_de1) == int:
            self.shift_de1 = np.ones(len(self.cluster_centroids)) * shift_de1
        elif len(shift_de1) == len(self.cluster_centroids):
            self.shift_de1 = shift_de1
        else:
            print("Makes sure shift_de1 is of type float or int or an array of the same length as the clusters")
        if type(shift_de2) == float or type(shift_de2) == int:
            self.shift_de2 = np.ones(len(self.cluster_centroids)) * shift_de2
        elif len(shift_de2) == len(self.cluster_centroids):
            self.shift_de2 = shift_de2
        else:
            print("Makes sure shift_de2 is of type float or int or an array of the same length as the clusters")
        self.regularisation_constant = regularisation_constant
        self.remove_temp_files = remove_temp_files

    def train_zlp_models_scaled(self, lr=1e-3):
        r"""
        Train the ZLP models. This functions calls up the other functions step by step to complete the whole process.
        Refer to each function for details what they specifically perform.

        Parameters
        ----------
        lr : float,
            Learning rate of the neural network

        """
        # Sets the display step in the console / log files per how many epochs the status of the training goes.
        if self.display_step is None:
            self.print_progress = False
            self.display_step = 1E6
        else:
            self.print_progress = True

        if not os.path.exists(self.path_to_models):
            os.makedirs(self.path_to_models)

        print("preparing hyperparameters!")
        self.set_minimum_intensities()
        self.set_y_data()
        self.set_dydx_data()
        self.set_sigma()
        self.find_fwhm_idx()
        self.find_difflog_local_minmax_idx()
        self.find_local_min_idx()
        self.find_kneedle_idx()
        self.de1, self.de2, self.mde1, self.mde2 = self.calculate_hyperparameters()
        if self.print_progress:
            print("dE1:", np.round(self.de1, 3))
            print("dE2:", np.round(self.de2, 3))
        self.scale_eaxis()
        self.calc_scale_var_log_int_i()
        self.save_scale_var_log_int_i()
        self.save_hyperparameters()
        print("Hyperparameters prepared!")

        self.loss_test_reps = np.zeros(self.n_replica)
        self.loss_train_reps = np.zeros(self.n_replica)
        for i in range(self.n_replica):
            j = i + self.n_replica * self.n_batch_of_replica - self.n_replica + 1
            path_nn_replica = os.path.join(self.path_to_models, 'nn_rep_{}'.format(j))
            if self.print_progress:
                print("Started training on replica number {}".format(j) + ", at time ", dt.datetime.now())
            self.data_y = np.empty((0, 1))
            self.data_x = np.empty((0, 2))
            self.data_sigma = np.empty((0, 1))
            # Use a list of tensors, because data from the different spectra in the replica will not
            # have equal shapes generally and Torch/Numpy cannot handle non-rectangular arrays.
            self.data_x_for_derivative = []
            for cluster_label in range(self.n_clusters):
                # Initialize the data of the replica by taking a spectra from each cluster
                self.initialize_x_y_sigma_input(cluster_label)

            # Split the replica into a train set and a test set
            self.train_test = train_test_split(self.data_x, self.data_y, self.data_sigma, test_size=0.25)

            self.set_train_x_y_sigma()
            self.set_test_x_y_sigma()

            # Train the replica/model
            self.model = MultilayerPerceptron(num_inputs=2, num_outputs=1)
            self.model.apply(weight_reset)
            self.train_and_evaluate_model(i, j, lr=lr)

            # save model state when max number of allowed epochs has been reached
            torch.save(self.model.state_dict(), path_nn_replica)

            # make a training report for every replica
            if self.training_report_step > 0:
                if j % self.training_report_step == 0 or j == 1:
                    self.set_path_for_training_report(j)
                    self.write_txt_of_loss(self.training_report_path, 'train_loss', self.loss_train_n)
                    self.write_txt_of_loss(self.training_report_path, 'test_loss', self.loss_test_n)
                    self.plot_training_report()

        self.write_txt_of_loss(self.path_to_models, f'costs_train_{self.n_batch_of_replica}', self.loss_train_reps)
        self.write_txt_of_loss(self.path_to_models, f'costs_test_{self.n_batch_of_replica}', self.loss_test_reps)

        # Check if training procedure is finished by counting the number of nn_rep files.
        # If required number of files is present the output is cleaned up.
        num_files = len(fnmatch.filter(os.listdir(self.path_to_models), 'nn_rep_*'))
        self.required_num_files = self.n_replica * self.n_batches
        if num_files == self.required_num_files:
            self.cleanup_files()

    def set_minimum_intensities(self):
        r"""
        Set all features smaller than 1 to 1.

        """

        for i in range(self.n_clusters):  # There is no even amount of spectra in each cluster, so we have to loop
            self.spectra[i][self.spectra[i] < 1] = 1

    def set_y_data(self):
        r"""
        Smooths all the spectra per cluster and takes the median per cluster.

        """

        y_raw = self.spectra
        self.y_raw_median = np.zeros(self.n_clusters, dtype=object)
        self.y_log_median = np.zeros(self.n_clusters, dtype=object)
        for i in range(self.n_clusters):
            if len(y_raw[i]) == 1:
                self.y_raw_median[i] = y_raw[i][0]
            else:
                self.y_raw_median[i] = np.nanpercentile(y_raw[i], 50, axis=0)
            self.y_log_median[i] = np.log(self.y_raw_median[i])

    def set_dydx_data(self):
        r"""
        Determines the slope of all spectra per cluster, smooths the slope and takes the median per cluster.

        """

        self.dydx_median = np.zeros(self.n_clusters, dtype=object)
        self.dydx_median_smooth_strong = np.zeros(self.n_clusters, dtype=object)
        self.dydx_median_log = np.zeros(self.n_clusters, dtype=object)
        self.dydx_median_smooth_light_log = np.zeros(self.n_clusters, dtype=object)
        for i in range(self.n_clusters):
            self.dydx_median[i] = np.diff(self.y_raw_median[i], axis=0)
            self.dydx_median_smooth_strong[i] = smooth_signal(self.dydx_median[i], window_length=31)
            self.dydx_median_log[i] = np.diff(self.y_log_median[i], axis=0)
            self.dydx_median_smooth_light_log[i] = smooth_signal(self.dydx_median_log[i], window_length=11)

    def set_sigma(self):
        r"""
        Determine the sigma (spread of spectra per cluster) per cluster.

        """

        if self.n_clusters == 1 and len(self.spectra[0]) == 1:
            print("Single spectra spotted, sigma will be determined by bootstrap parameterization")
            self.sigma = None
        else:
            self.sigma = np.zeros((self.n_clusters, self.n_ebins))
            for i in range(self.n_clusters):
                if len(self.spectra[i]) == 1:
                    print("Warning! Only a single spectra in the cluster, sigma will be set to 0.")
                    self.sigma[i, :] = np.zeros(self.n_ebins)
                else:
                    ci_low = np.nanpercentile(np.log(self.spectra[i]), 16, axis=0)
                    ci_high = np.nanpercentile(np.log(self.spectra[i]), 84, axis=0)
                    self.sigma[i, :] = np.absolute(ci_high - ci_low)

    def find_fwhm_idx(self):
        r"""
        Determine the FWHM indices per cluster (Full Width at Half Maximum):
            - indices of the left and right side of the ZLP
            - indices of the left and right side of the log of the ZLP

        These are all determine by taking the local minimum and maximum of the dy/dx

        """

        self.hwhm_gain_idx = np.zeros(self.n_clusters, dtype=int)
        self.hwhm_loss_idx = np.zeros(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            half_max = np.argwhere(
                self.y_raw_median[i][:int((5+np.abs(self.eaxis[0]))/self.deltaE)] > np.max(self.y_raw_median[i])/2).flatten()
            self.hwhm_gain_idx[i] = half_max[0] - 1
            self.hwhm_loss_idx[i] = half_max[-1] + 1

        # Values of the FWHMs
        self.fwhm = (self.eaxis[self.hwhm_loss_idx] - self.eaxis[self.hwhm_gain_idx])

    def find_difflog_local_minmax_idx(self):
        r"""
        Determine the local minimum and maximum of the differential of the log of the signals per cluster

        Returns
        -------

        """

        self.difflog_localmin_idx = np.zeros(self.n_clusters, dtype=int)
        self.difflog_localmax_idx = np.zeros(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            if 10*self.fwhm[i] >= np.abs(self.eaxis[0]):
                lower_bound_idx = 0
            else:
                lower_bound_idx = np.argwhere(self.eaxis < -10*self.fwhm[i]).flatten()[0]
            self.difflog_localmin_idx[i] = np.argmin(
                self.dydx_median_smooth_light_log[i][self.hwhm_gain_idx[i]:int((10*self.fwhm[i]+np.abs(self.eaxis[0]))/self.deltaE)]) + self.hwhm_gain_idx[i]
            self.difflog_localmax_idx[i] = np.argmax(
                self.dydx_median_smooth_light_log[i][lower_bound_idx:self.hwhm_loss_idx[i]]) + lower_bound_idx

    def find_local_min_idx(self):
        r"""
        Determine the first local minimum index of the signals per cluster by setting it to the point where the
        derivative crosses zero.

        """

        self.local_min_loss_idx = np.zeros(self.n_clusters, dtype=int)
        self.local_min_gain_idx = np.zeros(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            crossing_loss = (self.dydx_median_smooth_strong[i][self.difflog_localmin_idx[i]:] > 0)
            if not crossing_loss.any():
                print("No crossing found in loss region cluster " + str(i) + ", finding minimum of absolute of dydx")
                self.local_min_loss_idx[i] = np.argmin(
                    np.absolute(self.dydx_median_smooth_strong[i])[self.difflog_localmin_idx[i]:]) + self.difflog_localmin_idx[i]
            else:
                self.local_min_loss_idx[i] = np.argwhere(
                    self.dydx_median_smooth_strong[i][self.difflog_localmin_idx[i]:] > 0).flatten()[0] + self.difflog_localmin_idx[i]

            crossing_gain = (self.dydx_median_smooth_strong[i][:self.difflog_localmax_idx[i]] < 0)
            if not crossing_gain.any():
                print("No crossing found in gain region cluster " + str(i) + ", finding minimum of absolute of dydx")
                self.local_min_gain_idx[i] = np.argmin(
                    np.absolute(self.dydx_median_smooth_strong[i])[:self.difflog_localmax_idx[i]])
            else:
                self.local_min_gain_idx[i] = np.argwhere(
                    self.dydx_median_smooth_strong[i][:self.difflog_localmax_idx[i]] < 0).flatten()[-1]

    def find_kneedle_idx(self):
        r"""
        Find the kneedle index per cluster.
        The kneedle algorithm is used to find the point of highest curvature in your concave or convex data set.

        """

        self.kneedle_loss_idx = np.zeros(self.n_clusters, dtype=int)
        self.kneedle_gain_idx = np.zeros(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            x_loss_range = self.eaxis[self.difflog_localmin_idx[i]:self.local_min_loss_idx[i]]
            y_loss_range = self.y_log_median[i][self.difflog_localmin_idx[i]:self.local_min_loss_idx[i]]
            kneedle_loss = KneeLocator(x=x_loss_range, y=y_loss_range, curve='convex', direction='decreasing')
            self.kneedle_loss_idx[i] = np.argwhere(self.eaxis > kneedle_loss.knee).flatten()[0]

            x_gain_range = self.eaxis[self.local_min_gain_idx[i]:self.difflog_localmax_idx[i]]
            y_gain_range = self.y_log_median[i][self.local_min_gain_idx[i]:self.difflog_localmax_idx[i]]
            kneedle_gain = KneeLocator(x=x_gain_range, y=y_gain_range, curve='convex')
            if kneedle_gain.knee is None:
                kneedle_gain.knee = x_gain_range[0]
            self.kneedle_gain_idx[i] = np.argwhere(self.eaxis < kneedle_gain.knee).flatten()[-1]

    def calculate_hyperparameters(self):
        r"""
        Calculate the values of the hyperparameters in the gain and loss region, dE1 and mdE1 are calculated by taking
        the location of the kneedles at each side of the ZLP and shifting them with the gives shift value.

        dE2 and mdE2 are calcualted by taking the value of the eaxis where a fit of the log10 function intersects with
        a single count. If this value is not found the end point of the signal is taken as location for dE2.

        """

        def _find_log10_fit(x, y, idx1, idx2):
            r"""
            Calculates the log10 fit

            Parameters
            ----------
            x
            y
            idx1
            idx2

            Returns
            -------

            """

            slope = (np.log10(y[idx2]) - np.log10(y[idx1])) / (x[idx2] - x[idx1])
            factor = 10 ** (np.log10(y[idx2]) - slope * x[idx2])
            return log10_fit(x, slope, factor)

        de1 = np.zeros(self.n_clusters)
        de2 = np.zeros(self.n_clusters)
        mde1 = np.zeros(self.n_clusters)
        mde2 = np.zeros(self.n_clusters)
        self.intersect_loss_idx = np.zeros(self.n_clusters, dtype=int)
        self.intersect_gain_idx = np.zeros(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            # loss
            de1[i] = self.eaxis[self.kneedle_loss_idx[i]] * self.shift_de1[i]
            log10_loss = _find_log10_fit(x=self.eaxis, y=self.y_raw_median[i],
                                         idx1=self.kneedle_loss_idx[i], idx2=self.local_min_loss_idx[i])
            intersect_loss_single_count = (log10_loss < 1)
            if not intersect_loss_single_count.any():
                print("Log10 fit cluster ", i
                      , " does not cross a single count, setting intersect index to second last eaxis index")
                self.intersect_loss_idx[i] = int(len(self.eaxis) - 2)
            else:
                self.intersect_loss_idx[i] = np.argwhere(log10_loss < 1).flatten()[0]
            de2[i] = self.eaxis[self.intersect_loss_idx[i]] * self.shift_de2[i]
            if de2[i] > self.eaxis[-2]:
                print(de2[i], " is shifted beyond the eaxis, setting de2 to the second last value of the eaxis")
                de2[i] = self.eaxis[-2]

            # gain
            mde1[i] = self.eaxis[self.kneedle_gain_idx[i]] * self.shift_de1[i]
            log10_gain = _find_log10_fit(x=self.eaxis, y=self.y_raw_median[i],
                                         idx1=self.local_min_gain_idx[i], idx2=self.kneedle_gain_idx[i])
            intersect_gain_single_count = (log10_gain < 1)
            if not intersect_gain_single_count.any():
                print("Log10 fit cluster ", i
                      , " does not cross a single count, setting intersect index to second eaxis index")
                self.intersect_gain_idx[i] = int(len(self.eaxis) - 2)
            else:
                self.intersect_gain_idx[i] = np.argwhere(log10_gain < 1).flatten()[-1]
            mde2[i] = self.eaxis[self.intersect_gain_idx[i]] * self.shift_de2[i]
            if mde2[i] < self.eaxis[1]:
                print(mde2[i], " is shifted beyond the eaxis, setting de2 to the second value of the eaxis")
                mde2[i] = self.eaxis[1]
        return de1, de2, mde1, mde2

    def scale_eaxis(self):
        r"""
        Scales the features of the energy axis between [0.1, 0.9]. This is to optimize the speed
        of the neural network.

        """

        scale_var_eaxis = find_scale_var(self.eaxis)
        self.eaxis_scaled = scale(self.eaxis, scale_var_eaxis)

    def calc_scale_var_log_int_i(self, based_on='log_zlp'):
        r"""
        Calculate the scale variables of the log of the integrated intensity of the spectra for the three highest bins
        of the Zero Loss Peak.

        """

        # Collect all spectra from the data set into a single array
        all_spectra = np.empty((0, self.n_ebins))
        if self.n_clusters == 1 and len(self.spectra[0]) == 1:
            for i in range(len(self.spectra)):
                all_spectra = np.append(all_spectra, self.spectra[i] + self.spectra[i] / 2, axis=0)
                all_spectra = np.append(all_spectra, self.spectra[i] / 2, axis=0)
        else:
            for i in range(len(self.spectra)):
                all_spectra = np.append(all_spectra, self.spectra[i], axis=0)

        if based_on == 'log_zlp':
            log_int_i = np.zeros(len(all_spectra))
            for i in range(len(all_spectra)):
                max_idx = np.argmax(all_spectra[i])
                log_int_i[i] = np.log(np.sum(all_spectra[i][max_idx - 1:max_idx + 2]))
        else:
            log_int_i = np.log(np.sum(all_spectra, axis=1)).flatten()
        self.scale_var_log_int_i = find_scale_var(log_int_i)
        del all_spectra

    def save_scale_var_log_int_i(self):
        r"""
        Save the scale variables of the log of the total integrated intensity of the spectra, denoted ``I``.

        """

        path_scale_var = os.path.join(self.path_to_models, 'scale_var.txt')
        if not os.path.exists(path_scale_var):
            np.savetxt(path_scale_var, self.scale_var_log_int_i)

    def save_hyperparameters(self):
        r"""
        Save the hyperparameters in hyperparameters.txt. These are:
            - cluster centroids, keep note if they were determined from the raw data, or if the log had been taken.
            - dE1 for all clusters
            - dE2 for all clusters
            - FWHM for all clusters

        """

        print("I'm saving hyperparameters.txt so hang on...")
        print(f"clusters centroids={self.cluster_centroids} size {self.cluster_centroids.shape}")
        print(f"dE1={self.de1} has size {self.de1.shape}")
        print(f"dE2={self.de2} has size {self.de2.shape}")
        print(f"FWHM={self.fwhm} has size {self.fwhm.shape}")
        p1 = os.path.join(self.path_to_models, "hyperparameters.txt")
        np.savetxt(p1, np.vstack((self.cluster_centroids, self.de1, self.de2, self.fwhm)))
        print("Saved hyperparameters.txt!")

    def initialize_x_y_sigma_input(self, cluster_label):
        r"""
        Initialize the x, y and sigma input for the Neural Network. The spectrum is split into the
        3 regions as given by the toy model. For the y data, the data in region I is set to the log intensity
        up to dE1, the data in region III is set to zero. For the x data two input features, first is the values
        of the energy axis in region I and III, second is the rescaled log of the total integrated intensity.
        This factor is to ensure symmetry is retained between input and output values. For the sigma data

        Parameters
        ----------
        cluster_label : int
            Label of the cluster

        """

        # From the amount of spectra in the cluster, pick a random signal
        n_spectra = len(self.spectra[cluster_label])
        idx = random.randint(0, n_spectra - 1)
        signal = self.spectra[cluster_label][idx]

        # Get the size of regions I and III,
        # we will construct our replica with the information we have from these regions.
        # region1 = len(self.eaxis[self.eaxis < self.de1[cluster_label]])
        # region3 = len(self.eaxis[self.eaxis > self.de2[cluster_label]])

        # First line is mask where condition holds
        # Second line are the indices where the condition holds
        # Third line is the number of elements where condition holds
        region1 = self.eaxis < self.de1[cluster_label]
        region2 = np.logical_and(self.eaxis >= self.de1[cluster_label], self.eaxis <= self.de2[cluster_label])
        region3 = self.eaxis > self.de2[cluster_label]
        idx_region1 = np.nonzero(region1)
        idx_region2 = np.nonzero(region2)
        idx_region3 = np.nonzero(region3)
        num_elements_region1 = np.count_nonzero(region1)
        num_elements_region2 = np.count_nonzero(region2)
        num_elements_region3 = np.count_nonzero(region3)

        # The y data is constructed by taking the log of the data in region I from the random spectra and
        # by setting the data to zero in region III for the random spectra. These are then added to the array.
        self.data_y = np.append(self.data_y, np.log(signal[idx_region1]))
        self.data_y = np.append(self.data_y, np.zeros(num_elements_region3))

        # The x data has two input features: 1) the scaled energy axis,
        # 2) the scaled log of the integrated intensity of the random spectrum.
        features = np.ones((num_elements_region1 + num_elements_region3, 2))
        features[:num_elements_region1, 0] = self.eaxis_scaled[idx_region1]
        features[-num_elements_region3:, 0] = self.eaxis_scaled[idx_region3]

        features_for_derivative = np.ones((num_elements_region2, 2))
        features_for_derivative[:, 0] = self.eaxis_scaled[idx_region2]

        max_idx = np.argmax(signal)
        log_int_i = np.log(np.sum(signal[max_idx - 1:max_idx + 2]))
        features[:, 1] = scale(log_int_i, self.scale_var_log_int_i)
        features_for_derivative[:, 1] = scale(log_int_i, self.scale_var_log_int_i)
        self.data_x = np.concatenate((self.data_x, features))
        self.data_x_for_derivative.append(torch.from_numpy(features_for_derivative))

        # The values of sigma calculated from the cluster will be our input data for sigma
        self.data_sigma = np.append(self.data_sigma, self.sigma[cluster_label][idx_region1])
        self.data_sigma = np.append(self.data_sigma, 0.8 * np.ones(num_elements_region3))

    def set_train_x_y_sigma(self):
        r"""
        Take the x, y and sigma data for the train set and reshape them for neural network input

        """

        self.train_x = self.train_test[0]
        self.train_y = self.train_test[2]
        self.train_sigma = self.train_test[4]
        n_train = len(self.train_x)
        self.train_x = self.train_x.reshape(n_train, 2)
        self.train_y = self.train_y.reshape(n_train, 1)
        self.train_sigma = self.train_sigma.reshape(n_train, 1)
        self.train_x = torch.from_numpy(self.train_x)
        self.train_y = torch.from_numpy(self.train_y)
        self.train_sigma = torch.from_numpy(self.train_sigma)

    def set_test_x_y_sigma(self):
        r"""
        Take the x, y and sigma data for the test set and reshape them for neural network input

        """

        self.test_x = self.train_test[1]
        self.test_y = self.train_test[3]
        self.test_sigma = self.train_test[5]
        n_test = len(self.test_x)
        self.test_x = self.test_x.reshape(n_test, 2)
        self.test_y = self.test_y.reshape(n_test, 1)
        self.test_sigma = self.test_sigma.reshape(n_test, 1)
        self.test_x = torch.from_numpy(self.test_x)
        self.test_y = torch.from_numpy(self.test_y)
        self.test_sigma = torch.from_numpy(self.test_sigma)

    def train_and_evaluate_model(self, i, j, lr):
        r"""
        Train and evaluate the model. Also saves the values of cost_train and cost_test per epoch.

        Parameters
        ----------
        i : int
        j : int
        lr : float
            learning rate

        """

        # store the test and train loss per epoch
        self.loss_test_n = []
        self.loss_train_n = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, self.n_epochs + 1):
            # Set model to training mode
            self.model.train()
            output_train = self.model(self.train_x.float())
            # Because self.train_x_for_derivative is a list of tensors
            # we need to explicitly loop over the entries to compute the outputs
            output_for_derivative_train = []
            for input_tensor in self.data_x_for_derivative:
                output_for_derivative_train.append(self.model(input_tensor.float()))
            loss_train = self.loss_function(output_train, output_for_derivative_train, self.train_y, self.train_sigma)
            self.loss_train_n.append(loss_train.item())

            # update weights
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Set model to evaluation mode
            self.model.eval()
            with torch.no_grad():
                output_test = self.model(self.test_x.float())
                output_for_derivative_test = []
                for input_tensor in self.data_x_for_derivative:
                    output_for_derivative_test.append(self.model(input_tensor.float()))
                loss_test = self.loss_function(output_test, output_for_derivative_test, self.test_y, self.test_sigma)
                self.loss_test_n.append(loss_test.item())
                if epoch % self.display_step == 0 and self.print_progress:
                    training_loss = round(loss_train.item(), 3)
                    testing_loss = round(self.loss_test_n[epoch - 1], 3)
                    print('----------------------')
                    print(f'Rep {j}, Epoch {epoch}')
                    print(f'Training loss {training_loss}')
                    print(f'Testing loss {testing_loss}')

                # update the test and train loss of the replica
                self.loss_test_reps[i] = loss_test.item()
                self.loss_train_reps[i] = loss_train.item()

    def loss_function(self, output, output_for_derivative, target, error):
        r"""
        The loss function to train the ZLP takes the model ``output``, the raw  spectrum ``target`` and the associated
        ``error``. The latter corresponds to  the one sigma spread within a given cluster at fixed :math:`\Delta E`.
        It returns the cost function :math:`C_{\mathrm{ZLP}}^{(m)}` associated with the replica :math:`m` as

        .. math:: :label: eq:lossfunction

            C_{\mathrm{ZLP}}^{(m)} = \frac{1}{n_{E}} \sum_{k=1}^K \sum_{\ell_k=1}^{n_E^{(k)}} \frac{\left[I^{(i_{m,k}, j_{m,k})}(E_{\ell_k}) - I_{\rm ZLP}^{({\mathrm{NN}})(m)} \left(E_{\ell_k},\ln \left( N_{\mathrm{ tot}}^{(i_{m,k},j_{m,k})} \right) \right) \right]^2}{\sigma^2_k \left(E_{\ell_k} \right)}.

        Parameters
        ----------
        eaxis : np.ndarray
            Energy-loss axis
        output: torch.tensor
            Neural Network output
        output_for_derivative : list of torch.tensor
            Each entry in the list should correspond to the neural 
            network output between de1 and de2 of a single spectrum 
            in the replica
        target: torch.tensor
            Raw EELS spectrum
        error: torch.tensor
            Uncertainty on :math:`\log I_{\mathrm{EELS}}(\Delta E)`.

        Returns
        -------
        loss: torch.tensor
            Loss associated with the model ``output``.

        """

        cost = 0
        for output_tensor in output_for_derivative:
            derivative = torch.diff(output_tensor.flatten())
            # Penalise for bins where derivative is positive
            cost += torch.sum(derivative[derivative > 0])
        return torch.mean(torch.square((output - target) / error)) + self.regularisation_constant * cost

    def set_path_for_training_report(self, j):
        r"""
        Set the save directory for the training report of the replica being trained on.

        Parameters
        ----------
        j : int
            Index of the replica being trained on.

        """

        self.training_report_path = os.path.join(
            self.path_to_models, 'training_reports/rep_{}'.format(j))
        if not os.path.exists(self.training_report_path):
            os.makedirs(self.training_report_path)

    def plot_training_report(self):
        r"""
        Creat the training report plot: evolution of the training and validation
        loss per epoch.

        """

        fig, ax = plt.subplots()
        ax.plot(self.loss_train_n, label="Training Loss")
        ax.plot(self.loss_test_n, label="Validation Loss")
        ax.set_xlabel("epochs")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')

        path = os.path.join(self.training_report_path, 'loss.pdf')
        fig.savefig(path)

    def write_txt_of_loss(self, base_path, filename, loss):
        r"""
        Write train/test loss to a .txt file.

        Parameters
        ----------
        base_path : str
            Directory to store the report in.
        filename : str
            Filename of .txt file to store report in.
        loss : list
            List containing loss value per epoch.

        """

        path = os.path.join(
            base_path, f'{filename}.txt')
        with open(path, 'w') as text_file:
            for item in loss:
                text_file.write(f'{item}\n')

    def cleanup_files(self):
        r"""
        Cleans up the files generated by train_zlp_models_scaled.
        costs_train_*, costs_test_*, and nn_rep_* files are merged into single files
        costs_train, costs_test, and nn_parameters respectively.

        """

        # Output file names
        train_cost_path = os.path.join(
            self.path_to_models, 'costs_train.txt')
        test_cost_path = os.path.join(
            self.path_to_models, 'costs_test.txt')
        nn_replicas_path = os.path.join(
            self.path_to_models, 'nn_replicas')

        # Loop over all costs files, which are indexed starting from index 1.
        with open(train_cost_path, 'w') as text_file_train, open(test_cost_path, 'w') as text_file_test:
            for i in range(1, self.n_batches + 1):
                filename_train = os.path.join(
                    self.path_to_models, f'costs_train_{i}.txt')
                with open(filename_train) as f_train:
                    for line in f_train:
                        text_file_train.write(f'{float(line.strip())}\n')
                if self.remove_temp_files:
                    os.remove(filename_train)

                filename_test = os.path.join(
                    self.path_to_models, f'costs_test_{i}.txt')
                with open(filename_test) as f_test:
                    for line in f_test:
                        text_file_test.write(f'{float(line.strip())}\n')
                if self.remove_temp_files:
                    os.remove(filename_test)

        # Dictionary to store models in
        model_dict = dict()
        model = MultilayerPerceptron(num_inputs=2, num_outputs=1)
        # Loop over all nn_rep files, which are indexed starting from index 1.
        for i in range(1, self.required_num_files + 1):
            path_to_model = os.path.join(
                self.path_to_models, f'nn_rep_{i}')
            model.load_state_dict(torch.load(path_to_model))
            model_dict[f'model_{i}'] = copy.deepcopy(model.state_dict())
            if self.remove_temp_files:
                os.remove(path_to_model)

        torch.save(model_dict, nn_replicas_path)

    def save_figplot(self, fig, title="no_title.pdf"):
        r"""
        Display the computed values of dE1 (both methods) together with the
        raw EELS spectrum.

        Parameters
        ----------
        fig : matplotlib.Figure
            Figure to be saved.
        title : str
            Filename to store the plot in.

        """

        path = os.path.join(self.path_to_models, title)
        fig.savefig(path)
        print(f"Plot {title} stored in {path}")

    def plot_hp_cluster_slope(self, **kwargs):
        r"""
        Create a plot of the hyperparameters plotted on top of the slopes of the spectra per cluster.

        Parameters
        ----------
        kwargs: dict, optional
            Additional keyword arguments.

        """
        spectra_dydx = np.zeros(self.n_clusters, dtype=object)
        spectra_dydx_smooth = np.zeros(self.n_clusters, dtype=object)
        for i in range(self.n_clusters):
            spectra_dydx[i] = (self.spectra[i][:, 1:] - self.spectra[i][:, :-1]) / self.deltaE
            spectra_dydx_smooth[i] = smooth_signals_per_cluster(spectra_dydx[i], window_length=31)
        fig = plot_hp(eaxis=self.eaxis, clusters_data=spectra_dydx_smooth, de1=self.de1, de2=self.de2, **kwargs)
        return fig

    def plot_hp_cluster(self, **kwargs):
        r"""
        Create a plot of the hyperparameters plotted on top of the spectra per cluster.

        Parameters
        ----------
        kwargs: dict, optional
            Additional keyword arguments.

        """
        spectra_smooth = np.zeros(self.n_clusters, dtype=object)
        for i in range(self.n_clusters):
            spectra_smooth[i] = smooth_signals_per_cluster(self.spectra[i], window_length=31)
        fig = plot_hp(eaxis=self.eaxis, clusters_data=spectra_smooth, de1=self.de1, de2=self.de2, **kwargs)
        return fig


class MultilayerPerceptron(nn.Module):
    r"""
    Multilayer Perceptron (MLP) class. It uses the following architecture

    .. math::

       [n_i, 10, 15, 5, n_f],

    where :math:`n_i` and :math:`n_f` denote the number of input features and 
    output target values respectively.

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


# STATIC FUNCTIONS
def scale(inp, ab):
    r"""
    Rescale the training data to lie between 0.1 and 0.9. Rescaling features is to help speed up the neural network
    training process. The value range [0.1, 0.9] ensures the neuron activation states will typically lie close to the
    linear region of the sigmoid activation function.

    Parameters
    ----------
    inp: numpy.ndarray, shape=(M,)
        training data to be rescaled, e.g. :math:`\Delta E`
    ab: numpy.ndarray, shape=(M,)
        scaling parameters, which can be found with
        :py:meth:`find_scale_var() <find_scale_var>`.

    Returns
    -------
    Rescaled training data

    """

    return inp * ab[0] + ab[1]


def find_scale_var(inp, min_out=0.1, max_out=0.9):
    r"""
    Computes the scaling parameters needed to rescale the training data to lie 
    between ``min_out`` and ``max_out``. For our neural network the value range [0.1, 0.9] ensures the
    neuron activation states will typically lie close to the linear region of the sigmoid activation function.

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
    r"""
    Reset the weights and biases associated with the model ``m``.

    Parameters
    ----------
    m: MLP
        Model of type :py:meth:`MLP <MLP>`.

    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def smooth_signals_per_cluster(signals, window_length=51, window_type='hanning'):
    r"""
    Smooth all signals in a cluster using a window length ``window_len`` and a window type ``window``.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Parameters
    ----------
    signals: numpy.ndarray, shape=(M,)
        The input data
    window_length: int, optional
        The dimension of the smoothing window; should be an odd integer. Default is 51
    window_type: str, optional
        the type of window from ``"flat"``, ``"hanning"``, ``"hamming"``, ``"bartlett"``,
        ``"blackman"`` and ``"kasier"``. ``"flat"`` window will produce a moving average smoothing.
        Default is ``"hanning"``

    Returns
    -------
    signal_smooth: numpy.ndarray, shape=(M,)
        The smoothed signal

    """

    # Set window length to uneven number
    window_length += (window_length + 1) % 2

    # extend the signal with the window length on both ends
    signals_padded = np.r_['-1', signals[:, window_length - 1:0:-1], signals, signals[:, -2:-window_length - 1:-1]]

    # Pick the window type
    if window_type == 'flat':  # moving average
        window = np.ones(window_length, 'd')
    else:
        window = eval('np.' + window_type + '(window_length)')

    # Determine the smoothed signal and throw away the padded ends
    surplus_data = int((window_length - 1) * 0.5)

    def window_convolve(data):
        return np.convolve(data, window / window.sum(), mode='valid')

    signals_smooth = np.apply_along_axis(window_convolve, axis=1, arr=signals_padded)[:, surplus_data:-surplus_data]
    return signals_smooth


def smooth_signal(signal, window_length=51, window_type='hanning', beta=14.0):
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
        Default is ``'hanning'``
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

    # Pick the window type
    if window_type == 'flat':  # moving average
        window = np.ones(window_length, 'd')
    elif window_type == 'kaiser':
        window = np.kaiser(M=window_length, beta=beta)
    else:
        window = eval('np.' + window_type + '(window_length)')

    # Determine the smoothed signal and throw away the padded ends
    surplus_data = int((window_length - 1) * 0.5)
    signal_smooth = np.convolve(signal_padded, window / window.sum(), mode='valid')[surplus_data:-surplus_data]
    return signal_smooth


def log10_fit(x, a, b, order=1):
    return b * 10 ** (a * (x ** (1 / order)))


def power_fit(x, a, r):
    return a * x ** r
