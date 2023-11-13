import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def plot_dE1(image, y_smooth_clusters, dy_dx_clusters, min_clusters, de1_prob, de1_shift, prob):
    """
    Produces two plots of the locations of `dE1`:

    - The slope of the EELS spectrum for each cluster plus uncertainties.
    - The log EELS intensity per cluster plus uncertainties.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object
    y_smooth_clusters: array_like
        An array that contains an array for each cluster, which subsequently contains the smoothed spectrum at each
        pixel within the cluster.
    dy_dx_clusters: array_like
        An array that contains an array for each cluster, which subsequently contains the slope of the spectrum at each
        pixel within the cluster.
    min_clusters: array_like
        Location of first local minimum for each cluster
    dE1_prob: array_like
        Values of dE1 as determined from the 16% replica rule
    dE1_shift: array_like
        Values of dE1 as determined from the shifted first local minimum rule
    """

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 13})
    rc('text', usetex=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    der_deltaE = image.deltaE[:-1]

    # TODO: make dE2 more general
    dE1 = de1_prob if prob else de1_shift
    dE2 = dE1 * 3
    cluster_means = image.clusters

    n_clusters = len(dE1)
    ncols = 2
    nrows = n_clusters // ncols if n_clusters % ncols == 0 else n_clusters // ncols + 1
    wl1 = round(image.l / 20)

    fig = plt.figure(figsize=(ncols * 5, nrows * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # for cluster_idx in range(n_clusters):
    #     print(cluster_idx)
    #     ax = fig.add_subplot(nrows, ncols, cluster_idx + 1)
    #
    #     raw_spectra = image.get_cluster_spectra(conf_interval=1, signal='EELS')
    #     spectra_smooth = smooth_clusters(image, raw_spectra, wl1)
    #
    #     spectra_med = np.nanpercentile(spectra_smooth[cluster_idx], 50, axis=0)
    #     spectra_cl_high = np.nanpercentile(spectra_smooth[cluster_idx], 84, axis=0)
    #     spectra_cl_low = np.nanpercentile(spectra_smooth[cluster_idx], 16, axis=0)
    #
    #     ax.fill_between(image.deltaE, spectra_cl_low, spectra_cl_high, alpha=0.3)
    #     label = r'$\rm{Signal}$'
    #     ax.plot(image.deltaE, spectra_med, label=label)
    #     ax.axvline(dE1[cluster_idx], 0, 1, color='C3', linestyle='--')
    #     ax.axvline(dE2[cluster_idx], 0, 1, color='C3', linestyle='--')
    #
    #     cluster_mean = cluster_means[cluster_idx]
    #
    #     #TODO: can the code below be removed?
    #     #scaled_int = image.scale_var_log_sum_I[0] * cluster_mean + image.scale_var_log_sum_I[1]
    #     # zlp_cluster_mean = self.calc_zlp_ntot(scaled_int)
    #     # zlp_low = np.nanpercentile(zlp_cluster_mean, 16, axis=0)
    #     # zlp_high = np.nanpercentile(zlp_cluster_mean, 84, axis=0)
    #     # zlp_median = np.nanpercentile(zlp_cluster_mean, 50, axis=0)
    #     # label_zlp = r'$\rm{ZLP}$'
    #     #
    #     # ax.fill_between(self.deltaE, zlp_low, zlp_high, alpha=0.3)
    #     # ax.plot(self.deltaE, zlp_median, label=label_zlp)
    #     #ax.text(0.03, 0.30, r'$\rm{\log N_{\mathrm{tot}}=%.2f}$' % (scaled_int), horizontalalignment='left',
    #     #        verticalalignment='center', transform=ax.transAxes)
    #     title = r'$\rm{\;Cluster\;%d}$' % (cluster_idx)
    #     ax.set_title(title)
    #
    #     ax.set_ylim(1, 1e3)
    #     ax.set_xlim(0, dE2[cluster_idx] + 1)
    #     ax.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
    #     ax.set_ylabel(r"$I_{\rm{EELS}}\;\rm{[a.u.]}$")
    #     ax.legend(loc='lower left', frameon=False)
    #     plt.yscale('log')
    #
    #     fig.tight_layout()
    #
    # fig.savefig(os.path.join(image.output_path, 'raw_InSe_smoothened_percde1_5.pdf'))

    #######

    # plot with location of dE1 shown on top of the slope of the raw (smoothened) spectrum
    fig_1, ax = plt.subplots(dpi=200)
    for i in range(len(y_smooth_clusters)):

        ci_low = np.nanpercentile(dy_dx_clusters[i], 16, axis=0)
        ci_high = np.nanpercentile(dy_dx_clusters[i], 84, axis=0)
        #lab = r'$\rm{Cluster\;%s}$' % i
        if i == 0:
            lab = r'$\rm{Vacuum}$'
        else:
            lab = r'$\rm{Cluster\;%s}$' % i
        plt.fill_between(der_deltaE, ci_low, ci_high, color=colors[i], alpha=0.2, label=lab)
        if prob:
            plt.vlines(de1_prob[i], -3E3, 2E3, ls='dashdot', color=colors[i])
        else:
            plt.vlines(de1_shift[i], -3E3, 2E3, ls='dashdot', color=colors[i])

    plt.plot([der_deltaE[0], der_deltaE[-1]], [0, 0], color='black')
    plt.title(r"$\rm{WS_2\;nanoflower-Slope\;of\;EELS\;spectrum\;per\;cluster}$")
    plt.xlabel(r"$\rm{Energy\;loss}\;$" + r"$\rm{[eV]}$")
    plt.ylabel(r"$dI/d\Delta E$" + r"$\rm{\;[a.u.]}$")
    plt.legend(loc='lower right', frameon=True)
    #plt.xlim(np.min(min_clusters) / 4, np.max(min_clusters) * 2)
    plt.xlim(0.4, 5)
    plt.ylim(-500, 500)
    #TODO: make path more general

    #fig.savefig(os.path.join(image.output_path, 'eels_der_de1_InSe.pdf'))


    # plot with location of dE1 shown on top of raw (smoothened) spectrum
    fig_2, ax = plt.subplots(dpi=200)
    for i in range(len(y_smooth_clusters)):

        # dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        ci_low = np.nanpercentile(y_smooth_clusters[i], 16, axis=0)
        ci_high = np.nanpercentile(y_smooth_clusters[i], 84, axis=0)
        plt.fill_between(image.deltaE, ci_low, ci_high, color=colors[i], alpha=0.2)
        if prob:
            plt.vlines(de1_prob[i], 0, 3e4, ls='dashdot', color=colors[i])
        else:
            plt.vlines(de1_shift[i], 0, 3e4, ls='dotted', color=colors[i])
        if i == 0:
            lab = r'$\rm{Vacuum}$'
        else:
            lab = r'$\rm{Cluster\;%s}$' % i
        plt.plot(image.deltaE, np.average(y_smooth_clusters[i], axis=0), color=colors[i], label=lab)

    plt.plot([der_deltaE[0], der_deltaE[-1]], [0, 0], color='black')
    plt.title(r"$\rm{WS_2\;nanoflower-Position\;of\;}$" + r"$E_I\;$" + r"$\rm{per\;cluster}$")
    plt.xlabel(r"$\rm{Energy\;loss}\;$" + r"$\rm{[eV]}$")
    plt.ylabel(r"$\rm{Intensity\;[a.u.]\;}$")
    plt.legend(loc='upper right', frameon=True)
    plt.xlim(np.min(min_clusters) / 4, np.max(min_clusters) * 2)
    plt.ylim(1e1, 2e4)
    plt.xlim(0, 3.0)
    plt.yscale('log')
    ax.set_yticklabels([])
    #TODO: make path more general
    return fig_1, fig_2
    #fig.savefig(os.path.join(image.output_path, 'eels_de1_InSe.pdf'))