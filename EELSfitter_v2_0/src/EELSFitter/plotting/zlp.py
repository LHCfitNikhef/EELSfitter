import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from ..core.training import smooth_clusters

def plot_zlp_ntot(
        image,
        fontsize=13
):

    """
    Plot the trained ZLP including uncertainties for the cluster means.
    """

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': fontsize})
    rc('text', usetex=False)

    fig, ax = plt.subplots(dpi=200)
    ax.set_title(r"$\rm{WS_2\;nanoflower-Predicted\;ZLPs\;for\;cluster\;means}$")
    ax.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
    ax.set_ylabel(r"$\rm{Intensity\;[a.u]}$")

    cluster_means = image.clusters
    scaled_int = [image.scale_var_log_sum_I[0] * i + image.scale_var_log_sum_I[1] for i in cluster_means]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, cluster_mean_scaled in enumerate(scaled_int):
        zlps = image.calc_zlp_ntot(cluster_mean_scaled)

        low = np.nanpercentile(zlps, 16, axis=0)
        high = np.nanpercentile(zlps, 84, axis=0)
        median = np.nanpercentile(zlps, 50, axis=0)

        ax.fill_between(image.deltaE, low, high, alpha=0.3)
        label = r"$\rm{Cluster\;%d}$" % i if i != 0 else r"$\rm{Vacuum}$"
        ax.plot(image.deltaE, median, label=label, color=colors[i])

    ax.set_ylim(1, 1e3)
    ax.set_xlim(0.4, 6)
    ax.legend(frameon=False)
    plt.yscale('log')

    return fig


def plot_inel_pred(
        image,
        pixx,
        pixy,
        sig="pooled",
        title_specimen=""
):
    """
    Plots for ``pixx``, ``pixy``

    - the inelastic scattering distribution plus uncertainties
    - the ZLP plus uncertainties
    - the raw signal

    Parameters
    ----------
    pixx: int
        x-coordinate of the pixel.
    pixy: int
        y-coordinate of the pixel.
    sig: str, optional
        Type of signal, set to ``"pooled"`` by default.
    title_specimen: str, optional
        Name of specimen.
    """
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
    rc('text', usetex=False)

    dE1 = image.dE1[1, int(image.clustered[pixy, pixx])]
    dE2 = dE1 * 3

    signal = image.get_pixel_signal(pixy, pixx, signal=sig)

    ZLPs_gen = image.calc_zlps(pixy, pixx, signal=sig)
    low_gen = np.nanpercentile(ZLPs_gen, 16, axis=0)
    high_gen = np.nanpercentile(ZLPs_gen, 84, axis=0)
    mean_gen = np.nanpercentile(ZLPs_gen, 50, axis=0)

    ZLPs_match = image.calc_zlps_matched(pixy, pixx, signal=sig)
    low_match = np.nanpercentile(ZLPs_match, 16, axis=0)
    high_match = np.nanpercentile(ZLPs_match, 84, axis=0)
    mean_match = np.nanpercentile(ZLPs_match, 50, axis=0)

    fig, ax = plt.subplots(dpi=200)
    ax.set_title(title_specimen + r"$\rm{\;ZLP\;matching\;result\;at\;pixel[%d,%d]}$" % (pixx, pixy))
    ax.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
    ax.set_ylabel(r"$\rm{Intensity\;[a.u.]}$")
    ax.set_ylim(0, 600)
    ax.set_xlim(0, dE2 + 1)

    ax.plot(image.deltaE, signal, label=r"$\rm{Signal}$", color='black')
    ax.axvline(dE1, 0, 1, color='C3', linestyle='--')
    ax.axvline(dE2, 0, 1, color='C3', linestyle='--')
    #ax.fill_between(image.deltaE, low_gen, high_gen, alpha=0.2)
    #ax.plot(image.deltaE, mean_gen, label=r"$\rm{Model\;prediction\;}$" + "$I_{ZLP}$")
    ax.fill_between(image.deltaE, low_match, high_match, alpha=0.2, color="C0")
    ax.plot(image.deltaE, mean_match, label=r"$\rm{I_{ZLP}}$", color="C0")
    ax.fill_between(image.deltaE, signal - low_match, signal - high_match, alpha=0.2, color="C1")
    ax.plot(image.deltaE, signal - mean_match, label=r"$\rm{I_{inel}}$", color="C1")

    ax.legend(loc="lower right", frameon=False)

    # fig2, ax2 = plt.subplots(dpi=200)
    # ax2.plot(image.deltaE, signal, label=r"$\rm{I_{EELS}}$", color='black')
    # ax2.plot(image.deltaE, signal - mean_match, label=r"$\rm{I_{inel}}$")
    # ax2.fill_between(image.deltaE, low_match, high_match, alpha=0.3)
    # ax2.plot(image.deltaE, mean_match, label=r"$\rm{I_{ZLP}}}$")
    #
    # ax2.axvline(dE1, 0, 1, color='C3', linestyle='--')
    # ax2.axvline(dE2, 0, 1, color='C3', linestyle='--')
    # ax2.set_title(title_specimen + r"$\rm{\;ZLP\;at\;pixel[%d,%d]}$" % (pixx, pixy))
    # ax2.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
    # ax2.set_ylabel(r"$\rm{Intensity\;[a.u.]}$")
    # ax2.set_ylim(1e0, 1e7)
    # ax2.set_xlim(0, dE2 + 1)
    # ax2.legend(loc="lower right", frameon=False)
    # plt.yscale('log')

    return fig#, fig2


def plot_zlp_signal(image):
    """
    Plots for each cluster the raw signal and the trained ZLPs evaluated at the cluster mean.
    The 68% CI is also shown. The cluster mean lives in the space of integrated intensities.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object

    Returns
    -------
    fig: matplotlib.figure.Figure

    """
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
    rc('text', usetex=False)

    dE1 = image.dE1[1, :]
    dE2 = 3 + dE1 # TODO: make more general
    cluster_means = image.clusters

    n_clusters = len(dE1)
    ncols = 2
    nrows = n_clusters // ncols if n_clusters % ncols == 0 else n_clusters // ncols + 1
    wl1 = round(image.l / 20)

    fig = plt.figure(figsize=(ncols * 5, nrows * 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for cluster_idx in range(n_clusters):
        print(cluster_idx)
        ax = fig.add_subplot(nrows, ncols, cluster_idx + 1)

        raw_spectra = image.get_cluster_spectra(conf_interval=1, signal='EELS')
        spectra_smooth = smooth_clusters(image, raw_spectra, wl1)

        spectra_med = np.nanpercentile(spectra_smooth[cluster_idx], 50, axis=0)
        spectra_cl_high = np.nanpercentile(spectra_smooth[cluster_idx], 84, axis=0)
        spectra_cl_low = np.nanpercentile(spectra_smooth[cluster_idx], 16, axis=0)

        ax.fill_between(image.deltaE, spectra_cl_low, spectra_cl_high, alpha=0.3)
        label = r'$\rm{Signal}$'
        ax.plot(image.deltaE, spectra_med, label=label)
        ax.axvline(dE1[cluster_idx], 0, 1, color='C3', linestyle='--')
        ax.axvline(dE2[cluster_idx], 0, 1, color='C3', linestyle='--')

        cluster_mean = cluster_means[cluster_idx]
        scaled_int = image.scale_var_log_sum_I[0] * cluster_mean + image.scale_var_log_sum_I[1]
        zlp_cluster_mean = image.calc_zlp_ntot(scaled_int)
        zlp_low = np.nanpercentile(zlp_cluster_mean, 16, axis=0)
        zlp_high = np.nanpercentile(zlp_cluster_mean, 84, axis=0)
        zlp_median = np.nanpercentile(zlp_cluster_mean, 50, axis=0)
        label_zlp = r'$\rm{ZLP}$'

        ax.fill_between(image.deltaE, zlp_low, zlp_high, alpha=0.3)
        ax.plot(image.deltaE, zlp_median, label=label_zlp)
        ax.text(0.03, 0.30, r'$\rm{\log N_{\mathrm{tot}}=%.2f}$'%(scaled_int), horizontalalignment='left', verticalalignment = 'center', transform = ax.transAxes)
        title = r'$\rm{\;Cluster\;%d}$' % (cluster_idx)
        ax.set_title(title)
        ax.axes.yaxis.set_ticklabels([])

        ax.set_ylim(1, 1e3)
        ax.set_xlim(0, dE2[cluster_idx] + 1)
        ax.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
        ax.set_ylabel(r"$I_{\rm{EELS}}\;\rm{[a.u.]}$")
        ax.legend(loc='lower left', frameon=False)
        plt.yscale('log')

    fig.tight_layout()
    return fig