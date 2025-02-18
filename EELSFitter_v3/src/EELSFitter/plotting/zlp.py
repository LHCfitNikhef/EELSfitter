import numpy as np
import matplotlib.pyplot as plt

from ..core.training import smooth_signals_per_cluster


def summary_distribution(data, mean=50, lower=16, upper=84):
    median = np.nanpercentile(data, mean, axis=0)
    low = np.nanpercentile(data, lower, axis=0)
    high = np.nanpercentile(data, upper, axis=0)
    return [median, low, high]


def plot_zlp_cluster_predictions(image, cmap='coolwarm', **kwargs):
    r"""

    Parameters
    ----------
    image
    cmap
    kwargs

    Returns
    -------

    """

    fig, ax = plot_figs(**kwargs)
    colors = eval('plt.cm.' + cmap + '(np.linspace(0, 1, len(image.cluster_centroids)))')
    ax.set_prop_cycle(color=colors)
    labels = []
    top = 0
    for i, cluster_centroid in enumerate(image.cluster_centroids):
        labels.append(r"$\rm{Cluster\;%d}$" % i)
        zlps = summary_distribution(image.get_zlp_models(np.exp(cluster_centroid)))
        ax.fill_between(image.eaxis, zlps[1], zlps[2], alpha=0.1)
        ax.plot(image.eaxis, zlps[0], color=colors[i], label=labels[i])
        if np.max(zlps[2]) > top:
            top = np.max(zlps[2])
    ax.set_ylim([1, top])
    # ax.legend(loc=kwargs.get('loc'), frameon=False)
    return fig


def plot_zlp_per_pixel(image, pixx, pixy, signal_type='EELS', zlp_gen=False, zlp_match=True, subtract=False,
                       deconv=True, hyper_par=True, random_zlp=None, **kwargs):
    r"""
    Plots for ``pixx``, ``pixy``

    - the inelastic scattering distribution plus uncertainties
    - the ZLP plus uncertainties
    - the raw signal

    Parameters
    ----------
    image
    pixx
    pixy
    signal
    zlp_gen
    zlp_match
    subtract
    deconv
    hyper_par
    random_zlp
    kwargs

    Returns
    -------

    """

    fig, ax = plot_figs(**kwargs)
    signal = image.get_pixel_signal(i=pixy, j=pixx, signal_type=signal_type)
    # ax.plot(image.eaxis, signal, label=r"$\rm{I_{total}}$", color='black')
    step = 'mid'
    ax.step(x=image.eaxis, y=signal, where=step, label=r"$\rm{I_{total}}$", color='black')

    if (zlp_gen is True) or (random_zlp is not None):
        max_idx = np.argmax(signal)
        int_i = np.sum(signal[max_idx - 1:max_idx + 2])
        zlps_gen = image.get_zlp_models(int_i=int_i)
        if random_zlp is not None:
            if random_zlp >= len(zlps_gen):     # Plot all models
                for zlp_gen in zlps_gen:
                    ax.plot(image.eaxis, zlp_gen, color='tab:gray', alpha=0.1)
            else:   # Randomly select a model
                for k in range(random_zlp):
                    zlp_idx = np.random.randint(0, len(zlps_gen))
                    ax.plot(image.eaxis, zlps_gen[zlp_idx], color='tab:gray', alpha=0.1)
        if zlp_gen is True:
            zlps_gen_dist = summary_distribution(zlps_gen)
            ax.fill_between(image.eaxis, zlps_gen_dist[1], zlps_gen_dist[2], alpha=0.2, color='tab:red')
            ax.plot(image.eaxis, zlps_gen_dist[0], label=r"$\rm{I_{zlp_{model}}}$", color='tab:red')

    if (zlp_match is True) or (subtract is True):
        zlps_match = image.get_pixel_matched_zlp_models(i=pixy, j=pixx, signal_type=signal_type, signal=signal)
        if deconv is True:
            signal_extrp = image.extrp_signal(signal=signal)
            signal_ssds = np.zeros(zlps_match.shape)
            for k in range(zlps_match.shape[0]):
                zlp_k = zlps_match[k, :]
                signal_ssd_k = image.deconvolution(signal=signal_extrp, zlp=zlp_k).flatten()
                signal_ssds[k, :] = signal_ssd_k[:image.shape[2]]
            ssd = summary_distribution(signal_ssds)
            ax.fill_between(image.eaxis, ssd[1], ssd[2], step=step, alpha=0.2, color='tab:cyan')
            # ax.plot(image.eaxis, ssd[0], label=r"$\rm{I_{inel_{deconvoluted}}}$", color='tab:cyan')
            ax.step(x=image.eaxis, y=ssd[0], where=step, label=r"$\rm{I_{inel_{deconvoluted}}}$", color='tab:cyan')

        zlps_match_dist = summary_distribution(zlps_match)
        if zlp_match is True:
            ax.fill_between(image.eaxis, zlps_match_dist[1], zlps_match_dist[2], alpha=0.2, color='tab:orange')
            ax.plot(image.eaxis, zlps_match_dist[0], label=r"$\rm{I_{zlp_{matched}}}$", color='tab:orange')
        if subtract is True:
            ax.fill_between(image.eaxis, signal - zlps_match_dist[1], signal - zlps_match_dist[2],
                            step=step, alpha=0.2, color='tab:blue')
            # ax.plot(image.eaxis, signal - zlps_match_dist[0],
            #         label=r"$\rm{I_{inel_{subtracted}}}$", color='tab:blue')
            ax.step(x=image.eaxis, y=signal - zlps_match_dist[0],
                    where=step, label=r"$\rm{I_{inel_{subtracted}}}$", color='tab:blue')

    if hyper_par is True:
        de1 = image.dE1[int(image.cluster_labels[pixy, pixx])]
        de2 = image.dE2[int(image.cluster_labels[pixy, pixx])]
        fwhm = image.FWHM[int(image.cluster_labels[pixy, pixx])]
        de0 = (fwhm + de1) / 2
        ax.axvspan(de1, de2, alpha=0.1, color='tab:olive')
        ax.axvline(de1, color='tab:olive', linestyle='--')
        ax.axvline(de2, color='tab:olive', linestyle='--')
        ax.axvline(de0, color='tab:brown', linestyle='--')

    ax.legend(frameon=False)
    return fig


def plot_zlp_per_cluster(image, cluster, signal_type='EELS', zlp_gen=True, hyper_par=True, smooth=False, **kwargs):
    r"""

    Parameters
    ----------
    image
    cluster
    signal_type
    zlp_gen
    hyper_par
    kwargs

    Returns
    -------

    """

    fig, ax = plot_figs(**kwargs)
    cluster_signals = image.get_cluster_signals(signal_type=signal_type)[cluster]
    if smooth is True:
        cluster_signals = smooth_signals_per_cluster(cluster_signals)
    cluster_signals_dist = summary_distribution(cluster_signals, lower=1, upper=99)
    ax.fill_between(image.eaxis, cluster_signals_dist[1], cluster_signals_dist[2],
                    alpha=0.2, color='black', label=r"$\rm{I_{cluster}}$")

    if zlp_gen is True:
        max_idx = np.argmax(cluster_signals_dist[0])
        cluster_median = cluster_signals_dist[0]
        int_i = np.sum(cluster_median[max_idx - 1:max_idx + 2])
        zlp_dist = summary_distribution(image.get_zlp_models(int_i=int_i))
        ax.fill_between(image.eaxis, zlp_dist[1], zlp_dist[2],
                        alpha=0.2, color='tab:blue', label=r"$\rm{I_{zlp_{model}}}$")

    if hyper_par is True:
        de1 = image.dE1[cluster]
        de2 = image.dE2[cluster]
        fwhm = image.FWHM[cluster]
        de0 = (fwhm + de1) / 2
        ax.axvspan(de1, de2, alpha=0.1, color='tab:olive')
        ax.axvline(de1, color='tab:olive', linestyle='--')
        ax.axvline(de2, color='tab:olive', linestyle='--')
        ax.axvline(de0, color='tab:cyan', linestyle='--')

    ax.legend(frameon=False)
    return fig


def plot_figs(x=0, y=0, xlim=[0, 5], ylim=[-10, 3000], yscale='linear', **kwargs):
    r"""
    General parameters to plot figures

    Parameters
    ----------
    x
    y
    xlim
    ylim
    yscale
    kwargs

    Returns
    -------
    fig: matplotlib.figure.Figure

    """

    fig, ax = plt.subplots(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    ax.axhline(x, color='black')
    ax.axvline(y, color='black')
    ax.set_title(kwargs.get('title'))
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    ax.set_yticklabels([])
    return fig, ax
