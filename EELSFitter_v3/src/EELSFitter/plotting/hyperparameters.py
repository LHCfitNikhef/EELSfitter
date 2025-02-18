import numpy as np
import matplotlib.pyplot as plt


def plot_hp(eaxis, clusters_data, de1, de2, cmap='coolwarm', **kwargs):
    r"""
    Plot with location of dE1 & dE2 shown on top of the clusters.

    Parameters
    ----------
    eaxis : numpy.ndarray, shape=(M,)
        eaxis of the data
    clusters_data : numpy.ndarray, shape=(M,N)
        Data per cluster
    de1 : float
        Hyperparameter dE1
    de2 : float
        Hyperparameter dE2
    **kwargs : dictionary
        Additional keyword arguments.

    Returns
    -------
    fig: matplotlib.figure.Figure

    """

    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plot_figs(**kwargs)
    colors = eval('plt.cm.' + cmap + '(np.linspace(0, 1, len(clusters_data)))')
    ax.set_prop_cycle(color=colors)
    labels = []
    i = 0
    while i < len(clusters_data):
        labels.append(r"$\rm{Cluster\;%d}$" % i)
        ax.fill_between(eaxis[0:len(clusters_data[i][0])],
                        np.nanpercentile(clusters_data[i], 16, axis=0),
                        np.nanpercentile(clusters_data[i], 84, axis=0),
                        alpha=0.2, color=colors[i], label=labels[i])
        ax.plot(eaxis[0:len(clusters_data[i][0])],
                np.nanpercentile(clusters_data[i], 50, axis=0),
                alpha=0.2, color=colors[i])
        de1_idx = np.argwhere(eaxis < de1[i]).flatten()[-1]
        de2_idx = np.argwhere(eaxis < de2[i]).flatten()[-1]
        ax.vlines(x=de1[i], ymin=0, ymax=np.nanpercentile(clusters_data[i], 50, axis=0)[de1_idx],
                  ls='dashdot', color=colors[i])
        ax.vlines(x=de2[i], ymin=0, ymax=np.nanpercentile(clusters_data[i], 50, axis=0)[de2_idx],
                  ls='dotted', color=colors[i])
        i += 1
    if 'loc' in kwargs:
        ax.legend(loc=kwargs.get('loc'), frameon=False)
    return fig

def plot_figs(x=0, y=0, xlim=[0.4, 5], ylim=[-500, 500], yscale='linear', **kwargs):
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
