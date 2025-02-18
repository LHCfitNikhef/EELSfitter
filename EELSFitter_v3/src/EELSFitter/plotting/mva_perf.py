import numpy as np
import matplotlib.pyplot as plt


def plot_pca_variance(components, eigenvalues_ratio, **kwargs):
    r"""

    Parameters
    ----------
    components
    eigenvalues_ratio
    kwargs

    Returns
    -------

    """
    components_list = np.arange(components) + 1
    eigenvalues_cumulative = np.cumsum(eigenvalues_ratio)
    fig, ax = plot_figs(**kwargs)
    axcolor = 'tab:blue'
    ax.plot(components_list, eigenvalues_ratio, marker='o', linestyle='--', linewidth=1,
            label=r"$\rm{Eigenvalue\;Ratio\;}$", color=axcolor)
    ax.set_xticks(components)
    ax.set_title(r"$\rm{Scree\;Plot\;}$")
    ax.set_xlabel(r"$\rm{Principal\;Component\;}$")
    ax.set_ylabel(r"$\rm{Variance\;Ratio\;}$", color=axcolor)
    ax.tick_params(axis='y', labelcolor=axcolor)

    ax2 = ax.twinx()
    ax2color = 'tab:red'
    ax2.plot(components_list, eigenvalues_cumulative, marker='o', linestyle='--', linewidth=1,
             label=r"$\rm{Eigenvalue\;Cumulative\;}$", color=ax2color)
    ax2.set_ylabel(r"$\rm{Variance\;Cumulative\;}$", color=ax2color)
    ax2.tick_params(axis='y', labelcolor=ax2color)
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
