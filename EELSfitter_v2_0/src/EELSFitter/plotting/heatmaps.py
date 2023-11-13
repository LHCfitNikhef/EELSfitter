import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import warnings
import seaborn as sns
from matplotlib import cm

from ..core.spectral_image import round_scientific


def plot_heatmap(image, data, title=None, xlab='[nm]', ylab='[nm]', cmap='coolwarm',
                 discrete_colormap=False, sig_cbar=3, color_bin_size=None, equal_axis=True,
                 sig_ticks=2, npix_xtick=10, npix_ytick=10, scale_ticks=1, tick_int=False,
                 save_as=False, **kwargs):
    """
    Plots a heatmap for given data input.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object
    data : numpy.ndarray, shape=(M,)
        Input data for heatmap, but be 2 dimensional.
    title : str, optional
        Set the title of the heatmap. The default is None.
    xlab : str, optional
        Set the label of the x-axis. Nanometer ([nm]) is assumed as standard scale. The default is '[nm]'.
    ylab : str, optional
        Set the label of the y-axis. Nanometer ([nm]) is assumed as standard scale. The default is '[nm]'.
    cmap : str, optional
        Set the colormap of the heatmap. The default is 'coolwarm'.
    discrete_colormap : bool, optional
        Enables the heatmap values to be discretised. Best used in conjuction with color_bin_size. The default is False.
    sig_cbar : int, optional
        Set the amount of significant numbers displayed in the colorbar. The default is 3.
    color_bin_size : float, optional
        Set the size of the bins used for discretisation. Best used in conjuction discrete_colormap. The default is None.
    equal_axis : bool, optional
        Enables the pixels to look square or not. The default is True.
    sig_ticks : int, optional
        Set the amount of significant numbers displayed in the ticks. The default is 2.
    npix_xtick : float, optional
        Display a tick per n pixels in the x-axis. Note that this value can be a float. The default is 10.
    npix_ytick : float, optional
        Display a tick per n pixels in the y-axis. Note that this value can be a float. The default is 10.
    scale_ticks : float, optional
        Change the scaling of the numbers displayed in the ticks. Nanometer ([nm]) is assumed as standard scale adjust scaling from there. The default is 1.
    tick_int : bool, optional
        Set whether you only want the ticks to display as integers instead of floats. The default is False.
    save_as : str, optional
        Set the location and name for the heatmap to be saved to. The default is False.
    **kwargs : dictionary
        Additional keyword arguments.

    """

    fig = plt.figure(dpi=200)
    if title is None:
        if image.name is not None:
            plt.title(image.name)
    else:
        plt.title(title)

    if 'mask' in kwargs:
        mask = kwargs['mask']
        if mask.all():
            warnings.warn("Mask all True: no values to plot.")
            return
    else:
        mask = np.zeros(data.shape).astype('bool')

    if equal_axis:
        plt.axis('scaled')

    if discrete_colormap:

        unique_data_points = np.unique(data[~mask])
        if 'vmax' in kwargs:
            if len(unique_data_points[unique_data_points > kwargs['vmax']]) > 0:
                unique_data_points = unique_data_points[unique_data_points <= kwargs['vmax']]

        if 'vmin' in kwargs:
            if len(unique_data_points[unique_data_points < kwargs['vmin']]) > 0:
                unique_data_points = unique_data_points[unique_data_points >= kwargs['vmin']]

        if color_bin_size is None:
            if len(unique_data_points) == 1:
                color_bin_size = 1
            else:
                color_bin_size = np.nanpercentile(unique_data_points[1:] - unique_data_points[:-1], 30)

        n_colors = round((np.nanmax(unique_data_points) - np.nanmin(unique_data_points)) / color_bin_size + 1)
        cmap = cm.get_cmap(cmap, n_colors)
        spacing = color_bin_size / 2
        kwargs['vmax'] = np.max(unique_data_points) + spacing
        kwargs['vmin'] = np.min(unique_data_points) - spacing

    if image.pixelsize is not None:
        ax = sns.heatmap(data, cmap=cmap, **kwargs)
        xticks, yticks, xticks_labels, yticks_labels = image.get_ticks(sig_ticks, npix_xtick, npix_ytick, scale_ticks,
                                                                      tick_int)
        ax.xaxis.set_ticks(xticks)
        ax.yaxis.set_ticks(yticks)
        ax.set_xticklabels(xticks_labels, rotation=0)
        ax.set_yticklabels(yticks_labels)
    else:
        ax = sns.heatmap(data, **kwargs)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    colorbar = ax.collections[0].colorbar
    if discrete_colormap:
        if data.dtype == int:
            colorbar.set_ticks(np.unique(data[~mask]))
        else:
            colorbar.set_ticks(np.unique(data[~mask]))
            cbar_ticks_labels = []
            for tick in np.unique(data[~mask]):
                if tick >= 1:
                    cbar_ticks_labels.append(round_scientific(tick, sig_cbar + len(str(abs(int(math.floor(tick)))))))
                else:
                    cbar_ticks_labels.append(round_scientific(tick, sig_cbar))
            colorbar.ax.set_yticklabels(cbar_ticks_labels)

    if 'vmin' in kwargs:
        if np.nanmin(data[~mask]) < kwargs['vmin']:
            cbar_ticks = colorbar.ax.get_yticklabels()
            loc = 0
            if discrete_colormap:
                loc = np.min(np.argwhere(colorbar.ax.get_yticks() >= kwargs['vmin'] + spacing))
            cbar_ticks[loc] = r'$\leq$' + cbar_ticks[loc].get_text()
            colorbar.ax.set_yticklabels(cbar_ticks)

    if 'vmax' in kwargs:
        if np.nanmax(data[~mask]) > kwargs['vmax']:
            cbar_ticks = colorbar.ax.get_yticklabels()
            cbar_ticks_values = colorbar.ax.get_yticks()
            loc = -1
            if discrete_colormap:
                loc = np.max(np.argwhere(cbar_ticks_values <= kwargs['vmax'] - spacing))
            cbar_ticks[loc] = r'$\geq$' + cbar_ticks[loc].get_text()
            colorbar.ax.set_yticklabels(cbar_ticks)

    if save_as:
        if type(save_as) != str:
            if image.name is not None:
                save_as = image.name
        if 'mask' in kwargs:
            save_as += '_masked'
        save_as += '.pdf'
        plt.savefig(save_as, bbox_inches='tight')
    return fig