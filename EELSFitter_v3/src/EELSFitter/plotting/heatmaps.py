import math
import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..core.spectral_image import round_scientific


def plot_heatmap(image, data, mask=None, window=None, vmin=None, vmax=None, cbar=True,
                 xlabel=r"$\rm{[nm]\;}$", ylabel=r"$\rm{[nm]\;}$", cbarlabel=r"$\rm{[-]\;}$", cmap='coolwarm',
                 discrete_colormap=False, sig_cbar=3, color_bin_size=None, robust=False, **kwargs):
    r"""
    Plots a heatmap for given data input.

    Parameters
    ----------
    image : SpectralImage
        :py:meth:`spectral_image.SpectralImage <EELSFitter.core.spectral_image.SpectralImage>` object
    data : numpy.ndarray, shape=(M,N)
        Input data for heatmap, must be 2D.
    mask : numpy.ndarray, shape=(M,N)
        Input data for the mask, must be same shape as `data`. Default is ``None``.
    window: numpy.ndarray, shape=(4,)
        Input the window you want to zoom in on. Shape as follows: [y_start, y_end, x_start, x_end].
    vmin, vmax : float, optional
        Set the lower (upper) bounds of the heatmap. Default is ``None``.
    xlabel, ylabel : str, optional
        Set the label of the x-axis (y-axis). Nanometer ([nm]) is assumed as standard scale. Default is '[nm]'.
    cbar : bool, optional
        Set if you want to have a colorbar or not. Default is ``True``.
    cbarlabel : str, optional
        Set the label of the colorbar. Default is '[-]'.
    cmap : str, optional
        Set the colormap of the heatmap. The default is 'coolwarm'.
    discrete_colormap : bool, optional
        Enables the heatmap values to be discretised. Best used in conjuction with color_bin_size. Default is ``False``.
    sig_cbar : int, optional
        Set the amount of significant numbers displayed in the colorbar. Default is 3.
    color_bin_size : float, optional
        Set the size of the bins used for discretisation. Best used in conjuction discrete_colormap. Default is ``None``.
    robust : bool, optional
        Enable if you want to remove outliers in the plotting. Plots the 99 percentile.
    **kwargs : dictionary
        Additional keyword arguments. These are passed to plt.subplots()

    Returns
    -------
    fig: matplotlib.figure.Figure

    """

    fig, ax = plt.subplots(figsize=kwargs.get('figsize'), dpi=kwargs.get('dpi'))
    ax.set_frame_on(False)
    ax.set_title(kwargs.get('title'))

    if mask is None:
        mask = np.zeros(data.shape).astype('bool')
    else:
        if mask.all():
            warnings.warn("Mask all True: no values to plot.")
            return

    if window is None:
        y_start = x_start = 0
        x_end = image.shape[1]
        y_end = image.shape[0]
    else:
        y_start, y_end, x_start, x_end = window
    data = data[y_start:y_end, x_start:x_end]
    mask = mask[y_start:y_end, x_start:x_end]

    # Create the discretization using the given discretized data
    if discrete_colormap:
        unique_data_points = np.unique(data[~mask])
        if (vmax is not None) and (len(unique_data_points[unique_data_points > vmax]) > 0):
            unique_data_points = unique_data_points[unique_data_points <= vmax]

        if (vmin is not None) and (len(unique_data_points[unique_data_points < vmin]) > 0):
            unique_data_points = unique_data_points[unique_data_points >= vmin]

        if color_bin_size is None:
            if len(unique_data_points) == 1:
                color_bin_size = 1
            else:
                color_bin_size = np.nanpercentile(unique_data_points[1:] - unique_data_points[:-1], 30)

        n_colors = round((np.nanmax(unique_data_points) - np.nanmin(unique_data_points)) / color_bin_size + 1)
        cmap = cm.get_cmap(cmap, n_colors)
        spacing = color_bin_size / 2
        vmax = np.max(unique_data_points) + spacing
        vmin = np.min(unique_data_points) - spacing

    # Set vmin and vmax for plotting to avoid high values
    if robust and (vmin is None):
        vmin_plt = np.nanpercentile(data[~mask], 1)
    else:
        vmin_plt = vmin
    if robust and (vmax is None):
        vmax_plt = np.nanpercentile(data[~mask], 99)
    else:
        vmax_plt = vmax

    # Creat the heatmap
    if image.pixel_size is not None:
        data_masked = np.ma.masked_where(mask, data)
        ext = [x_start*image.pixel_size[0], x_end*image.pixel_size[0], y_end*image.pixel_size[1], y_start*image.pixel_size[1]]
        hmap = ax.imshow(data_masked, extent=ext, cmap=cmap, vmin=vmin_plt, vmax=vmax_plt, interpolation='none')
    else:
        hmap = ax.imshow(data, cmap=cmap, vmin=vmin_plt, vmax=vmax_plt, interpolation='none')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if cbar is True:
        divider = make_axes_locatable(hmap.axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        colorbar = fig.colorbar(hmap, cax=cax, label=cbarlabel)
        colorbar.outline.set_visible(False)
        # Create the discretized colorbar from the discretized data
        if discrete_colormap:
            if data.dtype == int:
                colorbar.set_ticks(unique_data_points)
            else:
                colorbar.set_ticks(unique_data_points)
                cbar_ticks_labels = []
                for tick in unique_data_points:
                    if tick >= 1:
                        cbar_ticks_labels.append(round_scientific(tick, sig_cbar + len(str(abs(int(math.floor(tick)))))))
                    else:
                        cbar_ticks_labels.append(round_scientific(tick, sig_cbar))
                colorbar.ax.set_yticklabels(cbar_ticks_labels)

        # Adds equal or greater than symbol to max color value
        if vmax is not None:
            if np.nanmax(data[~mask]) > vmax:
                cbar_ticks = colorbar.ax.get_yticklabels()
                loc = -1
                if discrete_colormap:
                    loc = np.max(np.argwhere(colorbar.ax.get_yticks() <= vmax - spacing))
                cbar_ticks[loc] = r'$\geq$' + cbar_ticks[loc].get_text()
                colorbar.ax.set_yticklabels(cbar_ticks)

        # Adds equal or less than symbol to min color value
        if vmin is not None:
            if np.nanmin(data[~mask]) < vmin:
                cbar_ticks = colorbar.ax.get_yticklabels()
                loc = 0
                if discrete_colormap:
                    loc = np.min(np.argwhere(colorbar.ax.get_yticks() >= vmin + spacing))
                cbar_ticks[loc] = r'$\leq$' + cbar_ticks[loc].get_text()
                colorbar.ax.set_yticklabels(cbar_ticks)
    return fig
