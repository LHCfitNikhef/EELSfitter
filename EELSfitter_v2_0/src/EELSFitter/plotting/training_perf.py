import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

def plot_cost_dist(cost_trains, cost_tests, cost_tests_std, title=" "):
    """

    Parameters
    ----------
    cost_trains: numpy.ndarray, shape=(M,)
        Cost values of the training set with M trained NNs
    cost_tests: numpy.ndarray, shape=(M,)
        Cost values of the test set with M trained NNs
    cost_tests_std: float
        68% CL of the test costs
    title: str, optional
        Title of the plot

    Returns
    -------
    fig: matplotlib.figure.Figure

    """
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 13})
    rc('text', usetex=False)
    fig, ax = plt.subplots(dpi=200)
    plt.hist(cost_trains, label=r'$\rm{Training}$', bins=40, range=(0, 5 * cost_tests_std), alpha=0.4)
    plt.hist(cost_tests, label=r'$\rm{Validation}$', bins=40, range=(0, 5 * cost_tests_std), alpha=0.4)
    plt.title(title)
    plt.xlabel(r'$\chi^2$')
    plt.legend(frameon=False, loc='upper right')
    plt.xlim((0, 0.4))
    return fig
