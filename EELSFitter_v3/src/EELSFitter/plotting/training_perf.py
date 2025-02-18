import matplotlib.pyplot as plt


def plot_cost_dist(cost_trains, cost_tests, cost_tests_std, title=r"$\rm{Cost\;distribution\;}$"):
    r"""

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

    fig, ax = plt.subplots(dpi=500)
    ax.hist(cost_trains, bins=40, label=r"$\rm{Training}$", range=(0, 5 * cost_tests_std), alpha=0.4)
    ax.hist(cost_tests, bins=40, label=r"$\rm{Validation}$", range=(0, 5 * cost_tests_std), alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(r"$\rm{\chi^2\;}$")
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim((0, 0.4))
    return fig
