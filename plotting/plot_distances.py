import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def binned_scatter(
    x,
    y,
    n_per_bin=100,
    color="k",
    label=None,
    show_scatter=False,
    n_mask=None,
    ax=None,
    verbose = True
):
    """
    Plot binned averages of y vs x, equal-counts in each bins.
    Parameters
    x, y : distance, uncertainty
    n_per_bin : int, Number of points per bin.
    show_scatter : bool, Whether to plot raw scatter points.
    n_mask : int or None, Use only the first n_mask points after sorting.
    verbose : bool, Print correlation statistics.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if ax is None:
        ax = plt.gca()

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    if n_mask is not None:
        x_sorted = x_sorted[:n_mask]
        y_sorted = y_sorted[:n_mask]

    n_points = len(x_sorted)
    n_bins = max(1, n_points // n_per_bin)

    bin_edges = np.linspace(0, n_points, n_bins + 1, dtype=int)

    bin_centers = np.empty(n_bins)
    y_means = np.empty(n_bins)
    y_stds = np.empty(n_bins)

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        xb = x_sorted[start:end]
        yb = y_sorted[start:end]

        bin_centers[i] = np.median(xb)
        y_means[i] = np.mean(yb)
        y_stds[i] = np.std(yb)

    if verbose:
        corr_raw, _ = spearmanr(x, y)
        corr_binned, _ = spearmanr(bin_centers, y_means)
        pearson_raw, _ = pearsonr(x, y)
        pearson_binned, _ = pearsonr(bin_centers, y_means)
        print(
            f"n_per_bin={n_per_bin} | "
            f"raw Spearman={corr_raw:.3f} | "
            f"binned Spearman={corr_binned:.3f}"
            f"raw pearson={pearson_raw:.3f} | "
            f"binned pearson={pearson_binned:.3f}"
        )

    if show_scatter:
        ax.scatter(x, y, alpha=0.2, s=20, color=color, label="raw")

    ax.scatter(
        bin_centers,
        y_means,
        color=color,
        edgecolor="k",
        linewidth=1.0,
        s=40,
        label="binned average",
        zorder=3,
    )

    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Error (averaged per bin)")
    if label is not None:
        ax.set_title(label)

    ax.legend()
    plt.tight_layout()


def compare_models(measure: str, ensembles, model_names, binned: bool = True, thr: float = 0.2):
    '''
    Compare two models (MACE vs LATTE) for a given measure.
    '''
    all_configs_dict = get_model_dict(ensembles, model_names)
    x = np.array([v['MACE'][measure] for v in all_configs_dict.values()])
    y = np.array([v['LATTE'][measure] for v in all_configs_dict.values()])
    fig, axs = plt.subplots(1,1)    
    diff = x - y
    mask = abs(diff) > thr
    axs.scatter(x[~mask], y[~mask], color='r', s=20)
    axs.scatter(x[mask], y[mask], color='r', s=20, alpha=0.02)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    axs.plot(lims, lims, "k--", linewidth=1)
    if binned:
        binned_scatter(x, y)
    axs.set_xlabel(f'MACE {measure} ', fontsize = 16) 
    axs.set_ylabel(f'LATTE {measure} ', fontsize = 16)
    # plt.xlim(0.00001, 10)
    # plt.ylim(0.00001, 10)
    fig.suptitle(measure)


