import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .era5_vs_jra3q import colors, half_window, category_bins


def main(matched_lmi):
    stats = rolling_window_stats(
        matched_lmi,
        half_window=half_window,
        datasets=["era5", "jra3q"],
        bins=category_bins,
    )

    fig, axes = plt.subplots(8, 4, figsize=(8, 8), width_ratios=[1, 0.1, 1, 1])
    fig.subplots_adjust(wspace=0.1)

    for ax in axes[0, :]:
        ax.set_axis_off()
    for ax in axes[:, 1]:
        ax.set_axis_off()

    for n, (dataset, color) in enumerate(
        [
            ("ERA5", colors[0]),
            ("JRA3Q", colors[1]),
        ]
    ):
        axes[0, 1].fill_betweenx(
            [np.nan, np.nan],
            np.nan,
            np.nan,
            color=color,
            label=dataset,
        )
    axes[0, 1].legend(ncol=2, bbox_to_anchor=[0, 1.25])

    axes = np.array([axes[1:, 0], axes[1:, 2], axes[1:, 3]]).T

    plot_rolling_window(stats, axes, bins=category_bins)

    axes[0, 0].set_title(r"$|\mathrm{B}|$")
    axes[0, 1].set_title(r"$\mathrm{V}_\mathrm{T}^\mathrm{L}$")
    axes[0, 2].set_title(r"$\mathrm{V}_\mathrm{T}^\mathrm{U}$")

    xmin = 1948 + half_window
    xmax = 2024 - half_window
    xticks = (
        [xmin]
        + list(
            range(int(np.ceil(xmin / 10) * 10), int(np.floor(xmax / 10) * 10) + 1, 10)
        )
        + [xmax]
    )

    for ax in axes.flatten():
        ax.spines[["top", "bottom"]].set_visible(False)
        ax.set_xticks([])
        ax.set_xlim(xmin, xmax)
        for x in xticks:
            ax.axvline(x, color="k", zorder=-1, linewidth=0.1, alpha=0.5)

    for ax in axes[:, 0]:
        ax.set_ylim(0, 15)
        ax.set_yticks([0, 15])
        for y in range(0, 15 + 1, 5):
            ax.axhline(y, color="k", zorder=-1, linewidth=0.1, alpha=0.5)

    for m, (ymin, ymax) in enumerate(
        [(50, 250), (50, 250), (0, 200), (0, 150), (0, 100), (0, 100), (-50, 100)]
    ):
        for ax in axes[m, 1:]:
            ax.set_ylim(ymin, ymax)

            if ymin < 0:
                ax.set_yticks([ymin, 0, ymax])
                ax.axhline(0, color="k", zorder=-1)
            else:
                ax.set_yticks([ymin, ymax])
            for y in range(ymin, ymax + 1, 50):
                ax.axhline(y, color="k", zorder=-1, linewidth=0.1, alpha=0.5)

        axes[m, 2].set_yticks([])

    for ax in axes[-1, :]:
        ax.set_xticks(xticks)
        ax.spines[["bottom"]].set_visible(True)

    axes[-1, 1].set_xlabel(f"Year ({2 * half_window + 1}-year rolling window)")

    for ax in axes[0, :]:
        ax.spines[["top"]].set_visible(True)

    axes[-1, 2].set_ylabel("")
    fig.text(0.925, 0.5, "Pressure category", rotation=270, va="center")
    fig.autofmt_xdate()

    plt.savefig("era5_vs_jra3q_structure.pdf")


def plot_rolling_window(stats_datasets, axes, bins):
    for m in range(len(bins) - 1):
        axes[m, -1].set_ylabel(f" {5 - m}", rotation="horizontal")
        axes[m, -1].yaxis.set_label_position("right")
        for i, (var, yspace) in enumerate([("b", 5), ("vtl", 25), ("vtu", 25)]):
            ax = axes[m, i]
            ymin = np.inf
            ymax = -np.inf
            for n, dataset in enumerate(stats_datasets):
                stats = stats_datasets[dataset]
                low_bound = stats[f"{var}_{m}"] - stats[f"confidence_{var}_{m}"]
                high_bound = stats[f"{var}_{m}"] + stats[f"confidence_{var}_{m}"]
                ax.fill_between(
                    stats.year,
                    low_bound,
                    high_bound,
                    color=colors[n],
                    alpha=0.5,
                )
                ax.plot(stats.year, stats[f"{var}_{m}"], color=colors[n])
                ymin = min(ymin, np.floor(low_bound.min() / yspace) * yspace)
                ymax = max(ymax, np.ceil(high_bound.max() / yspace) * yspace)


def rolling_window_stats(matched_lmi, half_window, datasets, bins):
    stats = {dataset: [] for dataset in datasets}
    for start_year in tqdm(range(1948, 2024 + 1 - 2 * half_window)):
        matched_lmi_ = matched_lmi[
            (matched_lmi.year >= start_year)
            & (matched_lmi.year <= start_year + 2 * half_window)
        ]

        for dataset in datasets:
            stats_window = dict(year=start_year + half_window)

            for m in range(len(bins) - 1):
                matched_lmi_category = matched_lmi_[
                    (matched_lmi_.mslp_ibtracs > bins[m])
                    & (matched_lmi_.mslp_ibtracs <= bins[m + 1])
                ]
                for var in ["b", "vtl", "vtu"]:
                    values = matched_lmi_category[f"cps_{var}_{dataset}"]
                    values = values[values != 1e25]
                    stats_window[f"{var}_{m}"] = np.mean(values)
                    stats_window[f"confidence_{var}_{m}"] = np.std(values) / np.sqrt(
                        len(values)
                    )

            stats[dataset].append(stats_window)

    return {dataset: pd.DataFrame(stats[dataset]) for dataset in datasets}


if __name__ == "__main__":
    summary = pd.read_parquet("WCSI_summary_all.parquet")
    matched_lmi = pd.read_parquet("matched_lmi.parquet")
    matched_lmi = matched_lmi[
        (matched_lmi.id_jra3q != -1) & (matched_lmi.id_era5 != -1)
    ]

    # Only take matched LMI where it is a "hit" for both ERA5 and JRA3Q
    ibtracs_ids = np.unique(summary[summary.WCSI & (summary.id_jra3q != -1)].id_ibtracs)
    matched_lmi = matched_lmi[np.isin(matched_lmi.id_ibtracs, ibtracs_ids)]

    main(matched_lmi)
