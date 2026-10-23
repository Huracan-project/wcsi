from string import ascii_lowercase

from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

from wcsi.scripts.statistics import get_stats
from wcsi.summary import filters

from .max_intensity_by_category import bins
from .roebber import colors

colors = colors[[3, 4]]
half_window = 5

# Klotzbach pressure categories
category_bins = np.asarray([-np.inf, 925, 945, 960, 975, 990, 1005, np.inf])


linestyles = [":", "-"]
bounds = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 246])
norm = BoundaryNorm(boundaries=bounds, ncolors=256)

cmap_kwargs = dict(
    cmap="cubehelix_r",
    norm=norm,
)


def main(summary, matched_lmi):
    matched_lmi = matched_lmi[
        (matched_lmi.id_jra3q != -1) & (matched_lmi.id_era5 != -1)
    ]
    stats = rolling_window_stats(
        summary,
        matched_lmi,
        half_window=5,
        datasets=["era5", "jra3q"],
        bins=category_bins,
    )

    summary = filters.year(summary, 2007)
    sets, invest_counts = get_venn_counts(summary)

    fig, axes = plt.subplots(
        9, 2, height_ratios=[1] + [0.1] + [0.2] * 7, figsize=(8, 8)
    )
    plt.subplots_adjust(hspace=0.25)

    gs = axes[0, 0].get_gridspec()
    axes[1, 1].set_axis_off()
    for ax in axes[:, 0]:
        ax.set_axis_off()

    ax_venn = fig.add_subplot(gs[:6, 0])
    ax_legend = fig.add_subplot(gs[6:, 0])
    ax_legend.set_axis_off()
    plot_venn(sets, invest_counts, ax_venn)
    ax_venn.set_title("Matching\n2007-2024")

    plot_rolling_window(stats, axes[:, 1], bins=category_bins)

    for n, ax in enumerate([ax_venn, axes[0, 1], axes[2, 1]]):
        ax.text(0.0, 1.1, f"({ascii_lowercase[n]})", transform=ax.transAxes)

    for n, (dataset, color) in enumerate(
        [
            ("IBTrACS", "k"),
            ("ERA5", colors[0]),
            ("JRA3Q", colors[1]),
        ]
    ):
        ax_legend.fill_betweenx(
            [np.nan, np.nan],
            np.nan,
            np.nan,
            color=color,
            label=dataset,
        )
    ax_legend.legend(loc="upper center")

    fig.text(
        0.475,
        0.4,
        "Average error at maximum intensity (hPa)",
        rotation="vertical",
        va="center",
    )
    fig.text(0.925, 0.4, "Pressure category", rotation=270, va="center")
    fig.autofmt_xdate()

    plt.savefig("era5_vs_jra3q.pdf")


def plot_venn(sets, invest_counts, ax):
    v = venn3(
        sets,
        set_labels=["ERA5", "JRA3Q", "IBTrACS"],
        set_colors=[colors[0], colors[1], "k"],
        alpha=0.75,
        ax=ax,
    )

    for n, subset in enumerate(["100", "010", "110"]):
        label = v.get_label_by_id(subset)
        label.set_text(f"{label.get_text()}\n({invest_counts[n]})")

    return v


def get_venn_counts(summary):
    sets = []

    is_era5 = summary["WCSI"]
    is_jra3q = summary.id_jra3q != -1
    is_ibtracs = summary.id_ibtracs != ""
    invest_counts = []
    # (100, 010, 110, 001, 101, 011, 111)
    # 100 - Only ERA5
    summary_ = summary[is_era5 & ~is_jra3q & ~is_ibtracs]
    sets.append(len(np.unique(summary_.id_era5)))
    invest_counts.append(summary_[summary_.id_superbt_era5 != ""].id_era5.nunique())

    # 010 - Only JRA3Q
    summary_ = summary[~is_era5 & is_jra3q & ~is_ibtracs]
    sets.append(len(np.unique(summary_.id_jra3q)))
    invest_counts.append(summary_[summary_.id_superbt_jra3q != ""].id_jra3q.nunique())

    # 110 - ERA5 and JRA3Q
    summary_ = summary[is_era5 & is_jra3q & ~is_ibtracs]
    sets.append(len(np.unique(summary_.id_era5)))
    invest_counts.append(summary_[summary_.id_superbt_era5 != ""].id_era5.nunique())

    # Avoid double counting on IBTrACS IDs due to extra rows in table for SyCLoPS
    ibtracs_ids = np.unique(summary.id_ibtracs[is_ibtracs])
    nibtracs = len(ibtracs_ids)

    # Succesively eliminate track IDs from IBTrACS IDs and count how many are eliminated
    # In all datasets
    hits = np.unique(summary[is_era5 & is_jra3q & is_ibtracs].id_ibtracs)
    nhits_all = len(hits)
    ibtracs_ids = ibtracs_ids[~np.isin(ibtracs_ids, hits)]

    # In ERA5
    hits = np.unique(summary[is_era5 & ~is_jra3q & is_ibtracs].id_ibtracs)
    hits = hits[np.isin(hits, ibtracs_ids)]
    nhits_era = len(hits)
    ibtracs_ids = ibtracs_ids[~np.isin(ibtracs_ids, hits)]

    # In JRA3Q
    hits = np.unique(summary[~is_era5 & is_jra3q & is_ibtracs].id_ibtracs)
    hits = hits[np.isin(hits, ibtracs_ids)]
    nhits_jra = len(hits)
    ibtracs_ids = ibtracs_ids[~np.isin(ibtracs_ids, hits)]

    # Only IBTrACS
    # Run code and add up numbers as a test this is working rather than just taking
    # the difference from other totals
    hits = np.unique(summary[~is_era5 & ~is_jra3q & is_ibtracs].id_ibtracs)
    hits = hits[np.isin(hits, ibtracs_ids)]
    nhits_ib = len(hits)
    ibtracs_ids = ibtracs_ids[~np.isin(ibtracs_ids, hits)]

    assert nibtracs == nhits_all + nhits_era + nhits_jra + nhits_ib
    assert len(ibtracs_ids) == 0

    # 001 - Only IBTrACS
    sets.append(nhits_ib)

    # 101 - ERA5 and IBTrACS
    sets.append(nhits_era)

    # 011 - JRA3Q and IBTrACS
    sets.append(nhits_jra)

    # 111 - In all datasets
    sets.append(nhits_all)

    return sets, invest_counts


def plot_rolling_window(stats_datasets, axes, bins):
    for n, datasets in enumerate(stats_datasets):
        stats = stats_datasets[datasets]
        for stat, linestyle in [
            ("csi", "-"),
            ("pod", "--"),
            ("far", ":"),
            ("csi_invest", "-"),
            ("far_invest", ":"),
        ]:
            axes[0].fill_between(
                stats.year,
                stats[f"{stat}_low"],
                stats[f"{stat}_high"],
                color=colors[n],
                alpha=0.5,
            )
            axes[0].plot(stats.year, stats[stat], color=colors[n], linestyle=linestyle)

    for m in range(len(bins) - 1):
        ax = axes[m + 2]
        ymin = np.inf
        ymax = -np.inf
        for n, dataset in enumerate(stats_datasets):
            stats = stats_datasets[dataset]
            low_bound = stats[f"error_{m}"] - stats[f"confidence_{m}"]
            high_bound = stats[f"error_{m}"] + stats[f"confidence_{m}"]
            ax.fill_between(
                stats.year,
                low_bound,
                high_bound,
                color=colors[n],
                alpha=0.5,
            )
            ax.plot(stats.year, stats[f"error_{m}"], color=colors[n])
            ymin = min(ymin, np.floor(low_bound.min() / 5) * 5)
            ymax = max(ymax, np.ceil(high_bound.max() / 5) * 5)
        ax.set(ylim=[ymin, ymax], yticks=[ymin, ymax], xticks=[])
        ax.set_ylabel(f" {5 - m}", rotation="horizontal")
        ax.yaxis.set_label_position("right")
        ax.spines[["top", "bottom"]].set_visible(False)
        for y in range(int(ymin), int(ymax) + 1, 5):
            ax.axhline(y, color="k", zorder=-1, linewidth=0.1, alpha=0.5)

    handles = [
        Line2D([0, 0], [0, 0], color="k", linestyle=l, label=label)
        for l, label in [("-", "CSI"), ("--", "POD"), (":", "FAR")]
    ]
    axes[0].legend(
        handles=handles, loc="lower left", ncol=3, columnspacing=1.5, handletextpad=0.5
    )

    axes[0].set(
        xticklabels=[],
        ylim=(0, 1),
        ylabel="Score",
        title="Skill",
    )
    xmin = 1948 + half_window
    xmax = 2024 - half_window
    xticks = (
        [xmin]
        + list(
            range(int(np.ceil(xmin / 10) * 10), int(np.floor(xmax / 10) * 10) + 1, 10)
        )
        + [xmax]
    )
    axes[-1].set(
        xlabel="Year (11-year rolling window)",
        xticks=xticks,
        ylabel="",
    )
    axes[2].spines[["top"]].set_visible(True)
    axes[2].set_title("MSLP Error")
    axes[-1].spines[["bottom"]].set_visible(True)

    for ax in axes:
        ax.set_xlim(xmin, xmax)
    for x in xticks:
        axes[0].axvline(x, color="k", zorder=-1, linewidth=0.1, alpha=0.5)
        for ax in axes[2:]:
            ax.axvline(x, color="k", zorder=-1, linewidth=0.1, alpha=0.5)

    for y in np.arange(0, 1 + 0.2, 0.2):
        axes[0].axhline(y, color="k", zorder=-1, linewidth=0.1, alpha=0.5)


def rolling_window_stats(summary, matched_lmi, half_window, datasets, bins):
    stats = {dataset: [] for dataset in datasets}
    for start_year in tqdm(range(1948, 2024 + 1 - 2 * half_window)):
        if start_year >= 2007:
            invests = True
        else:
            invests = False

        summary_ = filters.year(
            summary, start_year=start_year, end_year=start_year + 2 * half_window
        )
        matched_lmi_ = matched_lmi[
            (matched_lmi.year >= start_year)
            & (matched_lmi.year <= start_year + 2 * half_window)
        ]

        total = summary_.id_ibtracs.nunique() - 1
        for dataset in datasets:
            if dataset == "era5":
                stats_window = get_stats(
                    summary_[summary_.WCSI], total, label=dataset, invests=invests
                )
            else:
                stats_window = get_stats(
                    summary_, total, label=dataset, invests=invests
                )
            stats_window["year"] = start_year + half_window

            for m in range(len(bins) - 1):
                matched_lmi_category = matched_lmi_[
                    (matched_lmi_.mslp_ibtracs > bins[m])
                    & (matched_lmi_.mslp_ibtracs <= bins[m + 1])
                ]
                errors = (
                    matched_lmi_category[f"mslp_{dataset}"]
                    - matched_lmi_category.mslp_ibtracs
                )
                stats_window[f"error_{m}"] = np.mean(errors)
                stats_window[f"confidence_{m}"] = np.std(errors) / np.sqrt(len(errors))

            stats[dataset].append(stats_window)

    return {dataset: pd.DataFrame(stats[dataset]) for dataset in datasets}


def plot_matched_lmi(matched_lmi, datasets, axes):
    x = matched_lmi.mslp_ibtracs
    for dataset, ax in zip(datasets, axes):
        y = matched_lmi[f"mslp_{dataset}"]

        counts, _, _ = np.histogram2d(x, y, bins=bins)
        print(counts.max())
        im = ax.pcolormesh(bins, bins, counts.transpose(), zorder=1, **cmap_kwargs)

        for n, (x_, y_, label) in enumerate(
            [
                (x[matched_lmi.year < 1979], y[matched_lmi.year < 1979], "1948-1978"),
                (x[matched_lmi.year >= 1979], y[matched_lmi.year >= 1979], "1979-2024"),
            ]
        ):
            result = linregress(x_, y_)
            ax.plot(
                bins,
                result.slope * bins + result.intercept,
                linestyle=linestyles[n],
                color="C7",
                label=f"{label} ({result.slope:.3f} $\pm$ {result.stderr:.3f})",
            )

        ax.plot(bins, bins, "-k")
        ax.set(xlim=(bins[0], bins[-1]), ylim=(bins[0], bins[-1]))

    return im


if __name__ == "__main__":
    summary = pd.read_parquet("WCSI_summary_all.parquet")
    matched_lmi = pd.read_parquet("matched_lmi.parquet")

    main(summary, matched_lmi)
