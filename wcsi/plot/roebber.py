from string import ascii_lowercase

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

subsets = [
    ("era5_H2017", "H2017"),
    ("era5_WCS", "WCS"),
    ("syclops", "SyCLoPS"),
    ("era5_WCSI", "WCSI"),
    ("jra3q", "WCSI (JRA3Q)"),
]

colors = mpl.colormaps["okabe_ito"](np.linspace(0, 1, 8))[2:]


def main():
    # fig, axes = plt.subplots(
    #     4, 2, figsize=(8, 11), sharex="row", sharey="row", height_ratios=[1, 1, 1, 0.5]
    # )
    # fig.subplots_adjust(hspace=0.25)
    fig, axes = plt.subplot_mosaic(
        """
        lll
        lll
        abc
        abc
        abc
        abc
        abc
        abc
        def
        def
        def
        deg
        deg
        deg
        """,
        figsize=(8, 7),
    )
    # fig.subplots_adjust(hspace=0.0)
    axes = list(axes.values())
    for ax in axes[1:]:
        roebber_diagram(ax)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_axis_off()
    axes[1].tick_params(axis="y", labelleft=True)

    stats = pd.read_csv("WCSI_scores_1979-2024.csv")
    _main(stats, axes[1], marker="v")
    stats = pd.read_csv("WCSI_scores_invests_2007-2024.csv")
    _main(stats, axes[1], marker="s")
    axes[1].set_title("Global")

    l1 = axes[0].legend(
        handles=[
            Line2D([0], [0], color=colors[n], linestyle="", marker="o", label=label)
            for n, (subset, label) in enumerate(subsets)
        ],
        ncol=2,
        loc="upper left",
        handletextpad=0.0,
        labelspacing=0.25,
        columnspacing=0.5,
    )
    l2 = axes[0].legend(
        handles=[
            Line2D([0], [0], color="k", linestyle="", marker="v", label="1979-2024"),
            Line2D([0], [0], color="k", linestyle="", marker="s", label="2007-2024"),
            Line2D(
                [0], [0], color="k", linestyle="", marker="D", label="Excluding Invests"
            ),
        ],
        loc="upper right",
    )
    axes[0].add_artist(l1)

    for n, basin in enumerate(["WP", "EP", "NA", "NI", "SI", "SP"]):
        print(basin)
        ax = axes[n + 2]
        stats = pd.read_csv(f"WCSI_scores_{basin}_1979-2024.csv")
        _main(stats, ax, marker="v")
        stats = pd.read_csv(f"WCSI_scores_{basin}_invests_2007-2024.csv")
        _main(stats, ax, marker="s")

        ax.set_title(basin)

    for group, xlim, ylim, xticks, yticks in [
        ([1, 2, 3], (0.5, 1), (0.5, 1), 0.1, 0.1),
        ([4, 5], (0.4, 1), (0.4, 1), 0.2, 0.2),
        ([6, 7], (0.2, 1), (0.7, 1), 0.2, 0.1),
    ]:
        xticks = np.arange(xlim[0], xlim[1] + 0.01, xticks)
        yticks = np.arange(ylim[0], ylim[1] + 0.01, yticks)
        for n in group:
            axes[n].set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    for n in [1, 4]:
        axes[n].set(ylabel="POD")
    for n in [4, 5, 7]:
        axes[n].set(xlabel="1 - FAR")

    for n in [2, 3, 5]:
        axes[n].set_yticklabels([])
    axes[6].set_xticklabels([])

    for n in range(7):
        axes[n + 1].text(
            0, 1.05, f"({ascii_lowercase[n]})", transform=axes[n + 1].transAxes
        )

    fig.savefig("roebber_2panel.pdf")


def _main(stats, ax, marker="v", color=None):

    for n, (subset, name) in enumerate(subsets):
        if color is None:
            c = colors[n]
        else:
            c = color
        row = stats.loc[stats.subset == subset].iloc[0]
        ax.errorbar(
            1 - row.far,
            row.pod,
            xerr=[[row.far_high - row.far], [row.far - row.far_low]],
            yerr=[[row.pod - row.pod_low], [row.pod_high - row.pod]],
            color=c,
            marker=marker,
            mec="k",
            capsize=3,
        )

        if "far_invest" in row:
            ax.errorbar(
                1 - row.far_invest,
                row.pod,
                xerr=[
                    [row.far_invest_high - row.far_invest],
                    [row.far_invest - row.far_invest_low],
                ],
                yerr=[[row.pod - row.pod_low], [row.pod_high - row.pod]],
                color=c,
                marker="D",
                mec="k",
                capsize=3,
            )
            ax.plot(
                [1 - row.far, 1 - row.far_invest],
                [row.pod, row.pod],
                color=c,
                linewidth=5,
                alpha=0.4,
            )


def roebber_diagram(ax):
    x = np.arange(0.01, 1.01, 0.01)
    x, y = np.meshgrid(x, x)
    csi = 1 / ((1 / x) + (1 / y) - 1)
    cs = ax.contourf(x, y, csi, np.arange(0.0, 1.1, 0.1), cmap="Grays", vmin=0.15)
    cs = ax.contour(x, y, csi, np.arange(0.0, 1.0, 0.1), colors="k")

    # Calculate points of CSI with equal POD and 1 - FAR to place labels
    p = [2 / ((1 / c) + 1) for c in np.arange(0.1, 1.0, 0.1)]
    ax.clabel(cs, manual=[(value, value) for value in p], colors="k")

    bias = 2 ** np.arange(-5, 5, 0.25)
    for y in bias:
        ax.plot([0, 1], [0, y], "--k", alpha=0.5)


if __name__ == "__main__":
    main()
