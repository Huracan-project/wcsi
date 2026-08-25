import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

subsets = [
    ("era5_H2017", "H2017"),
    ("era5_WCSI", "WCSI"),
    ("syclops", "SyCLoPS"),
    ("era5_WCS", "WCS"),
    ("jra3q", "WCSI (JRA3Q)"),
]


def main():
    fig1, axes = plt.subplots(1, 2, figsize=(8, 3), sharey="all", sharex="all")
    roebber_diagram(axes[0])
    roebber_diagram(axes[1])
    stats = pd.read_csv("WCSI_scores_1979-2024.csv")
    _main(stats, axes[0], marker="v")
    stats = pd.read_csv("WCSI_scores_invests_2007-2024.csv")
    _main(stats, axes[0], marker="s")

    l1 = axes[0].legend(
        handles=[
            Line2D([0], [0], color=f"C{n}", linestyle="", marker="o", label=label)
            for n, (subset, label) in enumerate(subsets)
        ],
        ncol=2,
        loc="lower left",
        handletextpad=0.0,
        labelspacing=0.25,
        columnspacing=0.5,
    )
    l2 = axes[0].legend(
        handles=[
            Line2D([0], [0], color="k", linestyle="", marker="v", label="1979-2024"),
            Line2D([0], [0], color="k", linestyle="", marker="s", label="2007-2024"),
            Line2D([0], [0], color="k", linestyle="--", marker="", label="Invests"),
        ],
        loc="upper right",
    )
    axes[0].add_artist(l1)

    subsets.remove(("era5_H2017", "H2017"))
    subsets.remove(("era5_WCS", "WCS"))

    for basin in ["NA", "WP", "EP", "NI", "SI", "SP"]:
        print(basin)
        fig, ax = plt.subplots(1, 1)
        roebber_diagram(ax)
        stats = pd.read_csv(f"WCSI_scores_{basin}_1979-2024.csv")
        _main(stats, ax, marker="v")
        stats = pd.read_csv(f"WCSI_scores_{basin}_invests_2007-2024.csv")
        _main(stats, ax, marker="s")

        l1 = ax.legend(
            handles=[
                Line2D([0], [0], color=f"C{n}", linestyle="", marker="o", label=label)
                for n, (subset, label) in enumerate(subsets)
            ],
            ncol=2,
            loc="lower left",
        )
        l2 = ax.legend(
            handles=[
                Line2D(
                    [0], [0], color="k", linestyle="", marker="v", label="1979-2024"
                ),
                Line2D(
                    [0], [0], color="k", linestyle="", marker="s", label="2007-2024"
                ),
                Line2D([0], [0], color="k", linestyle="--", marker="", label="Invests"),
            ],
            loc=[0.015, 0.2],
        )
        ax.add_artist(l1)

        fig.savefig(f"roebber_combined_{basin}.png")
        plt.close(fig)

    subsets.remove(("syclops", "SyCLoPS"))
    subsets.remove(("jra3q", "WCSI (JRA3Q)"))
    for n, basin in enumerate(["NA", "WP", "EP", "NI", "SI", "SP"]):
        print(basin)

        stats = pd.read_csv(f"WCSI_scores_{basin}_1979-2024.csv")
        _main(stats, axes[1], marker="v", color=f"C{n}")
        stats = pd.read_csv(f"WCSI_scores_{basin}_invests_2007-2024.csv")
        _main(stats, axes[1], marker="s", color=f"C{n}")

        axes[1].legend(
            handles=[
                Line2D([0], [0], color=f"C{n}", linestyle="", marker="o", label=basin)
                for n, basin in enumerate(["NA", "WP", "EP", "NI", "SI", "SP"])
            ],
            ncol=2,
            loc="lower left",
            handletextpad=0.0,
            labelspacing=0.25,
            columnspacing=0.5,
        )

    axes[0].set(xlabel="", title="Global stats by subset")
    axes[1].set(xlabel="", ylabel="", title="WCSI (ERA5) by Basin")
    fig1.text(0.52, 0.0, "1 - FAR", ha="center")
    fig1.savefig("roebber_2panel.png")
    plt.close(fig1)


def _main(stats, ax, marker="v", color=None):

    for n, (subset, name) in enumerate(subsets):
        if color is None:
            c = f"C{n}"
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
                marker=marker,
                mec="k",
                capsize=3,
            )
            ax.plot([1 - row.far, 1 - row.far_invest], [row.pod, row.pod], f"--{c}")

    ax.set(xlabel="1 - FAR", ylabel="POD", xlim=[0.5, 1], ylim=[0.5, 1])

    return


def roebber_diagram(ax):
    x = np.arange(0.01, 1.01, 0.01)
    x, y = np.meshgrid(x, x)
    csi = 1 / ((1 / x) + (1 / y) - 1)
    cs = ax.contourf(x, y, csi, np.arange(0.1, 1.1, 0.1), cmap="Grays_r")
    cs = ax.contour(x, y, csi, np.arange(0.1, 1.0, 0.1), colors="k")

    # Calculate points of CSI with equal POD and 1 - FAR to place labels
    p = [2 / ((1 / c) + 1) for c in np.arange(0.1, 1.0, 0.1)]
    ax.clabel(cs, manual=[(value, value) for value in p], colors="k")

    for y in [1 / 1.75, 1 / 1.5, 1 / 1.25, 1, 1.25, 1.5, 1.75]:
        ax.plot([0, 1], [0, y], "--k", alpha=0.5)


if __name__ == "__main__":
    main()
