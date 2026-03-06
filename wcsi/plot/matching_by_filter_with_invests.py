import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


bar_kws = dict(
    width=0.4,
    edgecolor="k",
    align="edge",
)
text_kws = dict(
    ha="center",
    va="center",
)
colours = dict(
    hits="C0",
    weak_hits="white",
    misses="C7",
    false_alarms="C1",
    invests="C8",
    non_invests="C3",
)
colours["pod"] = colours["hits"]
colours["pod_weak"] = colours["weak_hits"]
colours["far"] = colours["false_alarms"]
colours["far_non_invest"] = colours["non_invests"]

default_labels = dict(
    hits="Hits",
    weak_hits="Hits (weak)",
    misses="Misses",
    false_alarms="False Alarms",
    invests="Invests",
    non_invests="False Alarms (non-Invests)",
    pod="POD",
    pod_weak="POD (weak)",
    far="FAR",
    far_non_invest="FAR (non-Invests)",
)


def main():
    summary = pd.read_parquet("WCSI_summary_all.parquet")
    fig, axes = _main(summary, start_year=1979, invests=False)
    fig.savefig("fig_matches_by_filtering_method.pdf")

    fig, axes = _main(summary, start_year=2007, invests=True)
    fig.savefig("fig_matches_by_filtering_method_superbt.pdf")


def _main(summary, start_year, invests=False):
    subsets = ["all", "H2017-nolat", "H2017", "WCS", "WCSI"]
    summary = summary[summary.storm_start.dt.year >= start_year]
    summary = summary[(summary.id_era5 != -1) | (summary.id_ibtracs != "")]

    # Create what looks like a two panel figure
    # The top 3 axes are the first panel, with the top two axes showing a break in the
    # y-scale for the "all" and "H2017-nolat" subsets
    # The 4th axis is to add whitespace
    # The 5th axis is the second panel
    fig, axes = plt.subplots(
        5,
        1,
        sharex=True,
        figsize=(8, 8),
        gridspec_kw=dict(height_ratios=[1, 1, 4, 1, 6]),
    )
    # Remove intermediate spines between the broken axes in the top panel
    for ax_label in [0, 1]:
        axes[ax_label].spines["bottom"].set_visible(False)
        axes[ax_label].set(xticks=[])
        # Don't use scientific notation for the large numbers in the broken panels
        axes[ax_label].ticklabel_format(style="plain")
    for ax_label in [1, 2]:
        axes[ax_label].spines["top"].set_visible(False)
    axes[3].set_axis_off()

    for m, subset in enumerate(subsets):
        print(subset)

        if subset == "all":
            table_ = summary
            total = summary.id_ibtracs.nunique() - 1
        else:
            table_ = summary[summary[subset]]

        # Get stats
        matches = table_[
            (table_.id_era5 != -1) & ~table_.weak_match & (table_.id_ibtracs != "")
        ]
        weak_matches = table_[table_.weak_match]
        weak_matches = weak_matches[
            ~np.isin(weak_matches.id_ibtracs, matches.id_ibtracs)
        ]

        # Total numbers
        hit = matches.id_ibtracs.nunique()
        weak_hit = weak_matches.id_ibtracs.nunique()
        miss = total - hit - weak_hit
        fa = table_.loc[
            (table_.id_era5 != -1) & (table_.id_ibtracs == ""), "id_era5"
        ].nunique()

        if invests:
            invest = table_.loc[table_.id_superbt != "", "id_era5"].nunique()
            fa = fa - invest

        # Scores
        pod = hit / total
        pod_weak = (hit + weak_hit) / total
        far = fa / (hit + fa)

        if invests:
            far_invests = (fa + invest) / (hit + fa + invest)

        if m == 0:
            labels = default_labels
        else:
            labels = {key: None for key in default_labels}

        # Hits and misses
        stacked_bar(
            axes[2],
            m,
            [("hits", hit), ("weak_hits", weak_hit), ("misses", miss)],
            labels,
        )

        axes[2].text(m + 0.2, (hit + weak_hit) / 2, f"{hit}\n", **text_kws)
        axes[2].text(
            m + 0.2, (hit + weak_hit) / 2, f"\n({weak_hit})", color="w", **text_kws
        )

        axes[2].text(m + 0.2, hit + weak_hit + miss / 2, str(miss), **text_kws)

        # False alarms (and invests)
        for ax_label in [0, 1, 2]:
            if invests:
                stacked_bar(
                    axes[ax_label],
                    m + 0.4,
                    [("non_invests", fa), ("invests", invest)],
                    labels,
                )
            else:
                axes[ax_label].bar(
                    m + 0.4,
                    fa,
                    color=colours["false_alarms"],
                    label=labels["false_alarms"],
                    **bar_kws,
                )

        if subset in ["all", "H2017-nolat"]:
            if subset == "all":
                ax_label = 0
            else:
                ax_label = 1

            if invests:
                ymin, ymax = fa - invest, fa + 1.25 * invest
                label_y = fa - invest * 0.5
            else:
                ymin, ymax = fa - fa / 20, fa + fa / 20
                label_y = fa - fa / 40
            axes[ax_label].set_ylim(ymin, ymax)

        else:
            ax_label = 2
            label_y = fa / 2

        axes[ax_label].text(m + 0.6, label_y, f"{fa}", **text_kws)

        if invests:
            axes[ax_label].text(m + 0.6, fa + invest / 2, f"{invest}", **text_kws)

        # Probability of detection
        stacked_bar(axes[4], m, [("pod", pod), ("pod_weak", pod_weak - pod)], labels)

        axes[4].text(m + 0.2, pod_weak / 2, f"{pod:.2f}\n", **text_kws)
        axes[4].text(
            m + 0.2,
            pod_weak / 2,
            f"\n({pod_weak:.2f})",
            color=colours["weak_hits"],
            **text_kws,
        )

        # False alarm rate (with/without invests)
        if invests:
            stacked_bar(
                axes[4],
                m + 0.4,
                [("far_non_invest", far), ("far", far_invests - far)],
                labels,
            )

            if m <= 1:
                y = far + 0.02
            else:
                y = far + (far_invests - far) / 2
            axes[4].text(
                m + 0.6,
                y,
                f"{far_invests:.2f}",
                **text_kws,
            )
        else:
            axes[4].bar(
                m + 0.4,
                far,
                color=colours["false_alarms"],
                label=labels["far"],
                **bar_kws,
            )
        axes[4].text(m + 0.6, far / 2, f"{far:.2f}", **text_kws)

        axes[4].text(m + 0.4, -0.05, subset, **text_kws)
        print(subset, pod, pod_weak, far)

    axes[2].legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.5))
    axes[4].legend(ncol=2, bbox_to_anchor=(0.5, 0.9))
    axes[2].set_ylabel("Number of tracks")
    axes[4].set_ylabel("Score")
    axes[2].set_xlim(0, len(subsets) - 0.1)
    axes[2].set_ylim(0, np.ceil(total / 1e3) * 1e3)
    axes[4].set_ylim(0, 1)

    axes[0].text(-0.05, 1.05, "(a)", transform=axes[0].transAxes)
    axes[4].text(-0.05, 1.05, "(b)", transform=axes[4].transAxes)
    fig.subplots_adjust(hspace=0.05)
    fig.suptitle(f"ERA5-IBTrACS Matches by Filtering Method ({start_year}-2024)")

    return fig, axes


def stacked_bar(ax, x, variables, labels):
    bottom = 0
    for variable, value in variables:
        ax.bar(
            x,
            value,
            bottom=bottom,
            color=colours[variable],
            label=labels[variable],
            **bar_kws,
        )
        bottom += value


if __name__ == "__main__":
    main()
