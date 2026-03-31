from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import numpy as np
import pandas as pd
from scipy.stats import linregress

from .max_intensity_by_category import bins
from .matching_by_filter_with_invests import colours, default_labels


linestyles = [":", "-"]
bounds = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 246])
norm = BoundaryNorm(boundaries=bounds, ncolors=256)

cmap_kwargs = dict(
    cmap="cubehelix_r",
    norm=norm,
)


def main(summary):
    sets = []

    is_era5 = summary["H2017-nolat"] & summary["WCSI"]
    is_jra3q = summary.id_jra3q != -1
    is_ibtracs = summary.id_ibtracs != ""
    # (100, 010, 110, 001, 101, 011, 111)
    # 100 - Only ERA5
    sets.append(len(np.unique(summary[is_era5 & ~is_jra3q & ~is_ibtracs].id_era5)))

    # 010 - Only JRA3Q
    sets.append(len(np.unique(summary[~is_era5 & is_jra3q & ~is_ibtracs].id_jra3q)))

    # 110 - ERA5 and JRA3Q
    sets.append(len(np.unique(summary[is_era5 & is_jra3q & ~is_ibtracs].id_era5)))

    # 001 - Only IBTrACS
    sets.append(len(np.unique(summary[~is_era5 & ~is_jra3q & is_ibtracs].id_ibtracs)))

    # 101 - ERA5 and IBTrACS
    sets.append(len(np.unique(summary[is_era5 & ~is_jra3q & is_ibtracs].id_ibtracs)))

    # 011 - JRA3Q and IBTrACS
    sets.append(len(np.unique(summary[~is_era5 & is_jra3q & is_ibtracs].id_ibtracs)))

    # 111 - In all datasets
    sets.append(len(np.unique(summary[is_era5 & is_jra3q & is_ibtracs].id_ibtracs)))

    print(sum(sets))

    fig, axes = plt.subplot_mosaic(
        """
        xxxzbbb
        xxxzbbb
        aaazbbb
        aaazbbb
        aaazbbb
        aaazccc
        aaazccc
        aaazccc
        aaazccc
        yyyzccc
        yyyzwww
        yyyz111
        
        """,
        figsize=(8, 6),
    )

    for ax in ["w", "x", "y", "z"]:
        axes[ax].set_axis_off()

    v = venn3(
        sets,
        set_labels=["ERA5", "JRA3Q", "IBTrACS"],
        set_colors=["C0", "C1", "C2"],
        alpha=0.99,
        ax=axes["a"],
    )

    for subset in ["101", "011", "111"]:
        v.get_patch_by_id(subset).set(color=colours["hits"])

    for subset in ["100", "110", "010"]:
        v.get_patch_by_id(subset).set(color=colours["false_alarms"])

    for subset in ["001"]:
        v.get_patch_by_id(subset).set(color=colours["misses"])

    for subset in ["010", "100"]:
        v.get_patch_by_id(subset).set(alpha=0.5)

    for subset in ["101", "011"]:
        v.get_patch_by_id(subset).set(alpha=0.5)

    matched_lmi = pd.read_parquet("matched_lmi.parquet")
    matched_lmi = matched_lmi[
        (matched_lmi.id_jra3q != -1) & (matched_lmi.id_era5 != -1)
    ]

    x = matched_lmi.mslp_ibtracs
    for dataset, ax in [("era5", axes["b"]), ("jra3q", axes["c"])]:
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
            print(result)
            ax.plot(
                bins,
                result.slope * bins + result.intercept,
                linestyle=linestyles[n],
                color="C7",
                label=f"{label} ({result.slope:.3f} $\pm$ {result.stderr:.3f})",
            )

        ax.plot(bins, bins, "-k")
        ax.set(xlim=(bins[0], bins[-1]), ylim=(bins[0], bins[-1]))

    plt.colorbar(im, cax=axes["1"], orientation="horizontal")
    axes["b"].set_xticklabels([])
    axes["b"].set(xlabel="", ylabel="MSLP ERA5 (hPa)")
    axes["c"].set(xlabel="MSLP IBTrACS (hPa)", ylabel="MSLP JRA3Q (hPa)")
    axes["b"].legend()
    axes["c"].legend()

    axes["a"].set_title("Matching\n1979-2024")

    for ax in ["a", "b", "c"]:
        axes[ax].text(0.01, 0.925, f"({ax})", transform=axes[ax].transAxes)

    for category in ["hits", "false_alarms", "misses"]:
        axes["y"].fill_betweenx(
            [np.nan, np.nan],
            np.nan,
            np.nan,
            color=colours[category],
            label=default_labels[category],
        )
    axes["y"].legend(ncol=2, bbox_to_anchor=[1, 1])

    fig.suptitle("ERA5 vs JRA3Q")
    plt.savefig("era5_vs_jra3q.pdf")


if __name__ == "__main__":
    from .. import filters

    summary = pd.read_parquet("WCSI_summary_all.parquet")
    summary = filters.year(summary, 1979)
    main(summary)
