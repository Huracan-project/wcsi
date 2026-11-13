import huracanpy
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

from wcsi import wcsi
from wcsi.summary import filters

bounds = np.array([0, 1, 2, 5, 10, 20, 40, 60, 80])
norm = BoundaryNorm(boundaries=bounds, ncolors=256)
bins = np.arange(870, 1020 + 1, 5)
bins_lat = np.arange(0, 75 + 1, 5)

hist2d_kwargs = dict(
    cmap="cubehelix_r",
    norm=norm,
)


def main():
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8, 6),
        sharex="row",
        sharey="row",
    )

    for ax in axes[0]:
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(
        left=0.1, right=0.97, top=0.95, bottom=0.19, wspace=0.35, hspace=0.4
    )
    colorbar_ax = fig.add_axes((0.1, 0.045, 0.87, 0.035))

    summary = pd.read_parquet("WCSI_summary_all.parquet")
    summary = filters.year(summary, 2007)

    matched_lmi = pd.read_parquet("matched_lmi.parquet")
    matched_lmi = matched_lmi[(matched_lmi.year >= 2007) & (matched_lmi.id_era5 != -1)]

    lmi_hits = matched_lmi[
        np.isin(matched_lmi.id_ibtracs, summary.id_ibtracs[summary.WCSI])
    ]
    lmi_miss = matched_lmi[
        ~np.isin(matched_lmi.id_ibtracs, summary.id_ibtracs[summary.WCSI])
    ]

    print("ERA5:", matched_lmi.mslp_era5.min(), matched_lmi.mslp_era5.max())
    print("IBTrACS:", matched_lmi.mslp_ibtracs.min(), matched_lmi.mslp_ibtracs.max())

    matched_lmi = pd.read_parquet("matched_lmi_superbt.parquet")
    matched_lmi = matched_lmi[(matched_lmi.year >= 2007) & (matched_lmi.id_era5 != -1)]
    lmi_invest = matched_lmi[
        np.isin(matched_lmi.id_ibtracs, summary.id_superbt_era5[summary.WCSI])
    ]

    print(
        "SuperBT:",
        lmi_invest[lmi_invest.mslp_ibtracs > 0].mslp_ibtracs.min(),
        lmi_invest.mslp_ibtracs.max(),
    )

    for ax, df in zip(axes[0], [lmi_hits, lmi_miss, lmi_invest]):
        counts, _, _, _ = ax.hist2d(
            df.mslp_ibtracs, df.mslp_era5, bins=bins, **hist2d_kwargs
        )
        ax.plot(bins, bins, "--k")

        print(counts.max())

    tracks = huracanpy.load("ERA5.nc")
    for ax, subset in [
        (axes[1, 0], "H2017"),
        (axes[1, 1], "WCS"),
        (axes[1, 2], "WCSI"),
    ]:
        _hits, _misses, false_alarms, _invests = filters.categories(
            summary, subset=subset, invests=True
        )
        fa = tracks.hrcn.sel_id(np.unique(false_alarms.id_era5))
        if subset == "H2017":
            fa["h2017"] = (
                "record",
                wcsi(
                    fa.track_id,
                    relative_vorticity=fa.relative_vorticity,
                    filter_all=fa.hrcn.get_is_ocean(),
                    filter_any=None,
                    npoints=4,
                    b_threshold=None,
                    vtl_threshold=None,
                    vtu_threshold=None,
                    vort_threshold=6,
                    vort_warm_core_threshold=6,
                    intensification_threshold=None,
                    coherent=True,
                    filter_size=None,
                ),
            )

        if subset == "WCSI":
            fa = fa.isel(record=fa.nature == "TC")
        else:
            fa = fa.isel(record=fa[subset.lower()])

        fa = fa.hrcn.get_apex_vals("mslp", stat="min")

        print(subset, fa.mslp.values.min(), fa.mslp.values.max())
        counts, _, _, im = ax.hist2d(
            fa.mslp, np.abs(fa.lat), bins=[bins, bins_lat], **hist2d_kwargs
        )
        ax.set_title("FA$_\\mathrm{EI}$ " + f"{subset}")
        print(counts.max())

    axes[0, 0].set_xlim(bins[0], bins[-1])
    axes[0, 0].set_xticks(range(bins[0], bins[-1] + 1, 30))
    axes[0, 0].set_ylim(bins[0], bins[-1])
    axes[0, 0].set_yticks(range(bins[0], bins[-1] + 1, 30))

    axes[1, 0].set_xlim(900, 1020)
    axes[1, 0].set_xticks(range(900, 1020 + 1, 30))
    axes[1, 0].set_ylim(bins_lat[0], bins_lat[-1])
    axes[1, 0].set_yticks([0, 15, 30, 45, 60, 75])

    axes[0, 0].set_title("Hits")
    axes[0, 1].set_title("Misses")
    axes[0, 2].set_title("Invests")

    axes[0, 0].set_xlabel("MSLP IBTrACS (hPa)")
    axes[0, 1].set_xlabel("MSLP IBTrACS (hPa)")
    axes[0, 2].set_xlabel("MSLP SuperBT (hPa)")
    axes[0, 0].set_ylabel("MSLP ERA5 (hPa)")

    for n in range(3):
        axes[1, n].set_xlabel("MSLP ERA5 (hPa)")
    axes[1, 0].set_ylabel("|Latitude|")

    for ax in axes.flat:
        ax.grid(color="k", linewidth=0.5, alpha=0.5)

    fig.colorbar(
        im, cax=colorbar_ax, orientation="horizontal", label="Number of tracks"
    )

    for label, ax in zip("abcdef", axes.flat):
        ax.text(0.0, 1.05, f"({label})", transform=ax.transAxes)

    plt.savefig("max_intensity_by_category.pdf")


if __name__ == "__main__":
    main()
