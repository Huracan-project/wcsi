import huracanpy
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

from wcsi import wcsi
from wcsi.summary import filters

bounds = np.array([0, 1, 2, 5, 10, 20, 40, 60, 71])
norm = BoundaryNorm(boundaries=bounds, ncolors=256)
bins = np.arange(870, 1020 + 1, 5)
bins_lon = np.arange(0, 75 + 1, 5)

hist2d_kwargs = dict(
    cmap="cubehelix_r",
    norm=norm,
)


def main():
    fig, axes = plt.subplot_mosaic(
        """
        abc
        yyy
        def
        zzz
        111
        """,
        figsize=(8, 6),
        height_ratios=[4, 1, 4, 1, 1],
    )
    for ax in ["y", "z"]:
        axes[ax].set_axis_off()
    fig.subplots_adjust(wspace=0.1)

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

    matched_lmi = pd.read_parquet("matched_lmi_superbt.parquet")
    matched_lmi = matched_lmi[(matched_lmi.year >= 2007) & (matched_lmi.id_era5 != -1)]
    lmi_invest = matched_lmi[
        np.isin(matched_lmi.id_ibtracs, summary.id_superbt_era5[summary.WCSI])
    ]

    for ax, df in [("a", lmi_hits), ("b", lmi_miss), ("c", lmi_invest)]:
        counts, _, _, _ = axes[ax].hist2d(
            df.mslp_ibtracs, df.mslp_era5, bins=bins, **hist2d_kwargs
        )
        axes[ax].plot(bins, bins, "--k")

        print(counts.max())

    tracks = huracanpy.load("ERA5.nc")
    for ax, subset in [
        ("d", "H2017"),
        ("e", "WCS"),
        ("f", "WCSI"),
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
        counts, _, _, im = axes[ax].hist2d(
            fa.mslp, np.abs(fa.lat), bins=[bins, bins_lon], **hist2d_kwargs
        )
        axes[ax].set_title("FA$_\\mathrm{EI}$ " + f"{subset}")
        axes[ax].set_xlim(900, 1020)
        print(counts.max())

    for ax in ["b", "c", "e", "f"]:
        axes[ax].set_yticks([])

    axes["a"].set_title("Hits")
    axes["b"].set_title("Misses")
    axes["c"].set_title("Invests")

    axes["a"].set_xlabel("MSLP IBTrACS (hPa)")
    axes["b"].set_xlabel("MSLP IBTrACS (hPa)")
    axes["c"].set_xlabel("MSLP SuperBT (hPa)")
    axes["d"].set_xlabel("MSLP ERA5 (hPa)")
    axes["e"].set_xlabel("MSLP ERA5 (hPa)")
    axes["f"].set_xlabel("MSLP ERA5 (hPa)")

    axes["a"].set_ylabel("MSLP ERA5 (hPa)")
    axes["d"].set_ylabel("|Latitude|")

    plt.colorbar(im, cax=axes["1"], orientation="horizontal")

    for ax in ["a", "b", "c", "d", "e", "f"]:
        axes[ax].text(0.0, 1.05, f"({ax})", transform=axes[ax].transAxes)

    plt.savefig("max_intensity_by_category.pdf")


if __name__ == "__main__":
    main()
