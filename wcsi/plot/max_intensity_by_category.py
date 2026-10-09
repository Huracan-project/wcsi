import huracanpy
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

from wcsi import wcsi
from wcsi.summary import filters

bounds = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 299])
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
        aaabbb
        aaabbb
        aaabbb
        yyyyyy
        ccddee
        ccddee
        ccddee
        zzzzzz
        111111
        """,
        figsize=(8, 8),
    )
    for ax in ["y", "z"]:
        axes[ax].set_axis_off()

    summary = pd.read_parquet("WCSI_summary_all.parquet")
    summary = filters.year(summary, 1979)

    matched_lmi = pd.read_parquet("matched_lmi.parquet")
    matched_lmi = matched_lmi[(matched_lmi.year >= 1979) & (matched_lmi.id_era5 != -1)]

    lmi_hits = matched_lmi[
        np.isin(matched_lmi.id_ibtracs, summary.id_ibtracs[summary.WCSI])
    ]
    lmi_miss = matched_lmi[
        ~np.isin(matched_lmi.id_ibtracs, summary.id_ibtracs[summary.WCSI])
    ]

    for ax, df in [("a", lmi_hits), ("b", lmi_miss)]:
        counts, _, _, _ = axes[ax].hist2d(
            df.mslp_ibtracs, df.mslp_era5, bins=bins, **hist2d_kwargs
        )
        axes[ax].plot(bins, bins, "--k")

        print(counts.max())

    tracks = huracanpy.load("ERA5.nc")
    for ax, subset in [
        ("c", "H2017"),
        ("d", "WCS"),
        ("e", "WCSI"),
    ]:
        fa = tracks.hrcn.sel_id(
            summary.loc[summary[subset] & (summary.id_ibtracs == ""), "id_era5"]
        )
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
        fa = fa.isel(record=fa[subset.lower()]).hrcn.get_apex_vals("mslp", stat="min")
        counts, _, _, im = axes[ax].hist2d(
            fa.mslp, np.abs(fa.lat), bins=[bins, bins_lon], **hist2d_kwargs
        )
        axes[ax].set_title(f"False Alarms {subset}")
        axes[ax].set_xlim(900, 1020)
        print(counts.max())

    for ax in ["b", "d", "e"]:
        axes[ax].set_yticks([])

    axes["a"].set_title("Hits WCSI")
    axes["b"].set_title("Misses WCSI")
    axes["a"].set_ylabel("MSLP ERA5 (hPa)")
    axes["c"].set_ylabel("|Latitude|")
    axes["d"].set_xlabel("MSLP ERA5 (hPa)")

    axes["y"].text(0.5, 0.5, "MSLP IBTrACS (hPa)", ha="center")

    plt.colorbar(im, cax=axes["1"], orientation="horizontal")

    for ax in ["a", "b", "c", "d", "e"]:
        axes[ax].text(0.0, 1.1, f"({ax})", transform=axes[ax].transAxes)

    plt.savefig("max_intensity_by_category.pdf")


if __name__ == "__main__":
    main()
