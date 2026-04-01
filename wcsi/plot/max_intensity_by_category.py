import huracanpy
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

from .. import filters, nature

bounds = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 199])
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

    tracks = huracanpy.load("ERA5_all.nc")
    for ax, subset, function in [
        ("c", "H2017", is_tc_h2017),
        ("d", "WCS", is_tc_wcs),
        ("e", "WCSI", is_tc_wcsi),
    ]:
        fa = tracks.hrcn.sel_id(
            summary.loc[summary[subset] & (summary.id_ibtracs == ""), "id_era5"]
        )
        fa = function(fa).hrcn.get_apex_vals("mslp", stat="min")
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


def is_tc_h2017(tracks):
    tracks, summary = nature.wcsi(
        tracks,
        npoints=4,
        vort_threshold=6,
        vort_warm_core_threshold=6,
        coherent=True,
        ocean=True,
        intensification_threshold=None,
    )

    return tracks.isel(record=np.where(tracks.is_tc)[0])


def is_tc_wcs(tracks):
    tracks, summary = nature.wcsi(
        tracks,
        npoints=4,
        b_threshold=15,
        vtl_threshold=0,
        vtu_threshold=0,
        vort_threshold=6,
        filter_size=5,
        intensification_threshold=None,
    )

    return tracks.isel(record=np.where(tracks.is_tc)[0])


def is_tc_wcsi(tracks):
    return tracks.isel(record=np.where(tracks.is_tc)[0])


if __name__ == "__main__":
    main()
