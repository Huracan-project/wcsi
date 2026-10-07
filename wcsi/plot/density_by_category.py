from pathlib import Path
from string import ascii_lowercase

from cartopy.crs import EqualEarth, PlateCarree
import huracanpy
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wcsi.summary import filters


titles = ["Hits", "Misses", "Invests", "False Alarms"]

bounds = np.arange(0, 0.15, 0.01)
norm = BoundaryNorm(boundaries=bounds, ncolors=256)


def main():
    table = pd.read_parquet("WCSI_summary_all.parquet")
    table = filters.year(table, 2007)

    hits, weak_hits, misses, false_alarms, invests = filters.categories(
        table, "WCSI", invests=True
    )

    track_ids = [np.unique(t.id_era5) for t in [hits, misses, invests, false_alarms]]
    ib_ids = np.unique(misses[misses.id_era5 == -1].id_ibtracs)

    if not Path("density_by_category.nc").exists():
        tracks = huracanpy.load("ERA5.nc")
        densities = get_densities(track_ids, tracks)

        ibtracs = xr.concat(
            [
                huracanpy.load("IBTrACS_1940-2024_dropped.nc"),
                huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc"),
            ],
            dim="record",
        )
        ibtracs = ibtracs.hrcn.sel_id(ib_ids)
        d = ibtracs.hrcn.get_density(bin_size=5, method="line")
        densities[1] += d

        densities.to_netcdf("density_by_category.nc")
    else:
        densities = xr.open_dataarray("density_by_category.nc")

    # Density by four different categories
    fig, axes = plt.subplots(
        2,
        2,
        sharex="all",
        sharey="all",
        figsize=(8, 5),
        subplot_kw=dict(projection=EqualEarth()),
    )

    axes = axes.flatten()

    for n in range(4):
        d = densities.sel(subset=n) / len(track_ids[n])
        print(d.max())

        im = axes[n].pcolormesh(
            d.lon,
            d.lat,
            d,
            cmap="cubehelix_r",
            norm=norm,
            transform=PlateCarree(),
        )
        axes[n].coastlines()
        axes[n].gridlines()
        axes[n].set_title(titles[n] + f"\n{len(track_ids[n])} tracks")
        axes[n].text(0.05, 0.95, f"({ascii_lowercase[n]})", transform=axes[n].transAxes)

    axes[1].set_title(titles[1] + f"\n{len(track_ids[1])} (+{len(ib_ids)}) tracks")

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes((0.1, 0.05, 0.8, 0.05))
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    fig.suptitle("ERA5 track density by category (2007-2024)")

    plt.savefig("fig_track_density_by_category.pdf")


def get_densities(subsets, tracks):
    return xr.concat(
        [
            tracks.hrcn.sel_id(subset).hrcn.get_density(bin_size=5, method="line")
            for subset in subsets
        ],
        dim="subset",
    )


if __name__ == "__main__":
    main()
