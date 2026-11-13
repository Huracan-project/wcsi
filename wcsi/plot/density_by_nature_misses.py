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


titles = ["Baroclinic", "Trough", "Mid-Level", "Other", "Vortex", "No Match"]
natures = ["BC", "Tr", "MV", "Ot", "Vo"]

bounds = [0, 1, 2, 5, 10, 20, 30, 40, 41]
norm = BoundaryNorm(boundaries=bounds, ncolors=256)


def main():
    if not Path("density_by_nature.nc").exists():
        table = pd.read_parquet("WCSI_summary_all.parquet")
        table = filters.year(table, 2007)

        hits, misses, false_alarms, invests = filters.categories(
            table, "WCSI", invests=True
        )
        tracks = huracanpy.load("ERA5.nc")
        tracks = tracks.hrcn.sel_id(np.unique(misses.id_era5))

        ibtracs = huracanpy.load("IBTrACS.nc")
        ibtracs = ibtracs.hrcn.sel_id(misses.id_ibtracs)
        ibtracs = ibtracs.isel(record=np.isin(ibtracs.nature, ["TS", "MX", "NR", "DS"]))

        df_ibtracs = ibtracs[["track_id", "lon", "lat", "time"]].to_dataframe()
        df_era5 = tracks[["track_id", "lon", "lat", "time", "nature"]].to_dataframe()

        merged = pd.merge(df_ibtracs, df_era5, on="time").drop_duplicates()
        merged["dist"] = (
            huracanpy.calc.distance(
                merged.lon_x,
                merged.lat_x,
                merged.lon_y,
                merged.lat_y,
                method="geod",
            )
            * 1e-3
        )
        merged = merged[merged.dist <= 444]

        densities = []
        for nature in natures:
            df = merged[merged.nature == nature]
            densities.append(huracanpy.calc.density(df.lon_x, df.lat_x))

        ibtracs = ibtracs.hrcn.sel_id(misses[misses.id_era5 == -1].id_ibtracs)
        densities.append(ibtracs.hrcn.get_density(bin_size=5))

        densities = xr.concat(densities, dim="subset")
        densities.to_netcdf("density_by_nature.nc")
    else:
        densities = xr.open_dataarray("density_by_nature.nc")

    # Density by four different categories
    fig, axes = plt.subplots(
        3,
        2,
        sharex="all",
        sharey="all",
        figsize=(8, 7),
        subplot_kw=dict(projection=EqualEarth()),
    )

    axes = axes.flatten()

    for n in range(6):
        d = densities.sel(subset=n)
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
        axes[n].set_title(f"{titles[n]} ({int(d.values.sum())})")
        axes[n].text(0.05, 1.05, f"({ascii_lowercase[n]})", transform=axes[n].transAxes)

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes((0.1, 0.05, 0.8, 0.05))
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Number of points")

    plt.savefig("density_by_nature.pdf")


if __name__ == "__main__":
    main()
