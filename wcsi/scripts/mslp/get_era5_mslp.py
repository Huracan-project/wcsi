"""Download all ERA5 MSLP data and truncate to T6-63

Keep a running list of filenames to pass to tempest-extremes
"""

from pathlib import Path

import cdsapi
from cf_units import Unit
import iris
from iris.analysis import Linear
from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np
from windspharm.iris import VectorWind
import xarray as xr


spacing = 1

latitude = DimCoord(
    np.arange(-90, 90 + spacing, spacing),
    standard_name="latitude",
    units="degrees_north",
)
longitude = DimCoord(
    np.arange(-180, 180 + spacing, spacing),
    standard_name="longitude",
    units="degrees_east",
)

target = Cube(
    data=np.zeros([len(latitude.points), len(longitude.points)]),
    dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
)

unit = Unit("hours since 1940-01-01", calendar="proleptic_gregorian")


def main():
    vw = None
    fnames = []
    for year in range(1940, 2024 + 1):
        for month in range(1, 12 + 1):
            print(year, month)
            fname = f"mslp_era5_T6-63_{year}{month:02d}.nc"
            fnames.append(fname)
            if not Path(fname).exists():
                mslp = download_era5(year, month).msl
                mslp = mslp.rename(valid_time="time").to_iris().regrid(target, Linear())
                mslp.coord("time").convert_units(unit)

                if vw is None:
                    vw = VectorWind(mslp, mslp)

                mslp_t63 = vw.truncate(mslp, truncation=63)
                mslp_t5 = vw.truncate(mslp, truncation=5)
                mslp_t63 = mslp_t63 - mslp_t5
                mslp_t63.rename("mslp")

                iris.save(mslp_t63, fname)

    with open("in_data_list_mslp.txt", "w") as f:
        f.write("\n".join(fnames))


def download_era5(year, month):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["mean_sea_level_pressure"],
        "year": year,
        "month": month,
        "day": list(range(1, 31 + 1)),
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "grid": [1.0, 1.0],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download("tmp.nc")

    return xr.open_dataset("tmp.nc")


if __name__ == "__main__":
    main()
