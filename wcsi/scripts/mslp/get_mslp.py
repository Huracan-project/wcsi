"""Download all ERA5 or JRA-3Q MSLP data and truncate to T6-63

Keep a running list of filenames to pass to tempest-extremes
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from urllib.request import build_opener

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

opener = build_opener()

jra3q_fname = (
    "https://osdf-director.osg-htc.org/ncar/gdex/d640000/anl_surf/{folder}/"
    "jra3q.anl_surf.0_3_1.prmsl-msl-an-gauss.{month_start}_{month_end}.nc"
)
jra3q_time_fmt = "%Y%m%d%H"


def main(dataset):
    vw = None
    fnames = []

    if dataset == "era5":
        start_year = 1940
    elif dataset == "jra3q":
        start_year = 1948
    else:
        raise ValueError(f"Dataset {dataset} not recognised")

    for year in range(start_year, 2024 + 1):
        for month in range(1, 12 + 1):
            print(year, month)
            fname = f"mslp_{dataset}_T6-63_{year}{month:02d}.nc"
            fnames.append(fname)
            if not Path(fname).exists():
                mslp = download_mslp(year, month, dataset)
                mslp, vw = truncate(mslp, low=6, high=63, vw=vw)
                iris.save(mslp, fname)

    with open("in_data_list_mslp.txt", "w") as f:
        f.write("\n".join(fnames))


def truncate(cube, low=6, high=63, vw=None):
    if vw is None:
        vw = VectorWind(cube, cube)

    high_pass = vw.truncate(cube, truncation=high)
    low_pass = vw.truncate(cube, truncation=low - 1)

    return high_pass - low_pass, vw


def download_mslp(year, month, dataset):
    if dataset == "era5":
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
        client.retrieve("reanalysis-era5-single-levels", request).download("tmp.nc")

        mslp = xr.open_dataset("tmp.nc").msl.rename(valid_time="time")

    elif dataset == "jra3q":
        time = datetime(year, month, 1)
        month_start = time.strftime(jra3q_time_fmt)
        time = time + relativedelta(months=1) - relativedelta(hours=6)
        month_end = time.strftime(jra3q_time_fmt)
        fname = jra3q_fname.format(
            folder=month_start[:6], month_start=month_start, month_end=month_end
        )
        infile = opener.open(fname)
        with open("tmp.nc", "wb") as outfile:
            outfile.write(infile.read())

        mslp = xr.open_dataset("tmp.nc")["prmsl-msl-an-gauss"]

    else:
        raise ValueError(f"Dataset {dataset} not recognised")

    mslp = mslp.to_iris().regrid(target, Linear())
    mslp.rename("mslp")

    if dataset == "era5":
        mslp.coord("time").convert_units(unit)

    return mslp


if __name__ == "__main__":
    for dataset in ["jra3q"]:
        main(dataset)
