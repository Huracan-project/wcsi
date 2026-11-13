import huracanpy
import numpy as np
import pandas as pd
from tqdm import tqdm


data_sources = [
    "usa",
    "tokyo",
    "cma",
    "hko",
    "kma",
    "newdelhi",
    "reunion",
    "bom",
    "nadi",
    "wellington",
    "ds824",
    "td9636",
    "td9635",
    "neumann",
    "mlc",
]
wind_vars = [f"{source}_wind" for source in data_sources]
mslp_vars = [f"{source}_pres" for source in data_sources]

usecols = (
    [
        "SID",
        "ISO_TIME",
        "LON",
        "LAT",
        "BASIN",
        "NAME",
        "NATURE",
        "WMO_AGENCY",
        "TRACK_TYPE",
    ]
    + [var.upper() for var in wind_vars]
    + [var.upper() for var in mslp_vars]
)


def main():
    # Load a subset of variables from the full set of online IBTrACS data
    ibtracs = huracanpy.load(
        filename="tmp.csv",
        source="ibtracs",
        ibtracs_subset="ALL",
        usecols=usecols,
    )

    # Only include tracks within 1940-2024
    year = ibtracs.track_id.str.slice(0, 4).astype(int)
    ibtracs = ibtracs.isel(record=(year >= 1940) & (year <= 2024))

    ibtracs_summary = generate_summary(ibtracs)
    print_stats(ibtracs_summary)

    # Only 6-hourly data
    ibtracs = ibtracs.isel(
        record=(ibtracs.time.dt.hour % 6 == 0) & (ibtracs.time.dt.minute == 0)
    )

    # Skip: Apply wind correction factors
    # correct_winds(ibtracs)
    # correct_mslp(ibtracs)
    # ibtracs = ibtracs.drop_vars(
    #     [var for var in ibtracs if "_wind" in var or "_pres" in var]
    # )
    ibtracs = ibtracs.drop_vars(["wmo_agency", "track_type"])

    # Take wind and pressure obs from agency with the most valid points for each track
    ibtracs, summary = fill_wind_pressure(ibtracs)

    ibtracs_summary = ibtracs_summary.merge(summary, on="id_ibtracs", how="outer")
    ibtracs_summary.to_parquet("ibtracs_summary.parquet")

    ibtracs.hrcn.save("IBTrACS.nc")

    # SuperBT
    superbt = huracanpy.load(source="superbt").rename(
        tccode="nature", vmax="wind", pmin="mslp"
    )
    superbt["wind"] = (
        "record",
        superbt.wind.metpy.convert_units("m s-1").metpy.dequantify().values,
    )
    superbt.hrcn.save("superbt.nc")


def fill_wind_pressure(ibtracs):
    _, idx, count = np.unique(ibtracs.track_id, return_index=True, return_counts=True)
    wind = np.full(len(ibtracs.time), np.nan)
    mslp = np.full(len(ibtracs.time), np.nan)

    summary = dict()

    track_ids = []
    npoints = dict(mslp=[], wind=[])
    agency = dict(mslp=[], wind=[])
    apex = dict(mslp=[], wind=[])

    for i, n in tqdm(zip(idx, count), total=len(idx)):
        ib = ibtracs.isel(record=slice(i, i + n))
        track_id = ib.track_id.values[0]
        track_ids.append(track_id)
        for x, varname, ibtracs_varnames in [
            (wind, "wind", wind_vars),
            (mslp, "mslp", mslp_vars),
        ]:
            lengths = [(~ib[var].isnull()).values.sum() for var in ibtracs_varnames]
            idx_maxlength = np.argmax(lengths)
            length = lengths[idx_maxlength]
            var = ibtracs_varnames[idx_maxlength]
            x[i : i + n] = ib[var].values

            npoints[varname].append(length)
            agency[varname].append(var.split("_")[0] if length > 0 else "")

            if length == 0:
                apex[varname].append(np.nan)
            elif varname == "mslp":
                apex["mslp"].append(np.nanmin(x[i : i + n]))
            elif varname == "wind":
                apex["wind"].append(np.nanmax(x[i : i + n]))

    ibtracs["wind"] = ("record", wind)
    ibtracs["mslp"] = ("record", mslp)

    ibtracs = ibtracs.drop_vars(wind_vars + mslp_vars)
    summary = pd.DataFrame(
        dict(
            id_ibtracs=track_ids,
            ibtracs_points_6h_wind=npoints["wind"],
            ibtracs_points_6h_mslp=npoints["mslp"],
            ibtracs_max_wind=apex["wind"],
            ibtracs_min_mslp=apex["mslp"],
            ibtracs_agency_wind=agency["wind"],
            ibtracs_agency_mslp=agency["mslp"],
        )
    )

    summary["ibtracs_category"] = huracanpy.tc.saffir_simpson_category(
        summary.ibtracs_max_wind.values, wind_units="kts", convention="1min"
    )

    return ibtracs, summary


def correct_winds(ibtracs):
    # Apply correction factors to winds to get 1-minutes sustained winds
    # From IBTrACS column documentation winds are
    # 1min for hurdat/atcf
    # 2min for CMA
    # 3min for newdelhi
    # 10min otherwise
    winds_3min = ibtracs.wmo_agency == "newdelhi"
    winds_10min = np.isin(
        ibtracs.wmo_agency, ["tokyo", "reunion", "bom", "nadi", "wellington"]
    )

    ibtracs["wind"][winds_3min] = ibtracs.wind[winds_3min] * 1.11 / 1.06
    ibtracs["wind"][winds_10min] = ibtracs.wind[winds_10min] * 1.11 / 1.03

    alternatives = [var for var in ibtracs if "_wind" in var]
    for var in alternatives:
        if "newdelhi" in var:
            factor = 1.11 / 1.06
        elif "cma" in var:
            factor = ""
        elif var.split("_")[0] in [
            "tokyo",
            "hko",
            "kma",
            "reunion",
            "bom",
            "nadi",
            "wellington",
        ]:
            factor = 1.11 / 1.03
        else:
            factor = 1

        missing_wind = ibtracs.wind.isnull()
        ibtracs["wind"][missing_wind] = ibtracs[var][missing_wind] * factor


def correct_mslp(ibtracs):
    alternatives = [var for var in ibtracs if "pres" in var]
    for var in alternatives:
        missing = ibtracs.mslp.isnull()
        ibtracs["mslp"][missing] = ibtracs[var][missing]


def generate_summary(ibtracs):
    # Create a summary of tracks and why they may be dropped
    # Do they have tropical nature/are they shorter than 1 day
    track_ids = np.unique(ibtracs.track_id)

    track_nature = []
    nt_6h = []
    track_type = []

    print("Generating summary")
    for track_id in tqdm(track_ids):
        track = ibtracs.hrcn.sel_id(track_id)

        if (track.nature == "TS").any():
            track_nature.append("TS")
        elif (track.nature == "MX").any():
            track_nature.append("MX")
        elif (track.nature == "NR").any():
            track_nature.append("NR")
        elif (track.nature == "DS").any():
            track_nature.append("DS")
        elif (track.nature == "SS").any():
            track_nature.append("SS")
        elif (track.nature == "ET").any():
            track_nature.append("ET")
        else:
            track_nature.append("")

        nt_6h.append(
            track.isel(
                record=(track.time.dt.hour % 6 == 0) & (track.time.dt.minute == 0)
            ).time.size
        )

        track_type.append(track.track_type.values[0])

    ibtracs_summary = pd.DataFrame(
        data=dict(
            id_ibtracs=track_ids,
            ibtracs_nature=track_nature,
            ibtracs_track_type=track_type,
            ibtracs_points_6h=nt_6h,
        )
    )

    return ibtracs_summary


def print_stats(ibtracs_summary):
    for year in [1979, 2007]:
        summary = ibtracs_summary[
            ibtracs_summary.id_ibtracs.str.slice(0, 4).astype(int) >= year
        ]
        isshort = summary.ibtracs_points_6h < 4
        print("Since ", year)
        for nature in np.unique(summary.ibtracs_nature):
            print(nature)
            print(
                np.count_nonzero((summary.ibtracs_nature == nature) & isshort),
                "Short",
            )
            print(
                np.count_nonzero((summary.ibtracs_nature == nature) & ~isshort),
                "Long",
                "\n",
            )


if __name__ == "__main__":
    main()
