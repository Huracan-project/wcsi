import huracanpy
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    # Load a subset of variables from the full set of online IBTrACS data
    ibtracs = huracanpy.load(
        source="ibtracs",
        ibtracs_subset="ALL",
        usecols=[
            "SID",
            "ISO_TIME",
            "LON",
            "LAT",
            "BASIN",
            "NAME",
            "NATURE",
            "WMO_WIND",
            "WMO_PRES",
            "WMO_AGENCY",
        ],
    ).rename(dict(wmo_wind="wind", wmo_pres="mslp"))

    # Only include tracks that start and end within 1940-2024
    start_points = ibtracs.hrcn.get_gen_vals()
    end_points = ibtracs.hrcn.get_apex_vals("time", stat="max")
    track_ids = start_points.track_id[
        (start_points.time.dt.year >= 1940) & (end_points.time.dt.year <= 2024)
    ]
    ibtracs = ibtracs.hrcn.sel_id(track_ids)

    # Apply wind correction factors and add units
    correct_winds(ibtracs)
    ibtracs["wind"].attrs["units"] = "knots"
    ibtracs["wind"] = ibtracs.wind.metpy.convert_units("m s-1").metpy.dequantify()
    ibtracs["mslp"].attrs["units"] = "hPa"
    ibtracs = ibtracs.drop_vars("wmo_agency")

    ibtracs_summary = generate_summary(ibtracs)
    ibtracs_summary.to_parquet("ibtracs_summary.parquet")

    # Save 6-hourly data
    ibtracs.hrcn.save("IBTrACS.nc")

    # SuperBT
    # superbt = huracanpy.load(source="superbt").rename(
    #     tccode="nature", vmax="wind", pmin="mslp"
    # )
    # superbt["wind"] = (
    #     "record", superbt.wind.metpy.convert_units("m s-1").metpy.dequantify().values
    # )
    # superbt.hrcn.save("superbt.nc")


def correct_winds(ibtracs):
    # Apply correction factors to winds to 1-minutes sustained winds
    # From IBTrACS column documentation winds are
    # 1min for hurdat/atcf
    # 3min for newdelhi
    # 10min otherwise
    winds_3min = ibtracs.wmo_agency == "newdelhi"
    winds_10min = np.isin(
        ibtracs.wmo_agency, ["tokyo", "reunion", "bom", "nadi", "wellington"]
    )

    ibtracs["wind"][winds_3min] = ibtracs.wind[winds_3min] * 1.11 / 1.06
    ibtracs["wind"][winds_10min] = ibtracs.wind[winds_10min] * 1.11 / 1.03


def generate_summary(ibtracs):
    # Create a summary of tracks and why they may be dropped
    # Do they have tropical nature/are they shorter than 1 day
    track_ids = np.unique(ibtracs.track_id)

    track_nature = []
    nt_6h = []

    print("Generating summary")
    for track_id in tqdm(track_ids):
        track = ibtracs.hrcn.sel_id(track_id)

        if (track.nature == "TS").any():
            track_nature.append("TS")
        elif (track.nature == "MX").any():
            track_nature.append("MX")
        elif (track.nature == "NR").any():
            track_nature.append("NR")
        elif (track.nature == "SS").any():
            track_nature.append("SS")
        elif (track.nature == "ET").any():
            track_nature.append("ET")
        elif (track.nature == "DS").any():
            track_nature.append("DS")
        else:
            track_nature.append("")

        nt_6h.append(track.isel(record=track.time.dt.hour % 6 == 0).time.size)

    ibtracs_summary = pd.DataFrame(
        data=dict(
            id_ibtracs=track_ids,
            ibtracs_nature=track_nature,
            ibtracs_points_6h=nt_6h,
        )
    )

    return ibtracs_summary


def print_stats(ibtracs_summary):
    for year in [1979, 2007]:
        summary = ibtracs_summary[
            ibtracs_summary.id_ibtracs.str.slice(0, 4).astype(int) >= year
        ]
        dropped = summary[summary.ibtracs_dropped]
        isshort = dropped.ibtracs_points_6h < 4
        print("Since", year)
        print(len(dropped), "dropped")
        print(
            np.count_nonzero((dropped.ibtracs_nature == "TS") & isshort),
            "Short tropical storms",
        )
        print(
            np.count_nonzero((dropped.ibtracs_nature == "NR") & isshort),
            "Short with no recorded nature",
        )
        print(
            np.count_nonzero((dropped.ibtracs_nature == "") & isshort),
            "Short with other nature",
        )
        print(
            np.count_nonzero((dropped.ibtracs_nature == "NR") & ~isshort),
            "Long with no recorded nature",
        )
        print(
            np.count_nonzero((dropped.ibtracs_nature == "") & ~isshort),
            "Long with other recorded nature",
        )
        print(
            np.count_nonzero((dropped.ibtracs_nature == "TS") & ~isshort),
            "Long tropical storm",
        )


if __name__ == "__main__":
    main()
