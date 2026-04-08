import huracanpy
import numpy as np
import pandas as pd


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

    # Select for tropical storms/6-hourly data
    ibtracs_6h_ts, ibtracs_dropped, ibtracs_summary = drop_tracks(ibtracs)

    ibtracs_summary.to_parquet("ibtracs_summary.parquet")

    ibtracs_6h_ts.hrcn.save("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    ibtracs_dropped.hrcn.save("IBTrACS_1940-2024_dropped.nc")

    # SuperBT
    superbt = huracanpy.load(source="superbt").rename(
        tccode="nature", vmax="wind", pmin="mslp"
    )
    superbt["wind"] = (
        "record", superbt.wind.metpy.convert_units("m s-1").metpy.dequantify().values
    )
    superbt.hrcn.save("superbt.nc")


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


def drop_tracks(ibtracs):
    # Only include 6-hourly data
    ibtracs_6h = ibtracs.isel(record=np.where(ibtracs.time.dt.hour % 6 == 0)[0])

    # Only include storms that are labelled as tropical storm for at least one point
    ibtracs_6h_ts = ibtracs_6h.hrcn.trackswhere(
        lambda track: (track.nature == "TS").any() and track.time.size >= 4
    )

    # Account for tracks that have been removed by these criteria
    track_ids = np.unique(ibtracs.track_id)
    track_ids = track_ids[~np.isin(track_ids, np.unique(ibtracs_6h_ts.track_id))]
    ibtracs_dropped = ibtracs.hrcn.sel_id(track_ids)

    # Create a summary of tracks and why they are/aren't dropped
    # Do they have tropical nature/are they shorted than 1 day
    track_ids = np.unique(ibtracs_6h_ts.track_id)
    track_nature = np.full(len(track_ids), "TS", dtype="U2")
    isshort = np.zeros(len(track_ids), dtype=bool)

    track_ids_dropped = np.unique(ibtracs_dropped.track_id)
    track_nature_dropped = np.full(len(track_ids_dropped), "", dtype="U2")
    istc_dropped = np.array(
        [
            (track.nature == "TS").any()
            for track_id, track in ibtracs_dropped.groupby("track_id")
        ]
    )
    isnr_dropped = np.array(
        [
            (track.nature == "NR").all()
            for track_id, track in ibtracs_dropped.groupby("track_id")
        ]
    )
    track_nature_dropped[istc_dropped] = "TS"
    track_nature_dropped[isnr_dropped] = "NR"

    isshort_dropped = (
        (
            ibtracs_dropped.hrcn.get_apex_vals("time").time
            - ibtracs_dropped.hrcn.get_gen_vals().time
        )
        / np.timedelta64(1, "h")
    ) < 18

    ibtracs_summary = pd.DataFrame(
        data=dict(
            id_ibtracs=np.concatenate([track_ids, track_ids_dropped]),
            ibtracs_nature=np.concatenate([track_nature, track_nature_dropped]),
            ibtracs_short=np.concatenate([isshort, isshort_dropped]),
        )
    )

    return ibtracs_6h_ts, ibtracs_dropped, ibtracs_summary


def print_stats(ibtracs_summary):
    for year in [1979, 2007]:
        summary = ibtracs_summary[
            ibtracs_summary.ibtracs_id.str.slice(0, 4).astype(int) >= year
        ]
        dropped = summary[(summary.ibtracs_nature != "TS") | summary.ibtracs_short]
        print("Since", year)
        print(len(dropped), "dropped")
        print(
            np.count_nonzero((summary.ibtracs_nature == "TS") & summary.ibtracs_short),
            "Short tropical storms",
        )
        print(
            np.count_nonzero((summary.ibtracs_nature == "NR") & summary.ibtracs_short),
            "Short with no recorded nature",
        )
        print(
            np.count_nonzero((summary.ibtracs_nature == "") & summary.ibtracs_short),
            "Short with other nature",
        )
        print(
            np.count_nonzero((summary.ibtracs_nature == "NR") & ~summary.ibtracs_short),
            "Long with no recorded nature",
        )
        print(
            np.count_nonzero((summary.ibtracs_nature == "") & ~summary.ibtracs_short),
            "Long with other recorded nature",
        )


if __name__ == "__main__":
    main()
