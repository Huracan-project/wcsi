import huracanpy
import numpy as np
import pandas as pd
import xarray as xr


# Column names with dtype and fill value because pandas fills unknown values with NaN
# when merging
columns = {
    "id_ibtracs": (str, ""),
    "id_superbt_era5": (str, ""),
    "id_superbt_jra3q": (str, ""),
    "id_superbt_syclops": (str, ""),
    "id_era5": (int, -1),
    "id_syclops": (int, -1),
    "id_jra3q": (int, -1),
    "H2017-nolat": (bool, False),
    "H2017": (bool, False),
    "WCS": (bool, False),
    "WCSI": (bool, False),
    "temp": (int, 0),
    "ibtracs_nature": (str, ""),
    "ibtracs_points_6h": (int, 0),
    "ibtracs_dropped": (bool, False),
    "weak_match": (bool, False),
    "weak_match_jra3q": (bool, False),
    "weak_match_syclops": (bool, False),
}


def main():
    print("Combine WCSI")
    summary = combine_filters()
    summary.to_parquet("WCSI_summary.parquet")

    # Match ERA5 to IBTrACS
    print("Match ERA5 to IBTrACS")
    summary = match_ibtracs(summary)
    summary.to_parquet("WCSI-IBTrACS_summary.parquet")

    # Add SyCLoPS tracks
    print("Add SyCLoPS")
    summary = add_syclops(summary)
    summary.to_parquet("WCSI-IBTrACS-SyCLoPS_summary.parquet")

    # Add WCSI information for JRA3Q
    print("Add JRA3Q")
    summary = add_jra3q(summary)
    summary.to_parquet("WCSI-IBTrACS-SyCLoPS-JRA3Q_summary.parquet")

    # Look for matches between invests and ERA5 tracks that don't already have a match
    # in IBTrACS
    print("Match invests")
    summary = match_invests(summary)
    summary.to_parquet("WCSI_summary_all.parquet")


def combine_filters():
    # Easy to combine the details from all/WCS/WCSI because the track IDs are identical
    summary = pd.read_parquet("ERA5_WCS.parquet")
    summary_wcsi = pd.read_parquet("ERA5_WCSI.parquet")
    summary = summary.rename(columns=dict(is_tc="WCS"))
    summary["WCSI"] = np.zeros(len(summary), dtype=bool)
    summary.loc[summary_wcsi.track_id, "WCSI"] = summary_wcsi.is_tc.values

    # Could match up track ID original, but matching the origin of track is quick and
    # easy enough
    print("Matching genesis points")
    tracks = load_genesis_points("ERA5_all_nature.nc")
    tracks_tcident = load_genesis_points("ERA5_tcident.nc")
    tracks_nolat_tcident = load_genesis_points("ERA5_nolat-tcident.nc")

    for name, points in [
        ("H2017-nolat", tracks_nolat_tcident),
        ("H2017", tracks_tcident),
    ]:
        matches = huracanpy.assess.match([tracks, points], ["all", name], max_dist=0)
        summary[name] = np.isin(summary.track_id, matches.id_all)

    return summary


def load_genesis_points(filename):
    tracks = huracanpy.load(filename).hrcn.get_gen_vals()
    track_ids = tracks.track_id.values
    tracks = tracks.rename(track_id="record").drop_vars("record")

    return tracks.assign(track_id=("record", track_ids))


def match_ibtracs(summary):
    ibtracs_summary = pd.read_parquet("ibtracs_summary.parquet")
    ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    ibtracs_dropped = huracanpy.load("IBTrACS_1940-2024_dropped.nc")
    ibtracs = xr.concat([ibtracs, ibtracs_dropped], dim="record")
    tracks_era5 = huracanpy.load("ERA5_all_nature.nc")

    matching_summary = _match_ibtracs(
        tracks_era5, ibtracs, ibtracs_summary, label="era5"
    )

    summary = summary.rename(columns=dict(track_id="id_era5"))
    summary = summary.merge(matching_summary, on="id_era5", how="outer")

    fix_columns(summary)

    return summary


def _match_ibtracs(tracks, ibtracs, ibtracs_summary, label):
    # Within 1-degree for 1 day
    print("Matching IBTrACS")
    matches = huracanpy.assess.match(
        [tracks, ibtracs],
        [label, "ibtracs"],
        min_overlap=4,
        max_dist=165,
        consecutive_overlap=True,
        distance_method="geod",
    )

    # Allow single timestep matches for short IBTrACS tracks
    print("IBTrACS short tracks")
    ibtracs_short = ibtracs.hrcn.sel_id(
        ibtracs_summary.id_ibtracs[
            (ibtracs_summary.ibtracs_points_6h < 4) & (ibtracs_summary.id_ibtracs != "")
        ]
    )
    matches_short = huracanpy.assess.match(
        [tracks, ibtracs_short],
        [label, "ibtracs"],
        max_dist=165,
        distance_method="geod",
    )
    matches = pd.concat([matches, matches_short])

    # Try weak matches for remaining tracks
    print("IBTrACS weak matches")
    track_ids = np.unique(ibtracs.track_id)
    ibtracs_unmatched = ibtracs.hrcn.sel_id(
        track_ids[~np.isin(track_ids, matches.id_ibtracs)]
    )
    matches_weak = huracanpy.assess.match(
        [tracks, ibtracs_unmatched],
        [label, "ibtracs"],
        max_dist=None,
        mean_dist=440,
        distance_method="geod",
    )
    matches_weak["weak_match"] = True
    matches = pd.concat([matches, matches_weak])

    return matches.merge(ibtracs_summary, on="id_ibtracs", how="outer")


def fix_columns(df):
    for col in df.columns:
        if col in columns:
            dtype, fill_value = columns[col]
            try:
                nans = df[col].isna()
            except TypeError:
                nans = df[col].astype(str) == "nan"
            df.loc[nans, col] = fill_value
            df[col] = df[col].astype(dtype)


def add_jra3q(summary):
    # Only analyse WCSI subset of JRA3Q here
    tracks_jra3q = huracanpy.load("JRA3Q_nolat-nwc-tcident_WCSI_nature.nc")

    ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    ibtracs_dropped = huracanpy.load("IBTrACS_1940-2024_dropped.nc")
    ibtracs = xr.concat([ibtracs, ibtracs_dropped], dim="record")
    ibtracs_summary = pd.read_parquet("ibtracs_summary.parquet")

    return _add_other(summary, tracks_jra3q, "jra3q", ibtracs, ibtracs_summary)


def add_syclops(summary):
    syclops = huracanpy.load(
        "SyCLoPS_classified_ERA5_1940_2024_6hr.parquet", rename=dict(tid="track_id")
    )
    syclops = syclops.isel(record=syclops.track_info.str.contains("Track_TC"))

    ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    ibtracs_dropped = huracanpy.load("IBTrACS_1940-2024_dropped.nc")
    ibtracs = xr.concat([ibtracs, ibtracs_dropped], dim="record")
    ibtracs_summary = pd.read_parquet("ibtracs_summary.parquet")

    return _add_other(summary, syclops, "syclops", ibtracs, ibtracs_summary)


def _add_other(summary, tracks, label, ibtracs, ibtracs_summary):
    matching_summary = _match_ibtracs(
        tracks, ibtracs, ibtracs_summary, label=label
    ).rename(columns=dict(weak_match=f"weak_match_{label}"))[
        [f"id_{label}", "id_ibtracs", f"weak_match_{label}"]
    ]

    summary = summary.merge(matching_summary, on="id_ibtracs", how="outer")

    # Use strict matching for ERA5
    # Within 1-degree for 1 day
    # Only looking for remaining tracks that are WCSI for ERA5 and don't match IBTrACS
    tracks_era5 = huracanpy.load("ERA5_all_nature.nc")
    tracks_era5 = tracks_era5.hrcn.sel_id(
        summary.id_era5[summary.WCSI & (summary.id_ibtracs == "")]
    )

    track_ids = np.unique(tracks.track_id)
    track_ids = track_ids[~np.isin(track_ids, np.unique(summary[f"id_{label}"]))]
    tracks = tracks.hrcn.sel_id(track_ids)

    matches_reanalysis = huracanpy.assess.match(
        [tracks_era5, tracks],
        ["era5", "label"],
        min_overlap=4,
        max_dist=165,
        consecutive_overlap=True,
        distance_method="geod",
    )

    # In some cases one JRA3Q track matches multiple ERA5 tracks
    # The code below just ignores anything past the first because the counting in the
    # paper is based on the number of ERA5 tracks
    # Printing it out showed two occurences
    # id_era5 = 154518, id_jra3q = 47915, 47925
    # id_era5 = 187683, id_jra3q = 59808, 59802
    for n, rows in matches_reanalysis.groupby("id_era5"):
        index = summary.id_era5 == rows.iloc[0].id_era5
        summary.loc[index, "id_jra3q"] = rows.iloc[0].id_jra3q

    # Add info for JRA3Q tracks not included by any matching
    track_ids = track_ids[~np.isin(track_ids, np.unique(matches_reanalysis.id_jra3q))]
    for track_id in track_ids:
        idx = summary.index[-1] + 1
        track = tracks.hrcn.sel_id(track_id)
        summary.loc[idx, f"id_{label}"] = track_id
        summary.loc[idx, "storm_start"] = pd.to_datetime(track.time.values)[0]

    fix_columns(summary)

    return summary


def match_invests(summary):
    tracks_superbt = huracanpy.load("superbt.nc")
    nt_superbt = tracks_superbt[["time", "track_id"]].groupby("track_id").count()
    superbt_short = tracks_superbt.hrcn.sel_id(nt_superbt.track_id[nt_superbt.time < 4])

    for filename, label in [
        ("ERA5_all_nature.nc", "era5"),
        ("JRA3Q_nolat-nwc-tcident_WCSI_nature.nc", "jra3q"),
        ("SyCLoPS_classified_ERA5_1940_2024_6hr.parquet", "syclops"),
    ]:
        print(f"Matching invests to {label}")
        tracks = huracanpy.load(filename, rename=dict(tid="track_id"))

        # Only consider tracks that are not already matched with IBTrACS
        track_ids = np.unique(
            summary.loc[
                (summary.id_ibtracs == "") & (summary[f"id_{label}"] != -1),
                f"id_{label}",
            ]
        )
        tracks = tracks.hrcn.sel_id(track_ids)

        matches = huracanpy.assess.match(
            [tracks_superbt, tracks],
            ["superbt", label],
            min_overlap=4,
            max_dist=165,
            consecutive_overlap=True,
            distance_method="geod",
        )[[f"id_{label}", f"id_superbt_{label}"]]

        # Allow for shorter length invests
        matches_short = huracanpy.assess.match(
            [superbt_short, tracks],
            ["superbt", label],
            max_dist=165,
            distance_method="geod",
        )[[f"id_{label}", f"id_superbt_{label}"]]

        matches = pd.concat([matches, matches_short])

        # Only include one match per track to avoid expanding the table
        # Only important whether it matches at least one invest
        matches = matches.drop_duplicates(subset=f"id_{label}")

        summary = summary.merge(matches, on=f"id_{label}", how="outer")

    fix_columns(summary)

    return summary


if __name__ == "__main__":
    main()
