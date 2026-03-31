import huracanpy
import numpy as np
import pandas as pd
import xarray as xr


# Column names with dtype and fill value because pandas fills unknown values with NaN
# when merging
columns = {
    "id_ibtracs": (str, ""),
    "id_superbt": (str, ""),
    "id_era5": (int, -1),
    "id_jra3q": (int, -1),
    "H2017-nolat": (bool, False),
    "H2017": (bool, False),
    "WCS": (bool, False),
    "WCSI": (bool, False),
    "WCSI_jra3q": (bool, False),
    "temp": (int, 0),
    "ibtracs_nature": (str, ""),
    "ibtracs_short": (bool, False),
    "weak_match": (bool, False),
    "weak_match_jra3q": (bool, False),
}


def main():
    print("Combine WCSI")
    summary = combine_filters()
    summary.to_parquet("WCSI_summary.parquet")

    # Match ERA5 to IBTrACS
    print("Match ERA5 to IBTrACS")
    summary = match_ibtracs(summary)
    summary.to_parquet("WCSI-IBTrACS_summary.parquet")

    # Add WCSI information for JRA3Q
    print("Add JRA3Q")
    summary = add_jra3q(summary)
    summary.to_parquet("WCSI-ERA5-IBTrACS-JRA3Q_summary")

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
    tracks = load_genesis_points("ERA5_all.nc")
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
    tracks_era5 = huracanpy.load("ERA5_all.nc")

    matching_summary = _match_ibtracs(
        tracks_era5, ibtracs, ibtracs_summary, label="era5"
    )

    summary = summary.rename(columns=dict(track_id="id_era5"))
    summary = summary.merge(matching_summary, on="id_era5", how="outer")

    fix_columns(summary)

    return summary


def _match_ibtracs(tracks, ibtracs, ibtracs_summary, label):
    # Within 1-degree for 1 day
    matches = huracanpy.assess.match(
        [tracks, ibtracs],
        [label, "ibtracs"],
        min_overlap=4,
        max_dist=165,
        consecutive_overlap=True,
        distance_method="geod",
    )

    # Allow single timestep matches for short IBTrACS tracks
    ibtracs_short = ibtracs.hrcn.sel_id(
        ibtracs_summary.id_ibtracs[ibtracs_summary.ibtracs_short]
    )
    matches_short = huracanpy.assess.match(
        [tracks, ibtracs_short],
        [label, "ibtracs"],
        max_dist=165,
        distance_method="geod",
    )
    matches = pd.concat([matches, matches_short])

    # Try weak matches for remaining tracks
    track_ids = np.unique(ibtracs.track_id)
    ibtracs_unmatched = ibtracs.hrcn.sel_id(
        track_ids[~np.isin(track_ids, matches.id_ibtracs)]
    )
    matches_weak = huracanpy.assess.match(
        [tracks, ibtracs],
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
    tracks_jra3q = huracanpy.load("JRA3Q_nolat-tcident.nc")
    summary_jra3q = pd.read_parquet("JRA3Q_nolat-tcident_WCSI.parquet")

    # Only analyse WCSI subset of JRA3Q here
    summary_jra3q = summary_jra3q[summary_jra3q.is_tc]
    tracks_jra3q = tracks_jra3q.hrcn.sel_id(summary_jra3q.track_id)

    ibtracs_summary = pd.read_parquet("ibtracs_summary.parquet")
    ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    ibtracs_dropped = huracanpy.load("IBTrACS_1940-2024_dropped.nc")
    ibtracs = xr.concat([ibtracs, ibtracs_dropped], dim="record")

    matching_summary = _match_ibtracs(
        tracks_jra3q, ibtracs, ibtracs_summary, label="jra3q"
    ).rename(columns=dict(weak_match="weak_match_jra3q"))[
        ["id_jra3q", "id_ibtracs", "weak_match_jra3q"]
    ]

    summary = summary.merge(matching_summary, on="id_ibtracs", how="outer")

    # Use strict matching between ERA5 and JRA3Q
    # Within 1-degree for 1 day
    # Only looking for remaining tracks that are WCSI for ERA5 and JRA3Q but don't
    # match IBTrACS
    tracks_era5 = huracanpy.load("ERA5_all.nc")
    tracks_era5 = tracks_era5.hrcn.sel_id(
        summary.id_era5[summary.WCSI & (summary.id_ibtracs == "")]
    )
    tracks_jra3q = tracks_jra3q.hrcn.sel_id(
        summary_jra3q.track_id[
            ~np.isin(summary_jra3q.track_id, matching_summary.id_jra3q)
        ]
    )

    matches_reanalysis = huracanpy.assess.match(
        [tracks_era5, tracks_jra3q],
        ["era5", "jra3q"],
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
    summary_jra3q = (
        summary_jra3q[~np.isin(summary_jra3q.track_id, np.unique(summary.id_jra3q))]
        .rename(columns=dict(track_id="id_jra3q"))
        .drop(columns=["is_tc"])
    )
    summary = pd.concat([summary, summary_jra3q])

    fix_columns(summary)

    return summary


def match_invests(summary):
    tracks_superbt = huracanpy.load("superbt.nc")
    tracks_era5 = huracanpy.load("ERA5_all.nc")

    # Smaller distance but for only one timestep minimum because invest tracks can be
    # short
    matches = huracanpy.assess.match(
        [tracks_superbt, tracks_era5],
        ["superbt", "era5"],
        max_dist=165,
        distance_method="geod",
    )[["id_era5", "id_superbt"]]

    summary = summary.merge(matches, on="id_era5", how="outer")
    fix_columns(summary)

    return summary


if __name__ == "__main__":
    main()
