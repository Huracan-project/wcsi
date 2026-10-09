import huracanpy
import numpy as np
import pandas as pd


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
}


def main():
    # Load datasets
    tracks_era5 = huracanpy.load("ERA5.nc")
    ibtracs = huracanpy.load("IBTrACS.nc")
    ibtracs_summary = pd.read_parquet("ibtracs_summary.parquet")

    syclops = huracanpy.load(
        "SyCLoPS_classified_ERA5_1940_2024_6hr.parquet", rename=dict(tid="track_id")
    )
    syclops = syclops.isel(record=syclops.track_info.str.contains("Track_TC"))

    # Only analyse WCSI subset of JRA3Q here
    tracks_jra3q = huracanpy.load("JRA3Q.nc")
    tracks_jra3q = tracks_jra3q.hrcn.sel_id(
        np.unique(tracks_jra3q.track_id[tracks_jra3q.wcsi])
    )

    tracks_superbt = huracanpy.load("superbt.nc")

    # Load summary for ERA5
    summary = pd.read_parquet("ERA5.parquet").rename(
        columns=dict(is_tc="WCSI", track_id="id_era5")
    )

    # Add H2017 and H2017-nolat
    summary = combine_filters(summary, tracks_era5)

    # Match ERA5 to IBTrACS
    print("Match ERA5 to IBTrACS")
    summary = match_ibtracs(summary, tracks_era5, ibtracs)
    summary = summary.merge(ibtracs_summary, on="id_ibtracs", how="outer")
    summary.to_parquet("WCSI-IBTrACS_summary.parquet")

    # Add SyCLoPS tracks
    print("Add SyCLoPS")
    summary = add_other(summary, tracks_era5, syclops, "syclops", ibtracs)
    summary.to_parquet("WCSI-IBTrACS-SyCLoPS_summary.parquet")

    # Add WCSI information for JRA3Q
    print("Add JRA3Q")
    summary = add_other(summary, tracks_era5, tracks_jra3q, "jra3q", ibtracs)
    summary.to_parquet("WCSI-IBTrACS-SyCLoPS-JRA3Q_summary.parquet")

    # Look for matches between invests and ERA5 tracks that don't already have a match
    # in IBTrACS
    print("Match invests")
    summary = match_invests(
        summary,
        tracks_superbt,
        dict(era5=tracks_era5, jra3q=tracks_jra3q, syclops=syclops),
    )

    summary.to_parquet("WCSI_summary_all.parquet")


def combine_filters(summary, tracks):
    # Could match up track ID original, but matching the origin of track is quick and
    # easy enough
    print("Matching genesis points")
    tracks_tcident = load_genesis_points("ERA5_tcident.nc")
    tracks_nolat_tcident = load_genesis_points("ERA5_nolat-tcident.nc")

    for name, points in [
        ("H2017-nolat", tracks_nolat_tcident),
        ("H2017", tracks_tcident),
    ]:
        matches = huracanpy.assess.match([tracks, points], ["all", name], max_dist=0)
        summary[name] = np.isin(summary.id_era5, matches.id_all)

    return summary


def load_genesis_points(filename):
    tracks = huracanpy.load(filename).hrcn.get_gen_vals()
    track_ids = tracks.track_id.values
    tracks = tracks.rename(track_id="record").drop_vars("record")

    return tracks.assign(track_id=("record", track_ids))


def match_ibtracs(summary, tracks_era5, ibtracs):
    matching_summary = _match_ibtracs(tracks_era5, ibtracs, label="era5")

    summary = summary.rename(columns=dict(track_id="id_era5"))
    summary = summary.merge(matching_summary, on="id_era5", how="outer")

    fix_columns(summary)

    return summary


def _match_ibtracs(tracks, ibtracs, label):
    # Within 1-degree for 1 day
    matches = huracanpy.assess.match(
        [tracks, ibtracs],
        [label, "ibtracs"],
        max_dist=None,
        mean_dist=444,
        distance_method="geod",
    )

    return matches


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


def add_other(summary, tracks_era5, tracks, label, ibtracs):
    matching_summary = _match_ibtracs(tracks, ibtracs, label=label)[
        [f"id_{label}", "id_ibtracs"]
    ]

    summary = summary.merge(matching_summary, on="id_ibtracs", how="outer")

    # Use strict matching for ERA5
    # Within 1-degree for 1 day
    # Only looking for remaining tracks that are WCSI for ERA5 and don't match IBTrACS
    tracks_era5 = tracks_era5.hrcn.sel_id(
        summary.id_era5[summary.WCSI & (summary.id_ibtracs == "")]
    )

    track_ids = np.unique(tracks.track_id)
    track_ids = track_ids[~np.isin(track_ids, np.unique(summary[f"id_{label}"]))]
    tracks = tracks.hrcn.sel_id(track_ids)

    matches_reanalysis = huracanpy.assess.match(
        [tracks_era5, tracks],
        ["era5", label],
        max_dist=None,
        mean_dist=444,
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
        summary.loc[index, f"id_{label}"] = rows.iloc[0][f"id_{label}"]

    # Add info for JRA3Q tracks not included by any matching
    track_ids = track_ids[
        ~np.isin(track_ids, np.unique(matches_reanalysis[f"id_{label}"]))
    ]
    for track_id in track_ids:
        idx = summary.index[-1] + 1
        track = tracks.hrcn.sel_id(track_id)
        summary.loc[idx, f"id_{label}"] = track_id
        summary.loc[idx, "storm_start"] = pd.to_datetime(track.time.values)[0]

    fix_columns(summary)

    return summary


def match_invests(summary, tracks_superbt, tracks_other):
    for label in tracks_other:
        print(f"Matching invests to {label}")
        tracks = tracks_other[label]

        # Only consider tracks that are not already matched with IBTrACS
        track_ids = np.unique(
            summary.loc[
                (summary.id_ibtracs == "") & (summary[f"id_{label}"] != -1),
                f"id_{label}",
            ]
        )
        tracks = tracks.hrcn.sel_id(track_ids)

        matches = huracanpy.assess.match(
            [tracks, tracks_superbt],
            [label, f"superbt_{label}"],
            max_dist=None,
            mean_dist=444,
            distance_method="geod",
        )[[f"id_{label}", f"id_superbt_{label}"]]

        # Only include one match per track to avoid expanding the table
        # Only important whether it matches at least one invest
        matches = matches.drop_duplicates(subset=f"id_{label}")

        summary = summary.merge(matches, on=f"id_{label}", how="outer")

    fix_columns(summary)

    return summary


if __name__ == "__main__":
    main()
