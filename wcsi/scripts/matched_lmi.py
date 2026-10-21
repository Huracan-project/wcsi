"""
Get the minimum pressure while a tropical storm for all IBTrACS tracks and the
corresponding MSLP from ERA5 and JRA3Q if there is a point within 444km (~4 degrees)
"""

import huracanpy
import numpy as np
from tqdm import tqdm


def main():
    ibtracs = huracanpy.load("IBTrACS.nc")

    datasets = dict(
        era5=huracanpy.load("ERA5.nc"),
        jra3q=huracanpy.load("JRA3Q.nc"),
        ecan=huracanpy.load("ECMWF-OP-AN_TRACK.nc"),
    )

    matched_lmi = match_lmi(ibtracs, datasets)
    matched_lmi.to_parquet("matched_lmi.parquet")


def match_lmi(ibtracs, datasets):
    # Select all tropical cyclone points with valid MSLP data
    # Take the minimum MSLP point per track
    ibtracs_tc_lmi = ibtracs.isel(
        record=np.where((ibtracs.nature == "TS") & ~np.isnan(ibtracs.mslp))[0]
    ).hrcn.get_apex_vals("mslp", stat="min")

    # Reassign the record dimension (otherwise assess.match fails)
    track_ids = ibtracs_tc_lmi.track_id.values
    ibtracs_tc_lmi = ibtracs_tc_lmi.rename(track_id="record").drop_vars("record")
    ibtracs_tc_lmi = ibtracs_tc_lmi.assign(track_id=("record", track_ids))

    # Match to each dataset
    matched_lmi = []
    for dataset in datasets:
        matched_lmi.append(
            huracanpy.assess.match(
                [ibtracs_tc_lmi, datasets[dataset]],
                ["ibtracs", dataset],
                max_dist=444,
            )
        )

    left = matched_lmi[0]
    for right in matched_lmi[1:]:
        left = left.merge(right, on="id_ibtracs", how="outer")

    matched_lmi = left

    matched_lmi["year"] = np.zeros(len(matched_lmi), dtype=int)
    matched_lmi["lat"] = np.zeros(len(matched_lmi))
    for dataset in ["ibtracs", "era5", "jra3q"]:
        matched_lmi[f"mslp_{dataset}"] = np.zeros(len(matched_lmi))

    # Add track info from each dataset to the matching table
    for n, row in tqdm(matched_lmi.iterrows(), total=len(matched_lmi)):
        lmi_ib = ibtracs_tc_lmi.hrcn.sel_id(track_id=row.id_ibtracs)
        matched_lmi.loc[n, "year"] = lmi_ib.time.dt.year.values[()]
        matched_lmi.loc[n, "mslp_ibtracs"] = lmi_ib.mslp.values[()]
        matched_lmi.loc[n, "lat"] = lmi_ib.lat.values[()]
        for dataset in datasets:
            tracks = datasets[dataset]
            if ~np.isnan(row[f"id_{dataset}"]):
                track = tracks.hrcn.sel_id(row[f"id_{dataset}"])
                track = track.isel(record=np.where(track.time == lmi_ib.time.values)[0])
                matched_lmi.loc[n, f"mslp_{dataset}"] = track.mslp.values[()]

    # Track IDs as integers. pandas likes to mix data types
    for dataset in datasets:
        column = f"id_{dataset}"
        matched_lmi.loc[np.isnan(matched_lmi[column]), column] = -1
        matched_lmi[column] = matched_lmi[column].astype(int)

    return matched_lmi


if __name__ == "__main__":
    main()
