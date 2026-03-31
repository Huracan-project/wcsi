"""
Get the minimum pressure while a tropical storm for all IBTrACS tracks and the
corresponding MSLP from ERA5 and JRA3Q if there is a point within 165km (~1.5 degrees)
"""

import huracanpy
import numpy as np
from tqdm import tqdm


def main():
    ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    tracks_era5 = huracanpy.load("ERA5_all.nc")
    tracks_jra3q = huracanpy.load("JRA3Q_nolat-tcident.nc")
    matched_lmi = match_lmi(ibtracs, tracks_era5, tracks_jra3q)
    matched_lmi.to_parquet("matched_lmi.parquet")


def match_lmi(ibtracs, tracks_era5, tracks_jra3q):
    ibtracs_tc_lmi = ibtracs.isel(
        record=np.where((ibtracs.nature == "TS") & ~np.isnan(ibtracs.mslp))[0]
    ).hrcn.get_apex_vals("mslp", stat="min")

    track_ids = ibtracs_tc_lmi.track_id.values
    ibtracs_tc_lmi = ibtracs_tc_lmi.rename(track_id="record").drop_vars("record")
    ibtracs_tc_lmi = ibtracs_tc_lmi.assign(track_id=("record", track_ids))

    matched_lmi = huracanpy.assess.match(
        [ibtracs_tc_lmi, tracks_era5, tracks_jra3q],
        ["ibtracs", "era5", "jra3q"],
        max_dist=165,
    )
    matched_lmi = matched_lmi[matched_lmi.id_ibtracs.astype(str) != "nan"]

    matched_lmi["year"] = np.zeros(len(matched_lmi), dtype=int)
    matched_lmi["lat"] = np.zeros(len(matched_lmi))
    for dataset in ["ibtracs", "era5", "jra3q"]:
        matched_lmi[f"mslp_{dataset}"] = np.zeros(len(matched_lmi))

    for n, row in tqdm(matched_lmi.iterrows()):
        lmi_ib = ibtracs_tc_lmi.hrcn.sel_id(track_id=row.id_ibtracs)
        matched_lmi.loc[n, "year"] = lmi_ib.time.dt.year.values[()]
        matched_lmi.loc[n, "mslp_ibtracs"] = lmi_ib.mslp.values[()]
        matched_lmi.loc[n, "lat"] = lmi_ib.lat.values[()]
        for dataset, tracks in [("era5", tracks_era5), ("jra3q", tracks_jra3q)]:
            if ~np.isnan(row[f"id_{dataset}"]):
                track = tracks.hrcn.sel_id(row[f"id_{dataset}"])
                track = track.isel(record=np.where(track.time == lmi_ib.time.values)[0])
                matched_lmi.loc[n, f"mslp_{dataset}"] = track.mslp.values[()]

    for column in ["id_era5", "id_jra3q"]:
        matched_lmi.loc[np.isnan(matched_lmi[column]), column] = -1
        matched_lmi[column] = matched_lmi[column].astype(int)

    return matched_lmi


if __name__ == "__main__":
    main()
