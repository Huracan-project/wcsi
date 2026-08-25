import huracanpy
import numpy as np
from tqdm import tqdm

from wcsi import nature


def main():
    ibtracs = huracanpy.load("IBTrACS_6h_1940-2024_Tropical-Storms.nc")
    ibtracs = ibtracs.isel(record=ibtracs.track_id.str.slice(0, 4).astype(int) >= 1979)

    # Only tracks with valid wind data and tropical storm category for five consecutive
    # timesteps
    ibtracs.isel(record=(ibtracs.nature == "TS") & ~np.isnan(ibtracs.wind))

    consecutive = (np.diff(ibtracs.time) == np.timedelta64(6, "h")) & (
        ibtracs.track_id[1:] == ibtracs.track_id[:-1]
    )
    nconsecutive = consecutive.groupby(ibtracs.track_id[1:]).sum()

    track_ids = nconsecutive.track_id[nconsecutive.values >= 5]
    ibtracs = ibtracs.hrcn.sel_id(track_ids.values)

    trid_gaps = []
    trid_tc = []
    trid_tc_flat = []
    trid_not_tc = []
    for track_id, track in tqdm(
        ibtracs.groupby("track_id"), total=ibtracs.track_id.hrcn.nunique()
    ):
        wind = (
            track.wind.copy()
            .expand_dims("pressure")
            .assign_coords(coords={"pressure": ("pressure", [850])})
        )
        if (np.diff(track.time) == np.timedelta64(6, "h")).all():
            istc = nature.wcsi_track(
                relative_vorticity=wind, coherent=False, vort_threshold=None
            )
            if istc.any():
                trid_tc.append(track_id)
            else:
                istc = nature.wcsi_track(
                    relative_vorticity=wind,
                    coherent=False,
                    vort_threshold=None,
                    intensification_threshold=-0.01,
                )
                if istc.any():
                    trid_tc_flat.append(track_id)
                else:
                    trid_not_tc.append(track_id)
        else:
            trid_gaps.append(track_id)

    print(len(trid_tc), len(trid_tc_flat), len(trid_not_tc), len(trid_gaps))

    tracks_gaps = ibtracs.hrcn.sel_id(trid_gaps).hrcn.interp_time(freq="6h")
    trid_gaps_tc = []
    trid_gaps_tc_flat = []
    trid_gaps_not_tc = []
    for track_id, track in tqdm(
        tracks_gaps.groupby("track_id"), total=tracks_gaps.track_id.hrcn.nunique()
    ):
        wind = (
            track.wind.copy()
            .expand_dims("pressure")
            .assign_coords(coords={"pressure": ("pressure", [850])})
        )
        istc = nature.wcsi_track(
            relative_vorticity=wind, coherent=False, vort_threshold=None
        )
        if istc.any():
            trid_gaps_tc.append(track_id)
        else:
            istc = nature.wcsi_track(
                relative_vorticity=wind,
                coherent=False,
                vort_threshold=None,
                intensification_threshold=-0.01,
            )
            if istc.any():
                trid_gaps_tc_flat.append(track_id)
            else:
                trid_gaps_not_tc.append(track_id)

    print(len(trid_gaps_tc), len(trid_gaps_not_tc), len(trid_gaps_tc_flat))


if __name__ == "__main__":
    main()
