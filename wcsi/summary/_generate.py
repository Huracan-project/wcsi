import numpy as np
import pandas as pd


def generate(track_id, time, lon, lat, wcsi):
    track_ids, start_idx = np.unique(track_id, return_index=True)

    end_idx = np.concat([start_idx[1:] + 1, [len(track_id) - 1]])

    wcsi_track_ids = np.unique(track_id[wcsi])

    summary = pd.DataFrame(
        dict(
            track_id=track_ids,
            storm_start=time[start_idx],
            storm_end=time[end_idx],
            origin_lat=lat[start_idx],
            origin_lon=lon[start_idx],
            end_lat=lat[end_idx],
            end_lon=lon[end_idx],
            is_tc=np.isin(track_ids, wcsi_track_ids),
        )
    )

    return summary
