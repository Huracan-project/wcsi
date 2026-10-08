from pathlib import Path

import huracanpy
import numpy as np


def main(filename_in, filename_out=None):
    tracks = huracanpy.load(filename_in)

    tracks_df = tracks[
        ["track_id", "time", "lon", "lat", "mslp_lon", "mslp_lat"]
    ].to_dataframe()
    tracks_df = tracks_df[tracks_df.mslp_lon != 1e25]

    duplicates = tracks_df.duplicated(
        subset=["time", "mslp_lon", "mslp_lat"], keep=False
    )
    tracks_df = tracks_df[duplicates]

    tracks_df["distance"] = huracanpy.calc.distance(
        tracks_df.lon,
        tracks_df.lat,
        tracks_df.mslp_lon,
        tracks_df.mslp_lat,
        method="geod",
    )

    # Index of row with track closest to each MSLP minima
    idx = (
        tracks_df[["time", "mslp_lon", "mslp_lat", "distance"]]
        .groupby(["time", "mslp_lon", "mslp_lat"])
        .idxmin()
        .distance.values
    )

    # Get all the indices no associated to the closest track (and set MSLP to NaN there)
    record = tracks_df.index.values
    record = record[~np.isin(record, idx)]

    tracks["mslp_min"] = ("record", tracks.mslp.values.copy())
    tracks.mslp_min[tracks.mslp_lon == 1e25] = np.nan
    tracks.mslp_min[record] = np.nan

    if filename_out is None:
        huracanpy.save(tracks, "tmp.nc")
        tracks.close()
        Path("tmp.nc").move(filename_in)
    else:
        huracanpy.save(tracks, filename_out)


if __name__ == "__main__":
    from argparse import ArgumentParser

    argy = ArgumentParser()
    argy.add_argument("filename_in")
    argy.add_argument("filename_out", nargs="?", default=None)
    args = argy.parse_args()
    print(args)
    main(args.filename_in, args.filename_out)
