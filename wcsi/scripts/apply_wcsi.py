from pathlib import Path

import huracanpy
import numpy as np

from wcsi import wcsi, nature, summary


def main(filename_in, filename_out=None):
    tracks = huracanpy.load(filename_in)

    # If lon/lat associated with MSLP are NaN then no genuine minima was identified
    closed_mslp = ~np.isnan(tracks.mslp_min.values)

    print("WCS")
    wcs = wcsi(
        tracks.track_id,
        tracks.cps_b,
        tracks.cps_vtl,
        tracks.cps_vtu,
        tracks.relative_vorticity,
        filter_all=closed_mslp,
        filter_any=None,
        npoints=4,
        b_threshold=15,
        vtl_threshold=0,
        vtu_threshold=0,
        vort_threshold=None,
        vort_warm_core_threshold=None,
        intensification_threshold=None,
        coherent=False,
        filter_size=5,
    )
    tracks["wcs"] = ("record", wcs)

    print("WCSI")
    tracks["wcsi"] = (
        "record",
        wcsi(
            tracks.track_id,
            tracks.cps_b,
            tracks.cps_vtl,
            tracks.cps_vtu,
            tracks.relative_vorticity,
            filter_all=closed_mslp,
            filter_any=None,
            npoints=4,
            b_threshold=15,
            vtl_threshold=0,
            vtu_threshold=0,
            vort_threshold=None,
            vort_warm_core_threshold=None,
            intensification_threshold=0,
            coherent=False,
            filter_size=5,
        ),
    )

    print("Add nature")
    tracks["nature"] = (
        "record",
        nature(
            tracks.track_id,
            tracks.cps_b,
            tracks.cps_vtl,
            tracks.cps_vtu,
            np.isnan(tracks.mslp_min),
            tracks.wcsi,
        ),
    )

    print("Generating summary")
    table = summary.generate(
        tracks.track_id, tracks.time, tracks.lon, tracks.lat, tracks.wcsi
    )

    track_ids = np.unique(tracks.track_id[tracks.wcs])
    table["WCS"] = np.isin(table.track_id, track_ids)

    table.to_parquet(filename_in.replace(".nc", ".parquet"))

    # Can't save to same filename, move after
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
