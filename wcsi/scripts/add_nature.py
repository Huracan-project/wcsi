"""
Add a nature tag to cyclone tracks using the cyclone phase space

Usage:
    wcsi-nature <filename_in>
        [<filename_out>]
        [--filter_size=<val>]
        [--b_threshold=<val>] [--vtl_threshold=<val>] [--vtu_threshold=<val>]
        [--vort_threshold=<val>]
        [--min_count=<val>]
        [--et]
        [--smooth]
    wcsi-nature (-h | --help)

Arguments:
    <filename_in>  The filename containing tracks
    <filename_out> The filename to save the tracks with added nature to. Defaults to the
        input file with "_nature" added. Can't overwrite the file inplace. Do this
        manually after running this script
    --filter_size=<val>
        Size of uniform filter to apply to cyclone phase space parameters and vorticity
        for intensification rate [default: 5]
    --b_threshold=<val>
        Cyclone phase space asymmetry threshold [default: 15]
    --vtl_threshold=<val>
        Cyclone phase space low-level warm core threshold [default: 0]
    --vtu_threshold=<val>
        Cyclone phase space upper-level warm core threshold [default: 0]
    --vort_threshold=<val>
        850hPa vorticity threshold (units 1e-5 s-1) [default: 6]
    --min_count=<val>
        Number of consecutive points in a given nature to be labelled [default: 4]
    --et
        Add extra classifications for extratropical (Transition, Warm Seclusion)
        following Hart definitions [default: False]
    --smooth
        If the nature changes to another nature and bag again for less than min_count
        points, label it as the original nature [default: False]
    --npoints
        If "is_tc" is not already included in the tracks, this argument is passed to the
        WCSI identification [default: 4]



Options:
    -h --help
        Show this screen.
"""

import huracanpy
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from . import parse_docopt
from .. import nature


def main():
    args = parse_docopt(__doc__)
    print(args)
    _main(**args)


def _main(
    filename_in,
    filename_out=None,
    filter_size=5,
    b_threshold=15,
    vtl_threshold=0,
    vtu_threshold=0,
    vort_threshold=6,
    min_count=4,
    et=False,
    smooth=False,
    npoints=4,
):
    if filename_out is None:
        filename = filename_in.split(".")
        filename_out = ".".join(filename[:-1]) + "_nature." + filename[-1]

    tracks = huracanpy.load(filename_in)

    tracks["nature"] = ("record", np.zeros(len(tracks.record), dtype="U2"))
    for track_id, track in tqdm(tracks.groupby("track_id")):
        # Cyclone phase space
        b = uniform_filter1d(np.abs(track.cps_b), size=filter_size, mode="nearest")
        vtl = uniform_filter1d(track.cps_vtl, size=filter_size, mode="nearest")
        vtu = uniform_filter1d(track.cps_vtu, size=filter_size, mode="nearest")
        vorticity = uniform_filter1d(
            track.relative_vorticity.sel(pressure=850), size=filter_size, mode="nearest"
        )

        if "is_tc" not in track:
            is_tc = nature.wcsi_track(
                b,
                vtl,
                vtu,
                track.relative_vorticity,
                npoints=npoints,
                b_threshold=b_threshold,
                vtl_threshold=vtl_threshold,
                vtu_threshold=vtu_threshold,
                vort_threshold=vort_threshold,
                filter_size_cps=None,
                filter_size_vorticity=filter_size,
            )
        else:
            is_tc = track.is_tc.values

        nat = nature.nature(
            b,
            vtl,
            vtu,
            vorticity,
            is_tc,
            b_threshold=b_threshold,
            vtl_threshold=vtl_threshold,
            vtu_threshold=vtu_threshold,
            vort_threshold=vort_threshold,
            min_count=min_count,
            et=et,
            smooth=smooth,
        )

        tracks.nature[tracks.track_id == track_id] = nat

    tracks.hrcn.save(filename_out)


if __name__ == "__main__":
    main()
