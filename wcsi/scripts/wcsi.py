"""
Filter for tracks that are warm core, symmetric, and intensifying (WCSI)

Usage:
    wcsi <filename_in> <filename_out>
        [--npoints=<val>]
        [--basin=<str>]
        [--b_threshold=<val>] [--vtl_threshold=<val>] [--vtu_threshold=<val>]
        [--vort_threshold=<val>] [--intensification_threshold=<val>]
        [--vort_warm_core_threshold=<val>]
        [--filter_size=<val>]
        [--coherent] [--ocean]
    wcsi (-h | --help)

Arguments:
    <filename_in>  The filename containing tracks
    --npoints=<val>
        Number of consecutive points satisfying the selected criteria required to be
        considered a TC [default: 4]
    --basin=<str>
        One of the basins supported by huracanpy. Only consider tracks that reach
        maximum intensity in this basin, and only track points in this basin
        [default: None]
    --b_threshold=<val>
        Cyclone phase space asymmetry threshold [default: 15]
    --vtl_threshold=<val>
        Cyclone phase space low-level warm core threshold [default: 0]
    --vtu_threshold=<val>
        Cyclone phase space upper-level warm core threshold [default: 0]
    --vort_threshold=<val>
        850hPa vorticity threshold (units 1e-5 s-1) [default: 6]
    --intensification_threshold=<val>
        Rate of change of (smoothed) 850hPa vorticity with time threshold [default: 0]
    --vort_warm_core_threshold=<val>
        Difference between 850hPa and 200hPa vorticity. The value from
        Hodges et al. (2017) is 6 (units 1e-5 s-1), but is not used by default here
        [default: None]
    --coherent
        Require a vortex to be identified at all pressure levels [default: False]
    --ocean
        Only count track points over ocean [default: False]
    --filter_size=<val>
        Size of uniform filter to apply to cyclone phase space parameters and vorticity
        for intensification rate [default: 5]

Options:
    -h --help
        Show this screen.
"""

import huracanpy

from . import parse_docopt
from .. import nature


def main():
    print(parse_docopt(__doc__))
    _main(**parse_docopt(__doc__))


def _main(
    filename_in,
    filename_out,
    npoints=4,
    basin=None,
    b_threshold=None,
    vtl_threshold=None,
    vtu_threshold=None,
    vort_threshold=None,
    intensification_threshold=None,
    vort_warm_core_threshold=None,
    coherent=False,
    ocean=False,
    filter_size=5,
):
    tracks = huracanpy.load(filename_in)
    wcsi_tracks, summary = nature.wcsi(
        tracks=tracks,
        npoints=npoints,
        basin=basin,
        b_threshold=b_threshold,
        vtl_threshold=vtl_threshold,
        vtu_threshold=vtu_threshold,
        vort_threshold=vort_threshold,
        vort_warm_core_threshold=vort_warm_core_threshold,
        intensification_threshold=intensification_threshold,
        coherent=coherent,
        ocean=ocean,
        filter_size=filter_size,
    )

    summary.to_parquet(filename_out + ".parquet")
    huracanpy.save(wcsi_tracks, filename_out + ".nc")

if __name__ == "__main__":
    main()
