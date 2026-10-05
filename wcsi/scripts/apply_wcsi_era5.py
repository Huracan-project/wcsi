import huracanpy
import numpy as np

from wcsi import wcsi, summary


def main():
    tracks = huracanpy.load("ERA5_all_nature-mslp_wcsi-new.nc")

    closed_mslp = ~tracks.mslp_minima.isnull().values

    print("WCS")
    wcs = wcsi(
        tracks.track_id,
        tracks.cps_b,
        tracks.cps_vtl,
        tracks.cps_vtu,
        tracks.relative_vorticity,
        filter_all=None,
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
    tracks["wcs"] = ("record", wcs & closed_mslp)

    # Save time by passing the already calculated WCS criteria
    # Don't apply closed MSLP because this may remove tracks that have been shorted to
    # less than four points
    print("WCSI")
    tracks["wcsi"] = (
        "record",
        wcsi(
            tracks.track_id,
            relative_vorticity=tracks.relative_vorticity,
            filter_all=wcs,
            filter_any=closed_mslp,
            npoints=4,
            vort_threshold=None,
            vort_warm_core_threshold=None,
            intensification_threshold=0,
            coherent=False,
            filter_size=5,
        ),
    )

    print("Generating summary")
    table = summary.generate(
        tracks.track_id, tracks.time, tracks.lon, tracks.lat, tracks.wcsi
    )

    track_ids = np.unique(tracks.track_id[tracks.wcs])
    table["WCS"] = np.isin(table.track_id, track_ids)

    table.to_parquet("WCSI_summary_ERA5.parquet")
    huracanpy.save(tracks, "ERA5.nc")


if __name__ == "__main__":
    main()
