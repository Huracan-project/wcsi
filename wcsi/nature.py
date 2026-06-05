from itertools import groupby

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import xarray as xr


def wcsi(
    tracks: xr.Dataset,
    npoints: int = 4,
    basin=None,
    b_threshold: float | None = 15,
    vtl_threshold: float | None = 0,
    vtu_threshold: float | None = 0,
    vort_threshold: float | None = 6,
    vort_warm_core_threshold: float | None = None,
    intensification_threshold: float | None = 0,
    coherent: bool = True,
    ocean: bool = False,
    filter_size: int = 5,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """

    Parameters
    ----------
    tracks
    npoints
        Number of consecutive points that all criteria need to be met
    basin
        Restrict subsetting to a specific basin. Determined by maximum intensity
    b_threshold
        Maximum value of the cyclone phase space asymmetry parameter
    vtl_threshold
        Minimum value of the cyclone phase space low-level warm core parameter
    vtu_threshold
        Minimum value of the cyclone phase space upper-level warm core parameter
    vort_threshold
        Minimum value of 850hPa vorticity. Default has units of 10e-5 s-1 following
        conventions from TRACK
    vort_warm_core_threshold
        Minimum value for the difference between 850hPa vorticity and 200hPa vorticity.
        Default has units of 10e-5 s-1 following conventions from TRACK
    intensification_threshold
        Minimum value for the increase in 850hPa vorticity
    coherent
        Require that the vorticity has a value at each vertical level. Not NaN or 1e25
        (the fill value in TRACK)
    ocean
        Require that the points must be over Ocean
    filter_size
        Size of running mean (in number of points) to apply to the cyclone phase space
        parameters and 850hPa vorticity (for intensification) before thresholding

    Returns
    -------
    The subset of tracks that meet the criteria and a table summarising all the tracks
    """
    summary = []
    tc_tracks = []

    if basin is not None:
        # Skip tracks that have max intensity in a different basin if filtering by
        # basin
        if "basin" not in tracks:
            tracks = tracks.hrcn.add_basin()

        tracks_max_intensity = tracks.hrcn.get_apex_vals("vorticity")
        tracks = tracks.hrcn.sel_id(
            tracks_max_intensity.track_id[tracks_max_intensity.basin == basin]
        )

    for track_id, track in tqdm(tracks.groupby("track_id")):
        # Only storms that are tropical cyclones at some point in their lifecycle
        track["is_tc"] = wcsi_track(
            track.cps_b,
            track.cps_vtl,
            track.cps_vtu,
            track.relative_vorticity,
            track.hrcn.get_is_ocean() if ocean else None,
            npoints=npoints,
            b_threshold=b_threshold,
            vtl_threshold=vtl_threshold,
            vtu_threshold=vtu_threshold,
            vort_threshold=vort_threshold,
            vort_warm_core_threshold=vort_warm_core_threshold,
            intensification_threshold=intensification_threshold,
            coherent=coherent,
            ocean=ocean,
            filter_size_cps=filter_size,
            filter_size_vorticity=filter_size,
        )

        if basin is not None:
            is_tc = (track.is_tc & (track.basin == basin)).values.any()
        else:
            is_tc = track.is_tc.values.any()

        if is_tc:
            tc_tracks.append(track)

        times = pd.to_datetime(track.time)
        summary.append(
            pd.DataFrame(
                [
                    dict(
                        track_id=track.track_id.values[0],
                        storm_start=times[0],
                        storm_end=times[-1],
                        origin_lat=track.lat.data[0],
                        origin_lon=track.lon.data[0],
                        end_lat=track.lat.data[-1],
                        end_lon=track.lon.data[-1],
                        is_tc=is_tc,
                    )
                ]
            )
        )

    summary = pd.concat(summary, ignore_index=True)
    tracks = xr.concat(tc_tracks, dim="record")

    return tracks, summary


def wcsi_track(
    cps_b=None,
    cps_vtl=None,
    cps_vtu=None,
    relative_vorticity=None,
    is_ocean=None,
    *,
    npoints: int = 4,
    b_threshold: float | None = 15,
    vtl_threshold: float | None = 0,
    vtu_threshold: float | None = 0,
    vort_threshold: float | None = 6,
    vort_warm_core_threshold: float | None = None,
    intensification_threshold: float | None = 0,
    coherent: bool = True,
    ocean: bool = False,
    filter_size_cps: int | None = 5,
    filter_size_vorticity: int | None = 5,
) -> NDArray[np.bool]:
    # Cyclone Phase Space
    try:
        tc = wcs(
            np.abs(cps_b) if cps_b is not None else None,
            cps_vtl,
            cps_vtu,
            filter_size=filter_size_cps,
            b_threshold=b_threshold,
            vtl_threshold=vtl_threshold,
            vtu_threshold=vtu_threshold,
        )
    except (ValueError, AttributeError):
        if relative_vorticity is not None:
            # If no CPS thresholds are set, start with True everywhere
            tc = np.ones(len(relative_vorticity), dtype=bool)
        else:
            msg = (
                "Must specifify at least one of the CPS parameters or relativevorticity"
            )
            raise ValueError(msg)

    # Minimum vorticity
    if vort_threshold is not None and relative_vorticity is not None:
        tc = tc & (relative_vorticity.sel(pressure=850) > vort_threshold)

    # Intensification rate
    if intensification_threshold is not None and relative_vorticity is not None:
        vo850 = relative_vorticity.sel(pressure=850)
        if filter_size_vorticity is not None:
            vo850 = uniform_filter1d(
                vo850,
                size=filter_size_vorticity,
                mode="nearest",
            )
        tc = tc & (np.gradient(vo850) > intensification_threshold)

    # Coherent
    if coherent and relative_vorticity is not None:
        # Check for NaNs and mask value in TRACK (1e25)
        tc = tc & ~(np.isnan(relative_vorticity) | (relative_vorticity == 1e25)).any(
            dim="pressure"
        )

    # Vorticity based warm core threshold
    if vort_warm_core_threshold is not None and relative_vorticity is not None:
        tc = tc & (
            (
                relative_vorticity.sel(pressure=850)
                - relative_vorticity.sel(pressure=200)
            )
            > vort_warm_core_threshold
        )

    # Over ocean
    if ocean and is_ocean is not None:
        tc = tc & is_ocean

    # Check that applied criteria are satisfied for consective npoints
    if npoints > 1:
        category_consecutive = [(k, sum(1 for i in g)) for k, g in groupby(tc)]

        idx = 0
        for category, count in category_consecutive:
            if category and count < npoints:
                tc[idx : idx + count] = False
            idx += count

    return tc


def wcs(
    b: NDArray[np.floating] | None,
    vtl: NDArray[np.floating] | None,
    vtu: NDArray[np.floating] | None,
    *,
    filter_size: int | None = None,
    b_threshold: float | None = 10,
    vtl_threshold: float | None = 0,
    vtu_threshold: float | None = 0,
) -> NDArray[np.bool]:
    """Identify where a track is a tropical cyclone by the cyclone phase space
    definition (warm core and symmetric)

    Default thresholds are using North Atlantic definitions from table 1 in
    https://www.sciencedirect.com/science/article/pii/S2225603223000516

    Parameters
    ----------
    b
        Cyclone phase space asymmetry
    vtl
        Cyclone phase space low-level warm core
    vtu
        Cyclone phase space upper-level warm core
    filter_size
        Length (in timesteps) of the uniform filter to apply to the cyclone phase space
        parameters. If None, don't apply filter

    b_threshold
        The threshold of the asymmetry parameter, below which is considered to be a
        tropical cyclone

    vtl_threshold
        The threshold of the low-level warm-core parameter, above which is considered
        to be a tropical cyclone

    vtu_threshold
        The threshold of the upper-level warm-core parameter, above which is considered
        to be a tropical cyclone

    Returns
    -------
    True where the CPS criteria for a tropical cyclone is achieved, False otherwise

    """
    if (b is None and vtl is None and vtu is None) or (
        not b_threshold and not vtl_threshold and not vtu_threshold
    ):
        raise ValueError("Need to pass at least one variable and threshold")

    if filter_size is not None:
        if b_threshold is not None and b is not None:
            b = uniform_filter1d(b, size=filter_size, mode="nearest")
        if vtl_threshold is not None and vtl is not None:
            vtl = uniform_filter1d(vtl, size=filter_size, mode="nearest")
        if vtu_threshold is not None and vtu is not None:
            vtu = uniform_filter1d(vtu, size=filter_size, mode="nearest")

    condition = []
    if b_threshold is not None and b is not None:
        condition.append(b <= b_threshold)
    if vtl_threshold is not None and vtl is not None:
        condition.append(vtl > vtl_threshold)
    if vtu_threshold is not None and vtu is not None:
        condition.append(vtu > vtu_threshold)

    if len(condition) == 1:
        return condition[0]
    else:
        is_tc = condition[0]
        for other_condition in condition[1:]:
            is_tc = is_tc & other_condition

    return is_tc


def nature(
    b: NDArray[np.floating],
    vtl: NDArray[np.floating],
    vtu: NDArray[np.floating],
    vort: NDArray[np.floating],
    is_tc: NDArray[np.bool],
    *,
    b_threshold: float = 15,
    vtl_threshold: float = 0,
    vtu_threshold: float = 0,
    vort_threshold: float = 6,
    min_count: int = 4,
    et: bool = False,
    smooth: bool = False,
) -> NDArray[np.str_]:
    """Derive a nature tag from the cyclone structure

    TC - Tropical Cyclone
    Vo - Weak Vortex
    BC - Baroclinic
    Tr - Trough
    MV - Mid-level vortex
    Ot - Other

    If et=True
    ET - Extratropical transition
    WS - Warm seclusion

    Parameters
    ----------
    b
        Cyclone phase space asymmetry
    vtl
        Cyclone phase space low-level warm core
    vtu
        Cyclone phase space upper-level warm core
    vort
        850hPa vorticity
    is_tc
        Points previously used to identify the cyclone as tropical cyclone (e.g. WCSI)
    b_threshold
        The threshold of the asymmetry parameter, below which is considered to be a
        tropical cyclone
    vtl_threshold
        The threshold of the low-level warm-core parameter, above which is considered
        to be a tropical cyclone
    vtu_threshold
        The threshold of the upper-level warm-core parameter, above which is considered
        to be a tropical cyclone
    vort_threshold
        The minimum threshold for 850-hPa vorticity, below which is considered as a
        weak vortex
    min_count
        Number of
    et
        Add labels for extratropical transition. Each stage must last for min_count
        points
    smooth
        Add a smoothing to the nature tags. Any excursions less than min_count are
        removed

    Returns
    -------


    """
    nat = np.zeros(len(vort), dtype="U2")

    # Too weak = vortex
    weak = vort < vort_threshold
    nat[weak] = "Vo"

    # WCSI label as tropical cyclone
    nat[is_tc] = "TC"

    # Other CPS categories
    symmetric = b <= b_threshold
    warm_core = vtl > vtl_threshold
    trough = vtu <= vtu_threshold

    # Any Warm core/symmetric periods adjacent to TC are also TC
    # Label as tropical storm for now
    nat[(nat == "") & (b <= b_threshold) & (vtl > vtl_threshold)] = "TS"
    nat_consecutive = [(k, sum(1 for _ in g)) for k, g in groupby(nat)]
    idx = 0
    for m, (nat_, count) in enumerate(nat_consecutive):
        if nat_ == "TS":
            # Allow for <1 day excursions between TC-TS
            idx_start = max(0, idx - min_count)
            if (nat[idx_start : idx + count] == "TC").any():
                nat[idx_start : idx + count] = "TC"

            idx_end = min(len(nat), idx + count + min_count)
            if (nat[idx + count : idx_end] == "TC").any():
                nat[idx:idx_end] = "TC"

        idx += count
    nat[nat == "TS"] = ""

    # Extratropical transition
    # Look after the last TC point for ET
    if et:
        idx = np.where(nat == "TC")[0]
        if len(idx) > 0:
            idx = idx[-1] + 1
            new_idx = fill_next_nature(
                nat, ~symmetric & warm_core & ~weak, "ET", idx, min_count
            )
            if new_idx >= idx + min_count:
                idx = new_idx
                new_idx = fill_next_nature(
                    nat, ~symmetric & ~warm_core & ~weak, "BC", idx, min_count
                )
                if new_idx >= idx + min_count:
                    idx = new_idx
                    fill_next_nature(nat, warm_core & ~weak, "WS", idx, min_count)
                else:
                    # If nothing was labelled as baroclinic following ET, remove ET
                    nat[np.isin(nat, ["ET", "BC"])] = ""
            else:
                # ET lasted less than min_count remove ET
                nat[nat == "ET"] = ""

    # Label remaining unlabelled sections
    # Asymmetric = baroclinic
    nat[(nat == "") & ~symmetric] = "BC"
    # Upper-level cold core = Trough
    nat[(nat == "") & trough] = "Tr"
    # Low-level cold core = Mid level vortex
    nat[(nat == "") & ~warm_core] = "MV"
    # Warm core symmetric not TC (decaying)
    nat[(nat == "")] = "Ot"

    if smooth:
        smooth_excursions(nat, min_count)

    return nat


def fill_next_nature(nat, condition, label, idx, min_count):
    while idx < len(nat) and condition[idx]:
        nat[idx] = label
        idx += 1

    # Allow for short excursions. Look ahead to see if it comes back to the same
    # category
    slice_ahead = slice(idx, min(idx + min_count, len(nat)))
    if condition[slice_ahead].any():
        # Start again from the first point
        print(condition[slice_ahead])
        idx = idx + np.where(condition[slice_ahead])[0][0]
        fill_next_nature(nat, condition, label, idx, min_count)

    return idx


def smooth_excursions(nat, min_count):
    # Smooth out any shorter than 1 day excursions
    nat_consecutive = [(k, sum(1 for _ in g)) for k, g in groupby(nat)]
    idx = nat_consecutive[0][1]
    for m in range(1, len(nat_consecutive) - 1):
        nat_, count = nat_consecutive[m]
        if count < min_count:
            if nat_consecutive[m - 1][0] == nat_consecutive[m + 1][0]:
                new_nat = nat_consecutive[m - 1][0]
                nat[idx : idx + count] = new_nat
                nat_consecutive[m] = (new_nat, count)

        idx += count
