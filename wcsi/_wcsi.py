from typing import Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from .ragged import mask_short, padded_index, uniform_filter1d


def wcsi(
    track_id: NDArray[Any],
    cps_b: NDArray[np.floating] | None = None,
    cps_vtl: NDArray[np.floating] | None = None,
    cps_vtu: NDArray[np.floating] | None = None,
    relative_vorticity: xr.DataArray | None = None,
    filter_all=None,
    filter_any=None,
    *,
    npoints: int = 4,
    b_threshold: float | None = 15,
    vtl_threshold: float | None = 0,
    vtu_threshold: float | None = 0,
    vort_threshold: float | None = 6,
    vort_warm_core_threshold: float | None = None,
    intensification_threshold: float | None = 0,
    coherent: bool = False,
    filter_size: int | None = 5,
) -> NDArray[np.bool]:
    """

    filter_all and filter_any are for additional criteria that need to either be
    satifisfied for all points or any point
    e.g. We could require that all points need to be over ocean. So, with default
    parameters there needs to be four consecutive points with cyclone structure over
    Ocean. Or we could just require that at least one point is over ocean. So, with
    default parameters we would still requite at least four consecutive points with
    cyclone structure, but only one of those points needs to be over Ocean

    Parameters
    ----------
    track_id
    cps_b
    cps_vtl
    cps_vtu
    relative_vorticity
    filter_all
    filter_any
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
    filter_size
        Size of running mean (in number of points) to apply to the cyclone phase space
        parameters and 850hPa vorticity (for intensification) before thresholding

    Returns
    -------
    The subset of tracks that meet the criteria and a table summarising all the tracks
    """

    # Apply smoothing
    if filter_size is not None and filter_size > 1:
        idx, inverse = padded_index(
            track_id, pad=filter_size // 2, pad_location="outer"
        )

        if cps_b is not None:
            cps_b = uniform_filter1d(
                track_id, np.abs(cps_b), size=filter_size, idx=idx, inverse=inverse
            )
        if cps_vtl is not None:
            cps_vtl = uniform_filter1d(
                track_id, cps_vtl, size=filter_size, idx=idx, inverse=inverse
            )
        if cps_vtu is not None:
            cps_vtu = uniform_filter1d(
                track_id, cps_vtu, size=filter_size, idx=idx, inverse=inverse
            )

    # Initialise to True everywhere and succesively apply other filters
    tc = np.ones(len(track_id), dtype=bool)

    # Cyclone Phase Space
    if b_threshold is not None and cps_b is not None:
        tc = tc & (cps_b <= b_threshold)
    if vtl_threshold is not None and cps_vtl is not None:
        tc = tc & (cps_vtl > vtl_threshold)
    if vtu_threshold is not None and cps_vtu is not None:
        tc = tc & (cps_vtu > vtu_threshold)

    # Minimum vorticity
    if vort_threshold is not None and relative_vorticity is not None:
        tc = tc & (relative_vorticity.sel(pressure=850).values > vort_threshold)

    # Intensification rate
    if intensification_threshold is not None and relative_vorticity is not None:
        vo850 = relative_vorticity.sel(pressure=850).values
        if filter_size is not None and filter_size > 1:
            # Index for filtering has already been calculated
            vo850 = uniform_filter1d(
                track_id, vo850, size=filter_size, idx=idx, inverse=inverse
            )
        else:
            # Still need a padded index for gradient calculation, but it has not been
            # calculated if the filter size is unused
            idx, inverse = padded_index(track_id, pad=1, pad_location="inner")
        tc = tc & (np.gradient(vo850[idx])[inverse] > intensification_threshold)

    # Coherent
    if coherent and relative_vorticity is not None:
        # Check for NaNs and mask value in TRACK (1e25)
        tc = tc & (
            ~(relative_vorticity.isnull() | (relative_vorticity == 1e25))
            .any(dim="pressure")
            .values
        )

    # Vorticity based warm core threshold
    if vort_warm_core_threshold is not None and relative_vorticity is not None:
        tc = tc & (
            (
                relative_vorticity.sel(pressure=850)
                - relative_vorticity.sel(pressure=200)
            ).values
            > vort_warm_core_threshold
        )

    # Over ocean
    if filter_all is not None:
        tc = tc & filter_all

    # Check that applied criteria are satisfied for consecutive npoints
    if npoints > 1:
        tc = mask_short(track_id, tc, min_count=npoints)

    # Criteria applied after count check
    # Only at least one WCSI point needs to meet the criteria
    # Closed MSLP contour. e.g. intensify as a tropical vortex, develop MSLP minimum
    # then weaken
    # Basin - Develops as a TC in one basin then enters another basin, still as a TC
    if filter_any is not None:
        tc = tc & filter_any

    return np.asarray(tc)
