"""Command-line interface for applying WCSI and optional nature classifications."""

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
import huracanpy
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from wcsi import nature, wcsi


app = App()

FilterNames = Annotated[
    list[str] | None,
    Parameter(consume_multiple=True, choices=["closed_mslp", "ocean"]),
]


@app.default
def main(
    filename_in: Path,
    filename_out: Path,
    *,
    varname: str = "wcsi",
    add_nature: bool = False,
    nature_name: str = "nature",
    drop: bool = False,
    npoints: int = 4,
    basin: str | None = None,
    b_name: str | None = "cps_b",
    b_threshold: float | None = 15,
    vtl_name: str | None = "cps_vtl",
    vtl_threshold: float | None = 0,
    vtu_name: str | None = "cps_vtu",
    vtu_threshold: float | None = 0,
    vort_name: str | None = "relative_vorticity",
    vort_threshold: float | None = None,
    intensification_threshold: float | None = 0,
    vort_warm_core_threshold: float | None = None,
    coherent: bool = False,
    filter_all: FilterNames = None,
    closed_mslp_name: str = "mslp_min",
    filter_size: int | None = 5,
    vort_warm_core_filter_size: int | None = None,
    min_count: int = 4,
    et: bool = False,
    smooth: bool = False,
) -> None:
    """Classify warm-core, symmetric, intensifying cyclone-track points.

    The WCSI Boolean mask is added to the input dataset and, when requested, cyclone
    nature tags are derived from it. The augmented tracks are written to
    ``filename_out``. WCSI criteria with a ``None`` threshold or variable name are
    disabled. All running filters and consecutive-point checks respect contiguous
    ``track_id`` boundaries.

    Parameters
    ----------
    filename_in
        Input track dataset supported by ``huracanpy.load``.
    filename_out
        Path to which the classified dataset is written.
    varname
        Name of the output Boolean variable containing the WCSI mask.
    add_nature
        Also derive point-level cyclone-nature tags from the new WCSI mask, cyclone
        phase-space variables, and closed-MSLP mask.
    nature_name
        Name of the output variable containing the two-character nature tags.
    drop
        If true, retain only tracks containing at least one WCSI-classified point.
    npoints
        Minimum number of consecutive points that must satisfy all enabled WCSI
        criteria.
    basin
        Optional WMO-TC basin code. When supplied, only points within this basin can
        contribute to a qualifying consecutive run.
    b_name
        Name of the cyclone phase-space asymmetry variable. Set to ``None`` to omit
        it regardless of ``b_threshold``.
    b_threshold
        Maximum absolute cyclone asymmetry. Set to ``None`` to disable the criterion.
    vtl_name
        Name of the lower-tropospheric thermal-wind variable.
    vtl_threshold
        Minimum lower-tropospheric thermal wind. Set to ``None`` to disable it.
    vtu_name
        Name of the upper-tropospheric thermal-wind variable.
    vtu_threshold
        Minimum upper-tropospheric thermal wind. Set to ``None`` to disable it.
    vort_name
        Name of the relative-vorticity variable with a ``pressure`` coordinate. Set
        to ``None`` to disable all vorticity-dependent criteria.
    vort_threshold
        Minimum 850-hPa relative vorticity. Values conventionally use units of
        10^-5 s^-1. Disabled by default.
    intensification_threshold
        Minimum centred point-to-point gradient of smoothed 850-hPa vorticity. The
        gradient is not divided by elapsed time, so its units depend on the input
        timestep. Set to ``None`` to disable intensification.
    vort_warm_core_threshold
        Minimum difference between 850- and 200-hPa relative vorticity. Set to
        ``None`` to disable the vorticity warm-core criterion.
    coherent
        Require valid vorticity at every pressure level for a qualifying point.
    filter_all
        Additional pointwise criteria that must hold throughout a qualifying run.
        Accepted values are ``closed_mslp`` and ``ocean``; multiple values may be
        supplied.
    closed_mslp_name
        Variable whose non-missing values indicate an associated closed MSLP minimum.
        Used when ``filter_all`` includes ``closed_mslp``.
    filter_size
        Width, in track points, of the running mean applied to cyclone phase-space
        values and 850-hPa vorticity. Set to ``None`` or at most one to disable it.
    vort_warm_core_filter_size
        Optional running-mean width for the vertical vorticity difference.
    min_count
        Minimum duration, in consecutive track points, used when joining or smoothing
        nature classifications. Used only when ``add_nature`` is true.
    et
        Request the optional extratropical-transition and warm-seclusion nature
        categories. Used only when ``add_nature`` is true.
    smooth
        Fill short excursions away from the surrounding nature classification. Used
        only when ``add_nature`` is true.

    Notes
    -----
    ``ocean`` uses the ``get_is_ocean`` huracanpy dataset accessor. Basin and
    additional filters are combined before the consecutive-point requirement is
    evaluated.
    """
    tracks = huracanpy.load(str(filename_in))
    filter_mask = additional_filter_mask(
        tracks,
        filter_all=filter_all,
        basin=basin,
        closed_mslp_name=closed_mslp_name,
    )
    relative_vorticity = tracks[vort_name] if vort_name is not None else None

    istc = wcsi(
        track_id=tracks.track_id.values,
        cps_b=variable_values(tracks, b_name),
        cps_vtl=variable_values(tracks, vtl_name),
        cps_vtu=variable_values(tracks, vtu_name),
        relative_vorticity=relative_vorticity,
        filter_all=filter_mask,
        npoints=npoints,
        b_threshold=b_threshold,
        vtl_threshold=vtl_threshold,
        vtu_threshold=vtu_threshold,
        vort_threshold=vort_threshold,
        vort_warm_core_threshold=vort_warm_core_threshold,
        intensification_threshold=intensification_threshold,
        coherent=coherent,
        filter_size=filter_size,
        vort_warm_core_filter_size=vort_warm_core_filter_size,
    )
    tracks[varname] = ("record", istc)

    if add_nature:
        if b_name is None or vtl_name is None or vtu_name is None:
            raise ValueError(
                "b_name, vtl_name, and vtu_name are required when add_nature is true"
            )
        if b_threshold is None or vtl_threshold is None or vtu_threshold is None:
            raise ValueError(
                "b_threshold, vtl_threshold, and vtu_threshold are required when "
                "add_nature is true"
            )
        tracks[nature_name] = (
            "record",
            nature(
                track_id=tracks.track_id.values,
                b=tracks[b_name].values,
                vtl=tracks[vtl_name].values,
                vtu=tracks[vtu_name].values,
                vortex=tracks[closed_mslp_name].notnull().values,
                wcsi=istc,
                b_threshold=b_threshold,
                vtl_threshold=vtl_threshold,
                vtu_threshold=vtu_threshold,
                filter_size=filter_size,
                min_count=min_count,
                et=et,
                smooth=smooth,
            ),
        )

    if drop:
        tracks = tracks.hrcn.sel_id(np.unique(tracks.track_id.values[istc]))

    huracanpy.save(tracks, str(filename_out))


def variable_values(
    tracks: xr.Dataset, variable_name: str | None
) -> NDArray[np.floating] | None:
    """Return a named track variable as a NumPy array, or disable the criterion."""
    if variable_name is None:
        return None
    return tracks[variable_name].values


def additional_filter_mask(
    tracks: xr.Dataset,
    *,
    filter_all: list[str] | None,
    basin: str | None,
    closed_mslp_name: str,
) -> NDArray[np.bool_] | None:
    """Combine optional basin, closed-MSLP, and ocean point masks."""
    criteria = []
    if basin is not None:
        criteria.append(tracks.hrcn.get_basin().values == basin)
    for name in filter_all or []:
        if name == "closed_mslp":
            criteria.append(tracks[closed_mslp_name].notnull().values)
        elif name == "ocean":
            criteria.append(tracks.hrcn.get_is_ocean().values)

    if not criteria:
        return None
    return np.logical_and.reduce(criteria)


if __name__ == "__main__":
    app()
