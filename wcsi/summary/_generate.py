"""Generate track-level summaries from point-level cyclone data."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import xarray as xr


_Array1D = NDArray[Any] | xr.DataArray


def generate(
    track_id: _Array1D,
    time: _Array1D,
    lon: _Array1D,
    lat: _Array1D,
    wcsi: NDArray[np.bool_] | xr.DataArray,
) -> pd.DataFrame:
    """Summarise point-level cyclone data by track.

    One output row is created for each unique track identifier. The row contains the
    first and last track coordinates and times, together with a flag indicating
    whether any point in that track was classified by WCSI.

    Parameters
    ----------
    track_id
        Identifier for the track containing each point. Track points must be grouped
        contiguously, and all input arrays must use the same point order. Must have
        shape ``(n_points,)``.
    time
        Observation time for each point, with shape ``(n_points,)``.
    lon
        Longitude of each point, with shape ``(n_points,)``.
    lat
        Latitude of each point, with shape ``(n_points,)``.
    wcsi
        Boolean WCSI classification for each point, with shape ``(n_points,)``. A
        track is marked as a tropical cyclone when this is ``True`` for at least one
        of its points.

    Returns
    -------
    pandas.DataFrame
        Track-level table with the following columns:

        - ``track_id``
          Unique track identifier.
        - ``storm_start``, ``storm_end``
          Times at the track endpoints.
        - ``origin_lon``, ``origin_lat``
          Coordinates at the first track point.
        - ``end_lon``, ``end_lat``
          Coordinates at the last track point.
        - ``is_tc``
          Whether the track contains at least one WCSI-classified point.

    Notes
    -----
    Track identifiers are returned in the order produced by :func:`numpy.unique`.
    The function assumes each track occupies one contiguous block and does not sort
    observations by time.
    """
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
