from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve1d


def mask_short(
    track_id: NDArray[Any],
    istc: NDArray[np.bool_],
    min_count: int = 4,
) -> NDArray[np.bool_]:
    """Remove short consecutive ``True`` runs independently within each track.

    A run is retained only when it contains at least ``min_count`` consecutive
    elements. Track boundaries always terminate a run, even when the last value of
    one track and the first value of the next track are both ``True``.

    Parameters
    ----------
    track_id
        Identifier for the track containing each point. Points belonging to a track
        must be contiguous. Track identifiers may be any values accepted by
        :func:`numpy.unique`. Must have shape ``(n_points,)``.
    istc
        Boolean criteria evaluated at each track point. ``True`` runs shorter than
        ``min_count`` are changed to ``False``. Must have shape ``(n_points,)``.
    min_count
        Minimum number of consecutive ``True`` points required to retain a run.

    Returns
    -------
    numpy.ndarray of bool, shape (n_points,)
        A copy of the Boolean criteria with short runs removed. The result has the
        same length and point order as the inputs.

    Notes
    -----
    This function treats the input as a contiguous ragged array: all points for one
    track must occur together. It checks adjacency in array order; it does not verify
    that observation times are consecutive.

    Examples
    --------
    Runs do not continue across a change in track identifier:

    >>> track_id = np.array(["a", "a", "a", "b", "b", "b"])
    >>> criteria = np.array([False, True, True, True, True, False])
    >>> mask_short(track_id, criteria, min_count=3)
    array([False, False, False, False, False, False])
    """
    # First and last index of sequences where istc=True
    # Must use an array padded with zeros, otherwise a starting or ending
    # True will be missed
    _, idx = np.unique(track_id, return_index=True)
    idx = np.concat([idx, [len(track_id)]])
    istc_pad = np.insert(istc, idx, False)

    start = np.where(np.diff(istc_pad.astype(int)) == 1)[0] + 1
    end = np.where(np.diff(istc_pad.astype(int)) == -1)[0]
    length = end - start + 1
    short = length < min_count

    # Index of where to set istc=True to false instead
    if short.any():
        idx_short = np.concat(
            [np.arange(s, e + 1) for s, e in zip(start[short], end[short])]
        )
        istc_pad[idx_short] = False

    # Remove padding values
    return np.delete(istc_pad, idx + np.arange(len(idx)))


def padded_index(
    track_id: NDArray[Any],
    pad: int = 1,
    pad_location: Literal["inner", "outer", "start", "end"] = "inner",
    pad_method: str = "constant",
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Construct indices for padding each track in a contiguous ragged array.

    The returned index repeats boundary indices rather than allocating a padded
    values array. Applying it as ``values[idx]`` therefore pads using each track's
    boundary values. The accompanying Boolean mask selects the original, unpadded
    positions from the padded result.

    Parameters
    ----------
    track_id
        Identifier for the track containing each point. Points belonging to each
        track must be contiguous. Must have shape ``(n_points,)``.
    pad
        Number of repeated boundary values to insert at each selected boundary.
    pad_location
        Boundaries at which padding is inserted:

        - ``"inner"``
          Pad both sides of every boundary between adjacent tracks, but not the
          beginning or end of the complete array.
        - ``"outer"``
          Pad both sides of internal boundaries and both ends of the complete
          array. This pads the start and end of every track.
        - ``"start"``
          Pad only the first point of every track.
        - ``"end"``
          Pad only the last point of every track.
    pad_method
        Reserved for selecting a padding method. It is currently unused; padding
        always repeats the appropriate boundary index.

    Returns
    -------
    idx : numpy.ndarray of int
        Indices into the original array that construct the padded representation.
        For an input ``values``, ``values[idx]`` contains the repeated boundary
        values.
    inverse : numpy.ndarray of bool
        Boolean mask with the same length as ``idx``. Applying this mask to a result
        calculated on the padded representation restores the original number and
        ordering of points.

    Notes
    -----
    This helper assumes track IDs are grouped contiguously. It does not reorder the
    data or detect later, non-contiguous occurrences of the same identifier.

    Examples
    --------
    >>> track_id = np.array(["a", "a", "b", "b"])
    >>> idx, inverse = padded_index(track_id, pad=1, pad_location="outer")
    >>> idx
    array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> np.arange(4)[idx]
    array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> np.arange(4)[idx][inverse]
    array([0, 1, 2, 3])
    """
    # Get the locations of where to add padded vales
    # idx_pad = First index of each track
    _, idx_pad = np.unique(track_id, return_index=True)

    # Boolean array to get back to original length array from padded array
    inverse = np.ones(len(track_id), dtype=bool)

    if pad_location in ["inner", "outer"]:
        # Inner indices are always padded with twice the pad amount (start and end of tracks)
        # Apply to inverse first otherwise it is always inserted to the left of the
        # index (needs to be right of start, left of end)
        idx_pad = idx_pad[1:]
        inverse = np.insert(inverse, np.repeat(idx_pad, pad * 2), False)
        idx_pad = np.sort(np.concat([idx_pad, idx_pad - 1]))

        if pad_location == "outer":
            idx_pad = np.concat([[0], idx_pad, [len(track_id) - 1]])
            inverse = np.concat([[False] * pad, inverse, [False] * pad])

    elif pad_location in ["start", "end"]:
        if pad_location == "end":
            # Shift to last point of each track
            idx_pad = np.concat([idx_pad[1:] - 1, [len(track_id) - 1]])
        inverse = np.insert(inverse, np.repeat(idx_pad, pad), False)

    # Incread padding width
    idx_pad = np.repeat(idx_pad, pad)

    idx = np.insert(np.arange(len(track_id)), idx_pad, idx_pad)

    return idx, inverse


def uniform_filter1d(
    track_id: NDArray[Any],
    input: NDArray[Any],
    size: int,
    *,
    idx: NDArray[np.intp] | None = None,
    inverse: NDArray[np.bool_] | None = None,
) -> NDArray[Any]:
    """Apply a one-dimensional running mean separately to each track.

    Track boundaries are padded with the nearest value before convolution, preventing
    values from adjacent tracks from contributing to one another. Precomputed padding
    indices may be supplied when filtering several arrays with the same track layout
    and window size.

    Parameters
    ----------
    track_id
        Identifier for the track containing each point. Points belonging to each
        track must be contiguous. Must have shape ``(n_points,)``.
    input
        Values to smooth. The first dimension must correspond to ``track_id``.
        A NumPy representation is passed to the convolution routine.
    size
        Width of the running-mean window in points. Odd window sizes are recommended
        because padding uses ``size // 2`` points on each side.
    idx
        Precomputed padded index returned by :func:`padded_index`.
    inverse
        Precomputed mask returned by :func:`padded_index`, used to remove padding
        after convolution. If either ``idx`` or ``inverse`` is omitted, both are
        recalculated.

    Returns
    -------
    numpy.ndarray
        Running means in the original point order and with the same length as
        ``track_id``. The result is floating point for typical numeric inputs.

    Notes
    -----
    The calculation uses :func:`scipy.ndimage.convolve1d` with a boxcar kernel and
    nearest-value padding at track boundaries. Unlike
    :func:`scipy.ndimage.uniform_filter1d`, a non-finite value affects only windows
    that overlap that value rather than propagating recursively through the remainder
    of the array. This is not a NaN-omitting mean: any window containing ``NaN``
    normally produces ``NaN``.

    If ``size`` is even, ``size // 2`` values are added to both ends and SciPy's
    default convolution origin determines the alignment. Use an odd size for a
    running mean centred unambiguously on each point.

    Examples
    --------
    >>> track_id = np.array(["a", "a", "a", "b", "b", "b"])
    >>> values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    >>> uniform_filter1d(track_id, values, size=3)
    array([ 1.33333333,  2.        ,  2.66666667, 13.33333333, 20.        ,
           26.66666667])
    """
    if idx is None or inverse is None:
        idx, inverse = padded_index(track_id, size // 2, pad_location="outer")

    # Use convolve 1d instead of uniform_filter1d
    # Running mean in uniform_filter1d means NaNs or large masked values ruin the
    # result for the rest of the array
    # (see https://github.com/scipy/scipy/issues/7818)
    # Enforce numpy array. It is much slower to pass an xarray Dataset
    return convolve1d(np.asarray(input)[idx], np.ones(size))[inverse] / size
