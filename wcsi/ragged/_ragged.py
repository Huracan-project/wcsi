import numpy as np
from scipy.ndimage import convolve1d


def mask_short(track_id, istc, min_count=4):
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


def padded_index(track_id, pad=1, pad_location="inner", pad_method="constant"):
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


def uniform_filter1d(track_id, input, size, *, idx=None, inverse=None):
    if idx is None or inverse is None:
        idx, inverse = padded_index(track_id, size // 2, pad_location="outer")

    # Use convolve 1d instead of uniform_filter1d
    # Running mean in uniform_filter1d means NaNs or large masked values ruin the
    # result for the rest of the array
    # (see https://github.com/scipy/scipy/issues/7818)
    # Enforce numpy array. It is much slower to pass an xarray Dataset
    return convolve1d(np.asarray(input)[idx], np.ones(size))[inverse] / size