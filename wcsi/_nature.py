from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ragged import padded_index, uniform_filter1d


def nature(
    track_id: NDArray[Any],
    b: NDArray[np.floating],
    vtl: NDArray[np.floating],
    vtu: NDArray[np.floating],
    vortex: NDArray[np.bool],
    wcsi: NDArray[np.bool],
    *,
    b_threshold: float = 15,
    vtl_threshold: float = 0,
    vtu_threshold: float = 0,
    filter_size=5,
    min_count: int = 4,
    et: bool = False,
    smooth: bool = False,
) -> NDArray[np.str_]:
    """Derive a nature tag from the cyclone structure

    - ``"TC"`` - Tropical Cyclone
    - ``"Vo"`` - Vortex
    - ``"BC"`` - Baroclinic
    - ``"Tr"`` - Trough
    - ``"MV"`` - Mid-level vortex
    - ``"Ot"`` - Other

    If ``et=True``

    - ``"ET"`` - Extratropical transition
    - ``"WS"`` - Warm seclusion

    Parameters
    ----------
    b
        Cyclone phase space asymmetry
    vtl
        Cyclone phase space low-level warm core
    vtu
        Cyclone phase space upper-level warm core
    vortex
        Array saying whether each point is a vortex or cyclone. e.g. is there an
        associated closed MSLP contour
    wcsi
        Points previously used to identify the cyclone as tropical cyclone
    b_threshold
        The threshold of the asymmetry parameter, below which is considered to be a
        tropical cyclone
    vtl_threshold
        The threshold of the low-level warm-core parameter, above which is considered
        to be a tropical cyclone
    vtu_threshold
        The threshold of the upper-level warm-core parameter, above which is considered
        to be a tropical cyclone
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
    nat = np.zeros(len(wcsi), dtype="U2")

    # No closed MSLP - vortex
    nat[vortex] = "Vo"

    # WCSI label as tropical cyclone
    # Any gaps less that min_count fill in as TC also
    wcsi, start_wcsi, end_wcsi = smooth_excursions(track_id, wcsi, min_count=min_count)
    nat[wcsi] = "TC"

    # Other CPS categories
    # Apply filtering
    if filter_size is not None and filter_size > 1:
        idx, inverse = padded_index(
            track_id, pad=filter_size // 2, pad_location="outer"
        )

        if b is not None:
            b = uniform_filter1d(
                track_id, np.abs(b), size=filter_size, idx=idx, inverse=inverse
            )
        if vtl is not None:
            vtl = uniform_filter1d(
                track_id, vtl, size=filter_size, idx=idx, inverse=inverse
            )
        if vtu is not None:
            vtu = uniform_filter1d(
                track_id, vtu, size=filter_size, idx=idx, inverse=inverse
            )

    symmetric = b <= b_threshold
    warm_core = vtl > vtl_threshold
    trough = vtu <= vtu_threshold

    # Any Warm core/symmetric periods adjacent to TC are also TC
    # First fill in any WCS gaps less than min_count
    wcs = (nat == "") & symmetric & warm_core
    wcs, start_wcs, end_wcs = smooth_excursions(track_id, wcs, min_count=min_count)

    # Look for sequences where WCSI joins WCS
    # Since WCS and WCSI has already filled gaps in, any new sequences found will be
    # joins between WCS and WCSI
    start_wcs = np.concat([start_wcsi, start_wcs])
    end_wcs = np.concat([end_wcsi, end_wcs])
    idx = np.argsort(start_wcs)
    joined, start_wcs, end_wcs = smooth_excursions(
        track_id,
        np.zeros(len(track_id), dtype=bool),
        min_count=min_count,
        starts=start_wcs[idx],
        ends=end_wcs[idx],
    )
    nat[joined] = "TC"

    # Label remaining unlabelled sections
    # Asymmetric = baroclinic
    nat[(nat == "") & ~symmetric] = "BC"
    # Upper-level cold core = Trough
    nat[(nat == "") & trough] = "Tr"
    # Low-level cold core = Mid level vortex
    nat[(nat == "") & ~warm_core] = "MV"
    # Warm core symmetric not TC (decaying)
    nat[(nat == "")] = "Ot"

    return nat


def smooth_excursions(track_id, criteria, *, min_count, starts=None, ends=None):
    # +1 to start_tc index because we want the index of the first TC point not the index
    # of the last non-TC point
    # Where do cyclones start/end being TC and start/end being WCS but not TC
    if starts is None:
        starts = np.where(~criteria[:-1] & criteria[1:])[0] + 1
    if ends is None:
        ends = np.where(criteria[:-1] & ~criteria[1:])[0]

    # Exceptions
    # First point is a TC point and is missed
    if starts[0] > ends[0]:
        starts = np.concat([[0], starts])

    # Last point is a TC point, so end point is missed
    if len(starts) > len(ends):
        ends = np.concat([ends, [len(criteria) - 1]])

    # Look for points where the last point of a TC is within min_count of the next TC
    # identification for the same track_id
    close = (
        np.where(
            ((starts[1:] - ends[:-1]) < min_count)
            & (track_id[starts[1:]] == track_id[ends[:-1]])
        )[0]
        + 1
    )

    # idx of all joined sequences
    # Start of first sequence to end of second sequence
    idx = np.concat(
        [
            np.arange(start, end + 1)
            for start, end in zip(starts[close - 1], ends[close])
        ]
    )

    # Update start and end indices to remove joined sections
    starts = starts[np.concat([[True], ~close])]
    ends = ends[np.concat([~close, [True]])]

    new_criteria = criteria.copy()
    new_criteria[idx] = True

    return new_criteria, starts, ends
