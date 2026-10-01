from itertools import groupby

import numpy as np
from numpy.typing import NDArray


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

    - ``"TC"`` - Tropical Cyclone
    - ``"Vo"`` - Weak Vortex
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
        _nat, count = nat_consecutive[m]
        if count < min_count and nat_consecutive[m - 1][0] == nat_consecutive[m + 1][0]:
            new_nat = nat_consecutive[m - 1][0]
            nat[idx : idx + count] = new_nat
            nat_consecutive[m] = (new_nat, count)

        idx += count
