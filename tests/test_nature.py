import pytest

import numpy as np

from wcsi._nature import nature, smooth_excursions


@pytest.mark.parametrize(
    "track_id",
    [
        # One TC track
        np.zeros(8),
        # Two tracks
        np.array([0] * 4 + [1] * 4),
    ],
)
@pytest.mark.parametrize(
    "is_wcsi",
    [
        # single TC phase
        np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=bool),
        # Single TC phase starting at zero
        np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        # TC phase with a gap
        np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=bool),
        # No TC phases
        np.zeros(8, dtype=bool),
    ],
)
def test_smooth_excursions(track_id, is_wcsi):
    result, starts, ends = smooth_excursions(track_id, is_wcsi, min_count=4)

    # Check that the result is unchange when re-run
    result_2, starts_2, ends_2 = smooth_excursions(
        track_id, result, starts=starts, ends=ends, min_count=4
    )

    for pair in [(result, result_2), (starts, starts_2), (ends, ends_2)]:
        assert np.all(pair[0] == pair[1])


def test_nature():
    # Check that a WCS region is connected to a WCSI region
    nat = nature(
        track_id=np.zeros(10),
        b=np.zeros(10),
        vtl=np.ones(10),
        vtu=np.array([-1] * 5 + [1] * 5),
        vortex=np.zeros(10, dtype=bool),
        wcsi=np.array([False] * 5 + [True] * 5),
    )

    assert (nat == "TC").all()