import numpy as np
import pytest

from wcsi._wcsi import wcsi


@pytest.mark.parametrize("parameter", ["b", "vtl", "vtu"])
@pytest.mark.parametrize("filter_size", [None, 3])
def test_cps_legacy_thresholds(parameter, filter_size):
    values = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    kwargs = {f"cps_{parameter}": values, "filter_size": filter_size, "npoints": 1}
    legacy = wcsi(np.zeros(5), **kwargs, **{f"{parameter}_threshold": 2})
    bound = "max" if parameter == "b" else "min"
    explicit = wcsi(np.zeros(5), **kwargs, **{f"{parameter}_{bound}": 2})
    np.testing.assert_array_equal(legacy, explicit)


@pytest.mark.parametrize("parameter", ["b", "vtl", "vtu"])
def test_cps_bounds_override_threshold(parameter):
    result = wcsi(
        np.zeros(5),
        **{
            f"cps_{parameter}": np.array([-2.0, 0.0, 2.0, 4.0, 6.0]),
            f"{parameter}_threshold": -100,
            f"{parameter}_min": 0,
            f"{parameter}_max": 4,
        },
        b_abs=False,
        npoints=1,
        filter_size=None,
    )
    np.testing.assert_array_equal(result, [False, False, True, True, False])


@pytest.mark.parametrize("parameter", ["b", "vtl", "vtu"])
def test_cps_disabled_threshold(parameter):
    result = wcsi(
        np.zeros(3),
        **{
            f"cps_{parameter}": np.array([-100.0, 0.0, 100.0]),
            f"{parameter}_threshold": None,
        },
        npoints=1,
        filter_size=None,
    )
    assert result.all()


def test_cps_missing_values_with_bounds():
    assert wcsi(np.zeros(4), b_min=0, vtl_max=10, vtu_max=10).all()


@pytest.mark.parametrize("filter_size", [None, 1, 3])
@pytest.mark.parametrize("b_abs", [False, True])
def test_cps_b_absolute_value(filter_size, b_abs):
    result = wcsi(
        np.zeros(5),
        cps_b=np.full(5, -20.0),
        b_abs=b_abs,
        filter_size=filter_size,
        npoints=1,
    )
    np.testing.assert_array_equal(result, np.full(5, not b_abs))


def test_cps_b_absolute_value_before_smoothing():
    result = wcsi(
        np.zeros(5),
        cps_b=np.array([-20.0, 20.0, -20.0, 20.0, -20.0]),
        filter_size=3,
        npoints=1,
    )
    assert not result.any()
