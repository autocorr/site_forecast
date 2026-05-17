import numpy as np
import pandas as pd
import pytest
from astropy import units as u

from site_forecast.sensitivity import (
    _band_averages,
    _interp_pres_to_times,
    _make_cloud_array,
)


BANDS = {
    "L": (1.0, 2.0),
    "S": (2.0, 4.0),
    "C": (4.0, 8.0),
    "X": (8.0, 12.0),
    "U": (12.0, 18.0),
    "K": (18.0, 26.5),
}


def _freq_df(freq, **cols):
    return pd.DataFrame(cols, index=pd.Index(freq, name="frequency"))


def test_band_averages_indexes_by_band_and_preserves_columns():
    df = _freq_df(
        [1.5, 3.0, 5.0, 10.0, 25.0],
        opacity=[0.1, 0.2, 0.3, 0.4, 0.5],
        transmittance=[0.9, 0.8, 0.7, 0.6, 0.5],
    )
    out = _band_averages(df, BANDS)
    assert out.index.name == "band"
    assert list(out.index) == list(BANDS.keys())
    assert list(out.columns) == ["opacity", "transmittance"]


def test_band_averages_means_within_band_and_empty_band_is_nan():
    df = _freq_df(
        [1.5, 2.5, 3.5, 5.0],
        opacity=[1.0, 3.0, 5.0, 7.0],
    )
    out = _band_averages(df, BANDS)
    # L band contains only 1.5 -> mean is 1.0
    assert out.loc["L", "opacity"] == pytest.approx(1.0)
    # S band contains 2.5 and 3.5 -> opacities 3.0 and 5.0 -> mean is 4.0
    assert out.loc["S", "opacity"] == pytest.approx(4.0)
    # X band has no samples -> NaN
    assert np.isnan(out.loc["X", "opacity"])


def test_band_averages_lower_inclusive_upper_exclusive():
    # 2.0 GHz sits exactly on the L/S boundary; the mask is `>= lo & < hi`,
    # so it belongs to S, not L.
    df = _freq_df([2.0], opacity=[42.0])
    out = _band_averages(df, BANDS)
    assert np.isnan(out.loc["L", "opacity"])
    assert out.loc["S", "opacity"] == pytest.approx(42.0)


def test_make_cloud_array_places_density_at_nearest_layer():
    pressure = np.array([1000, 800, 600, 400, 200]) * u.hPa
    sigma = 0.5 * u.kg / u.m**2
    arr = _make_cloud_array(pressure, sigma, target_pressure_level=620 * u.hPa)
    expected = np.array([0.0, 0.0, 0.5, 0.0, 0.0]) * (u.kg / u.m**2)
    assert arr.unit == expected.unit
    np.testing.assert_array_equal(arr.value, expected.value)


def test_make_cloud_array_length_matches_pressure_grid():
    pressure = np.linspace(1000, 100, 19) * u.hPa
    arr = _make_cloud_array(pressure, 1.0 * u.kg / u.m**2)
    assert arr.shape == pressure.shape


def _make_pres_frame(times, pressures, temp_values, rh_values):
    idx = pd.MultiIndex.from_product([times, pressures], names=["date", "pressure"])
    return pd.DataFrame(
        {"temperature": temp_values, "relative_humidity": rh_values},
        index=idx,
    )


def test_interp_pres_to_times_interpolates_at_midpoints():
    times = pd.date_range("2025-01-01", periods=3, freq="1h")
    pressures = [1000, 850, 700]
    df = _make_pres_frame(
        times,
        pressures,
        temp_values=np.arange(9, dtype=float),
        rh_values=np.arange(9, dtype=float) * 10,
    )
    # Callers in `sensitivity.py` always pass a date-named index; mirror that
    # so the output index has a proper "date" level name after stacking.
    new_times = pd.DatetimeIndex(
        pd.date_range("2025-01-01 00:30", periods=2, freq="1h"), name="date"
    )
    out = _interp_pres_to_times(df, new_times)

    assert out.index.names == ["date", "pressure"]
    assert list(out.index.get_level_values("date").unique()) == list(new_times)
    # At 00:30, pressure=1000 sits midway between t=0 (temp=0) and t=1 (temp=3).
    assert out.loc[(new_times[0], 1000), "temperature"] == pytest.approx(1.5)
    assert out.loc[(new_times[0], 1000), "relative_humidity"] == pytest.approx(15.0)
    # At 01:30, pressure=850 sits between t=1 (temp=4) and t=2 (temp=7).
    assert out.loc[(new_times[1], 850), "temperature"] == pytest.approx(5.5)


def test_interp_pres_to_times_returns_only_requested_times():
    times = pd.date_range("2025-01-01", periods=4, freq="1h")
    pressures = [1000, 500]
    df = _make_pres_frame(
        times,
        pressures,
        temp_values=np.arange(8, dtype=float),
        rh_values=np.arange(8, dtype=float),
    )
    new_times = pd.DatetimeIndex(["2025-01-01 00:15", "2025-01-01 02:45"], name="date")
    out = _interp_pres_to_times(df, new_times)
    assert list(out.index.get_level_values("date").unique()) == list(new_times)
