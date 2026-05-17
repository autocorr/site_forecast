import pandas as pd
import pytest
from astropy.coordinates import Latitude, Longitude
from darts import TimeSeries

from site_forecast.query import (
    normalize_time,
    timeseries_from_dataframe,
    wrap_coordinates,
)


def test_normalize_time_converts_to_utc_and_drops_subhour_precision():
    t = pd.Timestamp("2025-06-15T13:37:42", tz="America/Denver")
    out = normalize_time(t)
    # America/Denver in mid-June is UTC-6 (DST), so 13:37 local -> 19:37 UTC,
    # then truncated to the hour.
    assert out == pd.Timestamp(2025, 6, 15, 19)
    assert out.tz is None


def test_normalize_time_zero_minutes_when_already_on_hour():
    t = pd.Timestamp("2025-06-15T18:00:00", tz="utc")
    out = normalize_time(t)
    assert out == pd.Timestamp(2025, 6, 15, 18)


def test_wrap_coordinates_from_real_floats():
    lat, lon = wrap_coordinates(34.0773880, -107.6156450)
    assert isinstance(lat, Latitude)
    assert isinstance(lon, Longitude)
    assert lat.deg == pytest.approx(34.0773880)
    # Longitude wraps negatives into [0, 360).
    assert lon.deg == pytest.approx(360 - 107.6156450)


def test_wrap_coordinates_passes_through_astropy_objects():
    lat_in = Latitude(34.0, unit="deg")
    lon_in = Longitude(-107.0, unit="deg")
    lat, lon = wrap_coordinates(lat_in, lon_in)
    assert isinstance(lat, Latitude)
    assert isinstance(lon, Longitude)
    assert lat.deg == pytest.approx(34.0)
    assert lon.deg == pytest.approx(253.0)


def test_timeseries_from_dataframe_smoke_returns_timeseries():
    idx = pd.date_range("2025-01-01", periods=10, freq="15min")
    df = pd.DataFrame({"a": range(10)}, index=idx)
    ts = timeseries_from_dataframe(df)
    assert isinstance(ts, TimeSeries)
    assert len(ts) == 10


def test_timeseries_from_dataframe_strips_tz_without_mutating_input():
    idx = pd.date_range("2025-01-01", periods=8, freq="1h", tz="utc")
    df = pd.DataFrame({"a": range(8)}, index=idx)
    ts = timeseries_from_dataframe(df)

    # The returned darts TimeSeries should be tz-naive.
    ts_idx = ts.time_index
    assert getattr(ts_idx, "tz", None) is None

    # The caller's frame must not be mutated.
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
