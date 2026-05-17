from pathlib import Path

import pandas as pd
import pytest
from astropy.time import Time

from site_forecast import SITES_BY_NAME, _now_dir


def test_now_dir_utc_aware():
    t = pd.Timestamp("2025-06-15T13:30:00", tz="utc")
    assert _now_dir(t) == Path("2025/06/15/13")


def test_now_dir_zero_padding():
    t = pd.Timestamp("2025-01-02T00:30:00", tz="utc")
    assert _now_dir(t) == Path("2025/01/02/00")


def test_now_dir_naive_timestamp():
    t = pd.Timestamp("2025-06-15T13:30:00")
    assert _now_dir(t) == Path("2025/06/15/13")


def test_now_dir_default_is_now(monkeypatch):
    fixed = pd.Timestamp("2025-06-15T13:30:00", tz="utc")

    class _FakeTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr("site_forecast.pd.Timestamp", _FakeTimestamp)
    assert _now_dir() == Path("2025/06/15/13")


def _as_iso(time_list):
    return [Time(t).iso for t in time_list]


def test_sun_rise_and_sets_vla_summer_snapshot():
    """Regression snapshot of sun rise/set times for VLA on 2025-06-15.

    Values are used only for shading day/night on plots, not for analysis,
    so we capture the current astropy output and watch for accidental drift.
    """
    vla = SITES_BY_NAME["Y1"]
    t = pd.Timestamp("2025-06-15T12:00:00", tz="utc")
    rises, sets = vla.sun_rise_and_sets(t, delta="1d")
    assert _as_iso(rises) == [
        "2025-06-14 12:10:00.000",
        "2025-06-15 12:10:00.000",
    ]
    assert _as_iso(sets) == [
        "2025-06-14 12:00:00.000",
        "2025-06-15 02:20:00.000",
        "2025-06-16 02:20:00.000",
        "2025-06-16 12:00:00.000",
    ]


def test_sun_rise_and_sets_mk_winter_snapshot():
    """Regression snapshot for Mauna Kea around winter solstice."""
    mk = SITES_BY_NAME["MK"]
    t = pd.Timestamp("2025-12-21T12:00:00", tz="utc")
    rises, sets = mk.sun_rise_and_sets(t, delta="12h")
    assert _as_iso(rises) == ["2025-12-21 17:00:00.000"]
    assert _as_iso(sets) == ["2025-12-21 03:50:00.000"]


@pytest.mark.parametrize("site_name", ["Y1", "MK", "BR", "SC"])
def test_sun_rise_and_sets_returns_within_window(site_name):
    """For non-polar sites in temperate season the rises/sets lie inside
    the queried window."""
    station = SITES_BY_NAME[site_name]
    t = pd.Timestamp("2025-06-15T12:00:00", tz="utc")
    delta = pd.Timedelta("1d")
    rises, sets = station.sun_rise_and_sets(t, delta=delta)
    lo, hi = t - delta, t + delta
    for r in rises:
        assert lo <= pd.Timestamp(r.iso, tz="utc") <= hi
    for s in sets:
        assert lo <= pd.Timestamp(s.iso, tz="utc") <= hi
