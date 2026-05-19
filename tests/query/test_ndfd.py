"""Offline NDFD query test.

The NDFD parser (`pd.read_csv` on a whitespace-separated text response) is
not exercised offline — schema-change detection against the live endpoint
is the job of ``tests/schema_live/test_live_ndfd.py``. Here we replay a
snapshot of the parsed DataFrame, instantiate ``NdfdQuery`` with the live
fetch monkeypatched out, and validate the result against the pandera
contract.
"""

import pandas as pd

from site_forecast.query import ndfd as ndfd_mod
from site_forecast.query.ndfd import NdfdQuery

from schemas.queries import NdfdSchema


def test_ndfd_fixture_satisfies_schema(dataframes_dir):
    df = pd.read_parquet(dataframes_dir / "ndfd.parquet")
    NdfdSchema.validate(df)


def test_ndfd_query_replays_fixture_and_is_okay(monkeypatch, dataframes_dir):
    fixture = pd.read_parquet(dataframes_dir / "ndfd.parquet")
    monkeypatch.setattr(ndfd_mod, "query_weather", lambda url=ndfd_mod.URL: fixture)

    query = NdfdQuery()

    assert query.okay
    NdfdSchema.validate(query.df)


def test_ndfd_query_okay_false_when_fetch_returns_none(monkeypatch):
    monkeypatch.setattr(ndfd_mod, "query_weather", lambda url=ndfd_mod.URL: None)
    query = NdfdQuery()
    assert not query.okay
    assert query.df is None


def test_ndfd_query_save_data_writes_parquet(monkeypatch, dataframes_dir, tmp_path):
    fixture = pd.read_parquet(dataframes_dir / "ndfd.parquet")
    monkeypatch.setattr(ndfd_mod, "query_weather", lambda url=ndfd_mod.URL: fixture)
    query = NdfdQuery()

    out_root = tmp_path / "forecasts"
    monkeypatch.setattr(
        NdfdQuery,
        "forecast_dir",
        property(lambda self: out_root),
    )
    query.save_data(outname="ndfd")

    written = out_root / "ndfd.parquet"
    assert written.exists()
    round_trip = pd.read_parquet(written)
    pd.testing.assert_frame_equal(round_trip, fixture)
