"""Offline open-meteo query tests.

Cassettes replay the FlatBuffers binary responses (stored base64 in YAML) so
the real parsers (`parse_response`, `unpivot_pressure_levels`,
`parse_ensemble_response`) execute against representative data. VCR
intercepts at urllib3; the `bypass_openmeteo_cache` fixture swaps the
module-level cached/retry sessions for a plain `requests.Session` so
requests-cache doesn't short-circuit the call below where VCR can see it.

Schema-change detection against the live endpoints is the job of
`tests/schema_live/test_live_open_meteo.py`; here we only pin the current
structural contract and wrapper behavior.
"""

import pandas as pd
import pytest
from darts import TimeSeries

from site_forecast.query.open_meteo import (
    OpenMeteoMultiSiteQuery,
    OpenMeteoVlaEnsembleQuery,
    OpenMeteoVlaPressureQuery,
    OpenMeteoVlaQuery,
)

from schemas.queries import (
    OpenMeteoMultiSiteSchema,
    OpenMeteoVlaEnsembleSchema,
    OpenMeteoVlaPressureSchema,
    OpenMeteoVlaSchema,
)


@pytest.mark.vcr
@pytest.mark.cassette("open_meteo_vla")
def test_vla_query_replays_and_is_okay(bypass_openmeteo_cache):
    q = OpenMeteoVlaQuery()
    assert q.okay
    OpenMeteoVlaSchema.validate(q.df)


@pytest.mark.vcr
@pytest.mark.cassette("open_meteo_vla")
def test_vla_query_to_model_series_is_resampled_to_15min(bypass_openmeteo_cache):
    q = OpenMeteoVlaQuery()
    ts = q.to_model_series()
    assert isinstance(ts, TimeSeries)
    assert ts.freq_str.lower().startswith("15")


@pytest.mark.vcr
@pytest.mark.cassette("open_meteo_vla")
def test_vla_query_save_data_writes_parquet(
    bypass_openmeteo_cache, monkeypatch, tmp_path
):
    q = OpenMeteoVlaQuery()
    out_root = tmp_path / "forecasts"
    monkeypatch.setattr(
        OpenMeteoVlaQuery, "forecast_dir", property(lambda self: out_root)
    )
    q.save_data(outname="weather")

    written = out_root / "weather.parquet"
    assert written.exists()
    pd.read_parquet(written)


@pytest.mark.vcr
@pytest.mark.cassette("open_meteo_pressure")
def test_pressure_query_replays_and_is_okay(bypass_openmeteo_cache):
    q = OpenMeteoVlaPressureQuery()
    assert q.okay
    OpenMeteoVlaPressureSchema.validate(q.df)


@pytest.mark.vcr
@pytest.mark.cassette("open_meteo_ensemble")
def test_ensemble_query_replays_and_is_okay(bypass_openmeteo_cache):
    q = OpenMeteoVlaEnsembleQuery()
    assert q.okay
    OpenMeteoVlaEnsembleSchema.validate(q.df)


@pytest.mark.vcr
@pytest.mark.cassette("open_meteo_multi_site")
def test_multi_site_query_replays_and_is_okay(bypass_openmeteo_cache):
    q = OpenMeteoMultiSiteQuery()
    assert q.okay
    OpenMeteoMultiSiteSchema.validate(q.df)
