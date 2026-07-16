"""Live schema-drift tests for the VLA monitor database queries.

Runs the real ``ApiQuery`` and ``WeatherStationQuery`` against the monitor
PostgreSQL database and validates the parsed DataFrames against their pandera
contracts. Gates on ``df is not None`` rather than ``okay``: ``okay`` also
requires ``not has_bad``, which can be transiently true on live data even when
the structure is intact — the schema check only needs a valid DataFrame. Skips
unless the ``[Monitor]`` config section is present and the DB is reachable. Run
with ``pytest -m schema``.
"""

import pytest

from site_forecast.query.monitor import ApiQuery, WeatherStationQuery

from schemas.queries import ApiQuerySchema, WeatherStationSchema

pytestmark = pytest.mark.schema


def test_live_api(require_monitor):
    q = ApiQuery()
    assert q.df is not None
    ApiQuerySchema.validate(q.df)


def test_live_weather_station(require_monitor):
    q = WeatherStationQuery()
    assert q.df is not None
    WeatherStationSchema.validate(q.df)
