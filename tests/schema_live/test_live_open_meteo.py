"""Live schema-drift tests for the open-meteo queries.

Hits the real API and validates each query's DataFrame against the same
pandera contract the offline cassette tests use — the primary mechanism for
detecting upstream structural change. Run with ``pytest -m schema``.
"""

import pytest

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

pytestmark = pytest.mark.schema


def test_live_vla(require_open_meteo):
    q = OpenMeteoVlaQuery()
    assert q.okay
    OpenMeteoVlaSchema.validate(q.df)


def test_live_pressure(require_open_meteo):
    q = OpenMeteoVlaPressureQuery()
    assert q.okay
    OpenMeteoVlaPressureSchema.validate(q.df)


def test_live_ensemble(require_open_meteo):
    q = OpenMeteoVlaEnsembleQuery()
    assert q.okay
    OpenMeteoVlaEnsembleSchema.validate(q.df)


def test_live_multi_site(require_open_meteo):
    q = OpenMeteoMultiSiteQuery()
    assert q.okay
    OpenMeteoMultiSiteSchema.validate(q.df)
