"""Live schema-drift test for the NDFD forecast query.

The NDFD text parser is not exercised offline (the fixture stores the parsed
DataFrame), so this live test against the real endpoint is what catches an
upstream change to the whitespace-separated forecast table. Run with
``pytest -m schema``.
"""

import pytest

from site_forecast.query.ndfd import NdfdQuery

from schemas.queries import NdfdSchema

pytestmark = pytest.mark.schema


def test_live_ndfd(require_ndfd):
    q = NdfdQuery()
    assert q.okay
    NdfdSchema.validate(q.df)
