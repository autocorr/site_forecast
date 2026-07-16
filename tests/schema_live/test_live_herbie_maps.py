"""Live schema-drift test for the Herbie HRRR query.

Downloads a real HRRR run and validates the reduced Dataset against the
structural contract in ``assert_herbie_dataset``. Parametrized over one
sub-hourly product (``tcolw``, which also exercises the ``unknown``→named
rename path) and one surface product (``mcc``) to cover both code paths while
keeping the opt-in run bounded; more query types follow the same contract and
can be appended. Run with ``pytest -m schema``.
"""

import pytest

from site_forecast.query.herbie_maps import HerbieQuery

from schemas.queries import assert_herbie_dataset

pytestmark = pytest.mark.schema


@pytest.mark.parametrize("query_type", ["tcolw", "mcc"])
def test_live_herbie(require_internet, query_type):
    q = HerbieQuery(query_type=query_type)
    assert q.okay
    assert_herbie_dataset(q.ds, query_type)
