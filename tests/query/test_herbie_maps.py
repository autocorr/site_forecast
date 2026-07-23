"""Offline tests for the Herbie HRRR query.

Two layers:

1. **Pure-reduction unit tests** (no fixture, no network, no herbie accessor):
   hand-built synthetic Datasets with known-answer outputs for
   ``get_var_names``, ``geodetic_to_number``, ``subset_rectangular_region``,
   and ``add_coverage``. These carry the "expected value" assertions and
   survive fixture regeneration.
2. **Fixture-based integration test** for the ``pick_points`` reduction path:
   ``get_tcolw`` is monkeypatched to replay a tiny cached HRRR subset, so the
   real ``subset_rectangular_region`` → ``extract_position``/``extract_mean``/
   ``extract_quantiles`` → ``add_coverage`` pipeline runs offline. Assertions
   are structural (``assert_herbie_dataset``) plus reduction invariants rather
   than golden constants.

The fixture is produced by ``tests/scripts/refresh_fixtures.py herbie`` (one
HRRR download); upstream schema drift is the job of the live ``schema`` lane
(``tests/schema_live/test_live_herbie_maps.py``).
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import Latitude, Longitude

from site_forecast.query import herbie_maps as hm
from site_forecast.query.herbie_maps import HerbieQuery

from schemas.queries import assert_herbie_dataset


# ---------------------------------------------------------------------------
# Layer 1 — pure reductions with hand-built synthetic Datasets
# ---------------------------------------------------------------------------


def test_get_var_names_drops_reduction_suffixes():
    ds = xr.Dataset(
        {name: (("date",), [0.0]) for name in ("tcolw", "tcoli")}
        | {
            name: (("date",), [0.0])
            for name in ("tcolw_c", "tcolw_m", "tcolw_p", "tcolw_q")
        },
        coords={"date": [np.datetime64("2025-01-01")]},
    )
    # Only the bare quantity names survive; the _c/_m/_p/_q reductions drop out.
    assert hm.get_var_names(ds) == ["tcolw", "tcoli"]


def test_geodetic_to_number_returns_degrees_and_wraps_longitude():
    lat, lon = hm.geodetic_to_number(
        Latitude(34.0773880, "deg"), Longitude(-107.6156450, "deg")
    )
    assert lat == pytest.approx(34.0773880)
    # Longitude is wrapped into [0, 360).
    assert lon == pytest.approx(360 - 107.6156450)


def _synthetic_grid() -> xr.Dataset:
    """5x5 grid of 1-degree spacing with 2-D latitude/longitude coords."""
    lats = np.array([30, 31, 32, 33, 34.0])
    lons = np.array([250, 251, 252, 253, 254.0])
    lon2d, lat2d = np.meshgrid(lons, lats)
    return xr.Dataset(
        {"tcolw": (("y", "x"), np.arange(25.0).reshape(5, 5))},
        coords={
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
    )


def test_subset_rectangular_region_keeps_box_and_drops_outside():
    ds = _synthetic_grid()
    sub = hm.subset_rectangular_region(
        ds,
        lat=Latitude(32, "deg"),
        lon=Longitude(252, "deg"),
        lat_size=2.0,
        lon_size=2.0,
    )
    # A 2-degree box centred on (32, 252) retains the inner 3x3 block; the
    # outermost rows/cols (lat 30/34, lon 250/254) drop out entirely.
    assert dict(sub.sizes) == {"y": 3, "x": 3}
    assert float(sub.latitude.min()) == 31.0
    assert float(sub.latitude.max()) == 33.0
    assert float(sub.longitude.min()) == 251.0
    assert float(sub.longitude.max()) == 253.0


def test_subset_rectangular_region_rejects_nonpositive_size():
    ds = _synthetic_grid()
    with pytest.raises(ValueError):
        hm.subset_rectangular_region(ds, lat_size=0.0)


def test_add_coverage_threshold_crossing_known_answer():
    """``add_coverage`` walks the quantiles low→high and reports ``1 - q`` at
    the first quantile whose value clears ``threshold`` (i.e. the fraction of
    the region at or above it). If nothing clears it, the loop runs to the last
    quantile and coverage is ``1 - 1 = 0``."""
    quant = [0.0, 0.25, 0.5, 0.75, 1.0]
    radius = [10.0, 20.0]
    date = np.array(["2025-01-01T00", "2025-01-01T01"], dtype="datetime64[ns]")
    q_vals = np.array(
        [
            # radius 10 km: t0 crosses at q=0.5 -> 0.5; t1 crosses at q=0 -> 1.0
            [[0.1, 0.2, 0.6, 0.9, 1.0], [0.6, 0.7, 0.8, 0.9, 1.0]],
            # radius 20 km: t0 never crosses -> 0.0; t1 crosses at q=0.75 -> 0.25
            [[0.1, 0.2, 0.3, 0.4, 0.45], [0.0, 0.0, 0.0, 0.6, 1.0]],
        ]
    )
    ds = xr.Dataset(
        {
            "tcolw": (("date",), [0.0, 0.0]),
            "tcolw_q": (("radius", "date", "quantile"), q_vals),
        },
        coords={"radius": radius, "date": date, "quantile": quant},
    )

    hm.add_coverage(ds, threshold=0.5)

    np.testing.assert_allclose(ds["tcolw_c"].sel(radius=10).values, [0.5, 1.0])
    np.testing.assert_allclose(ds["tcolw_c"].sel(radius=20).values, [0.0, 0.25])


# ---------------------------------------------------------------------------
# Layer 2 — fixture replayed through the real reduction pipeline
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).parents[1] / "fixtures" / "dataframes" / "herbie_tcolw_tiny.nc"


@pytest.fixture(scope="module")
def herbie_query() -> HerbieQuery:
    """A ``HerbieQuery`` built by replaying the cached HRRR subset through the
    real ``subset``/``extract_*``/``add_coverage`` pipeline (no network).

    Module-scoped: the pipeline builds a spatial tree, so it runs once and the
    read-only assertions below share the result."""
    with patch.object(hm, "get_tcolw", lambda **kwargs: xr.load_dataset(FIXTURE)):
        yield HerbieQuery(query_type="tcolw")


def test_query_replays_fixture_and_satisfies_contract(herbie_query):
    assert herbie_query.okay
    assert_herbie_dataset(herbie_query.ds, "tcolw")


def test_reduction_invariants_hold(herbie_query):
    """Structural invariants that survive fixture regeneration (no golden
    constants): quantiles are non-decreasing, the radial mean lies within the
    region's [min, max], coverage is a fraction, and the point series is
    finite."""
    ds = herbie_query.ds

    assert float(ds["tcolw_q"].diff("quantile").min()) >= 0.0

    q_min = ds["tcolw_q"].min("quantile")
    q_max = ds["tcolw_q"].max("quantile")
    assert bool((ds["tcolw_m"] >= q_min - 1e-6).all())
    assert bool((ds["tcolw_m"] <= q_max + 1e-6).all())

    assert float(ds["tcolw_c"].min()) >= 0.0
    assert float(ds["tcolw_c"].max()) <= 1.0

    assert bool(np.isfinite(ds["tcolw_p"]).all())


def test_query_property_accessors(herbie_query):
    q = herbie_query
    assert q.query_type == "tcolw"
    assert q.label_name == "TCOLW"
    assert q.label_unit == "kg/m^2"
    assert "TCOLW" in q.label
    assert q.n_steps == q.ds.sizes["step"]
    assert q.data.name == "tcolw"
    assert q.plot_log is True
    assert q.plot_norm_type == "log"


def test_query_okay_false_when_download_raises(monkeypatch):
    """The constructor swallows pipeline exceptions to the logger and leaves
    ``ds`` as ``None`` (okay=False)."""

    def boom(**kwargs):
        raise RuntimeError("simulated HRRR failure")

    monkeypatch.setattr(hm, "get_tcolw", boom)
    q = HerbieQuery(query_type="tcolw")
    assert not q.okay
    assert q.ds is None


def test_save_data_writes_netcdf(herbie_query, monkeypatch, tmp_path):
    out_root = tmp_path / "forecasts"
    monkeypatch.setattr(HerbieQuery, "forecast_dir", property(lambda self: out_root))
    herbie_query.save_data(outname="hrrr_tcolw")

    written = out_root / "hrrr_tcolw.netcdf"
    assert written.exists()
    xr.load_dataset(written)  # round-trips
