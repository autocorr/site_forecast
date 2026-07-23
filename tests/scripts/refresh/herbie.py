"""Herbie fixture source: a tiny HRRR TCOLW subset at ``herbie_tcolw_tiny.nc``.

Requires network access to a HRRR bucket (Herbie downloads one latest run).
``tests/query/test_herbie_maps.py`` monkeypatches ``herbie_maps.get_tcolw`` to
replay this snapshot and then runs the *real* reduction pipeline against it, so
the fixture is the raw ``get_tcolw`` output — the full-resolution quantity grid
with its GRIB/plot attrs — merely cropped in ``y``/``x`` to a small window
around the VLA.

The crop must stay large enough that ``subset_rectangular_region`` still leaves
enough points for the largest default reduction radius (20 km -> k ~= 140); the
refresh asserts this so a future window/radius change can't silently produce a
degenerate fixture. Generated fresh from a live download — never copied from the
exploratory netcdfs under ``data/old_data/``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from site_forecast import SITE_LAT, SITE_LON
from site_forecast.query import herbie_maps as hm

from . import FixtureSource, Signature, atomic_write, dataset_signature, FIXTURES_DIR


NETCDF_PATH = FIXTURES_DIR / "dataframes" / "herbie_tcolw_tiny.nc"

WINDOW_HALF = 25  # grid points either side of the VLA -> ~50x50 crop


def _nearest_yx(ds) -> tuple[int, int]:
    """Index of the grid cell closest to the VLA in the 2-D lat/lon coords."""
    lat0, lon0 = hm.geodetic_to_number(SITE_LAT, SITE_LON)
    dist2 = (ds.latitude.values - lat0) ** 2 + (ds.longitude.values - lon0) ** 2
    return np.unravel_index(int(np.argmin(dist2)), dist2.shape)


def _min_points_for_default_radii() -> int:
    # Mirror extract_circular_region's k for the largest default radius (20 km,
    # HRRR 3 km resolution).
    return int(np.pi * (20 / 3) ** 2)


def refresh() -> None:
    ds = hm.get_tcolw()
    iy, ix = _nearest_yx(ds)
    tiny = ds.isel(
        y=slice(max(iy - WINDOW_HALF, 0), iy + WINDOW_HALF),
        x=slice(max(ix - WINDOW_HALF, 0), ix + WINDOW_HALF),
    )

    # Sanity: the rectangular subset the query performs must retain enough
    # points for the largest reduction radius.
    sub = hm.subset_rectangular_region(tiny)
    step0 = sub.tcolw.isel(step=0) if "step" in sub.tcolw.dims else sub.tcolw
    n_points = int(step0.count())
    k = _min_points_for_default_radii()
    if n_points < k:
        raise RuntimeError(
            f"Subset retains only {n_points} points; need >= {k} for radius=20 km. "
            "Widen WINDOW_HALF."
        )

    with atomic_write(NETCDF_PATH) as tmp:
        tiny.to_netcdf(tmp)

    print(
        f"Wrote {NETCDF_PATH}: dims={dict(tiny.sizes)}, "
        f"subset points={n_points} (>= k={k}), "
        f"{NETCDF_PATH.stat().st_size / 1024:.1f} KB",
        flush=True,
    )


def load() -> dict[str, Signature]:
    if not NETCDF_PATH.exists():
        return {}
    with xr.open_dataset(NETCDF_PATH) as ds:
        return {"herbie_tcolw_tiny.nc": dataset_signature(ds)}


SOURCE = FixtureSource("herbie", refresh=refresh, load=load)
