"""Record a tiny HRRR TCOLW subset for the offline Herbie test fixture.

Run from the repo root with ``uv run python tests/scripts/refresh_herbie_fixture.py``.
Requires network access to a HRRR bucket (Herbie downloads one latest run).

The offline test (``tests/query/test_herbie_maps.py``) monkeypatches
``herbie_maps.get_tcolw`` to replay this snapshot and then runs the *real*
reduction pipeline (``subset_rectangular_region`` → ``extract_*`` →
``add_coverage``) against it. So the fixture must be the raw ``get_tcolw``
output — the full-resolution quantity grid with its GRIB/plot attrs — merely
cropped in ``y``/``x`` to a small window around the VLA to keep it tiny.

The crop must stay large enough that ``subset_rectangular_region`` still leaves
enough grid points for the largest default reduction radius (``radius=20`` km →
``k ≈ 140`` nearest neighbours); the script asserts this so a future change to
the window or the radii can't silently produce a degenerate fixture.

This is generated fresh from a live download — never copied from the
exploratory netcdfs under ``data/old_data/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from site_forecast import SITE_LAT, SITE_LON
from site_forecast.query import herbie_maps as hm


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "dataframes"

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


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

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

    out_path = FIXTURES_DIR / "herbie_tcolw_tiny.nc"
    tiny.to_netcdf(out_path)

    print(
        f"Wrote {out_path}: dims={dict(tiny.sizes)}, "
        f"subset points={n_points} (>= k={k})"
    )
    print(f"      {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
