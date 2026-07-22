"""Record open-meteo surface + pressure inputs for the sensitivity test fixtures.

Run from the repo root with
``uv run python tests/scripts/refresh_sensitivity_fixtures.py``.
Requires network access to the open-meteo API.

``tests/unit/test_sensitivity.py`` builds ``VlaSensitivityEstimator`` (and the
lower-level ``AmModelPredictor``) from these two frames, then runs the real AM
radiative-transfer model offline. The estimator needs the surface and
pressure-level forecasts *together*, so — unlike the query tests, which replay a
single VCR cassette per test — we snapshot the parsed DataFrames directly:

- ``om_surface.parquet``  — ``OpenMeteoVlaQuery.df`` (``date``-indexed)
- ``om_pressure.parquet`` — ``OpenMeteoVlaPressureQuery.df`` (``(date, pressure)``)

Both are truncated to the first ~24 h so the fixtures stay tiny while still
sharing enough hourly timestamps for the estimator's clear-sky path. Regenerate
after an open-meteo schema change flagged by the live ``schema`` lane.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from site_forecast.query.open_meteo import (
    OpenMeteoVlaQuery,
    OpenMeteoVlaPressureQuery,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sensitivity"

WINDOW = pd.Timedelta("24h")


def _dates(df: pd.DataFrame, level: str | None):
    return df.index if level is None else df.index.get_level_values(level)


def _clip(df: pd.DataFrame, level: str | None, start, end) -> pd.DataFrame:
    dates = _dates(df, level)
    return df[(dates >= start) & (dates <= end)]


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    surf = OpenMeteoVlaQuery()
    pres = OpenMeteoVlaPressureQuery()
    if not (surf.okay and pres.okay):
        raise RuntimeError("open-meteo fetch failed; fixtures not written.")

    # Surface carries past days but pressure starts at the forecast time, so
    # anchor both to a shared window; otherwise the truncated frames barely
    # overlap and the estimator's clear-sky path finds almost no common times.
    start = max(_dates(surf.df, None).min(), _dates(pres.df, "date").min())
    end = start + WINDOW
    surf_df = _clip(surf.df, None, start, end)
    pres_df = _clip(pres.df, "date", start, end)

    surf_path = FIXTURES_DIR / "om_surface.parquet"
    pres_path = FIXTURES_DIR / "om_pressure.parquet"
    surf_df.to_parquet(surf_path)
    pres_df.to_parquet(pres_path)

    for name, df, path in (
        ("surface", surf_df, surf_path),
        ("pressure", pres_df, pres_path),
    ):
        print(
            f"Wrote {path}: {name} shape={df.shape}, "
            f"{path.stat().st_size / 1024:.1f} KB"
        )


if __name__ == "__main__":
    main()
