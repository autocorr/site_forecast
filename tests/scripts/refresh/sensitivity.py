"""Sensitivity fixture source: open-meteo surface + pressure inputs.

Requires network access to the open-meteo API. ``tests/unit/test_sensitivity.py``
builds ``VlaSensitivityEstimator`` (and ``AmModelPredictor``) from these two
frames, then runs the real AM radiative-transfer model offline:

- ``om_surface.parquet``  — ``OpenMeteoVlaQuery.df`` (``date``-indexed)
- ``om_pressure.parquet`` — ``OpenMeteoVlaPressureQuery.df`` (``(date, pressure)``)

Both are truncated to a shared ~24 h window so the fixtures stay tiny while
still sharing enough hourly timestamps for the estimator's clear-sky path.
"""

from __future__ import annotations

import pandas as pd

from site_forecast.query.open_meteo import (
    OpenMeteoVlaPressureQuery,
    OpenMeteoVlaQuery,
)

from . import FixtureSource, Signature, atomic_write, dataframe_signature, FIXTURES_DIR


SENSITIVITY_DIR = FIXTURES_DIR / "sensitivity"
SURFACE_PATH = SENSITIVITY_DIR / "om_surface.parquet"
PRESSURE_PATH = SENSITIVITY_DIR / "om_pressure.parquet"

WINDOW = pd.Timedelta("24h")


def _dates(df: pd.DataFrame, level: str | None):
    return df.index if level is None else df.index.get_level_values(level)


def _clip(df: pd.DataFrame, level: str | None, start, end) -> pd.DataFrame:
    dates = _dates(df, level)
    return df[(dates >= start) & (dates <= end)]


def refresh() -> None:
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

    for name, df, path in (
        ("surface", surf_df, SURFACE_PATH),
        ("pressure", pres_df, PRESSURE_PATH),
    ):
        with atomic_write(path) as tmp:
            df.to_parquet(tmp)
        print(
            f"Wrote {path}: {name} shape={df.shape}, "
            f"{path.stat().st_size / 1024:.1f} KB",
            flush=True,
        )


def load() -> dict[str, Signature]:
    out: dict[str, Signature] = {}
    for path in (SURFACE_PATH, PRESSURE_PATH):
        if path.exists():
            out[path.name] = dataframe_signature(pd.read_parquet(path))
    return out


SOURCE = FixtureSource("sensitivity", refresh=refresh, load=load)
