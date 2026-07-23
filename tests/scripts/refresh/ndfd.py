"""NDFD fixture source: the parsed forecast DataFrame at ``ndfd.parquet``.

Requires network access to the NRAO forecast endpoint (see ``ndfd.URL``). The
offline test replays the *parsed* DataFrame (the text parser is exercised only
by the live ``schema`` lane), so we fetch once via the real ``NdfdQuery`` and
snapshot ``query.df``. Because the snapshot is frozen rather than reparsed, its
dtypes track whatever pandas produced when written — the diff flags an
index-resolution shift after a pandas bump.
"""

from __future__ import annotations

import pandas as pd

from site_forecast.query.ndfd import NdfdQuery

from . import FixtureSource, Signature, atomic_write, dataframe_signature, FIXTURES_DIR


PARQUET_PATH = FIXTURES_DIR / "dataframes" / "ndfd.parquet"


def refresh() -> None:
    query = NdfdQuery()
    if not query.okay:
        raise RuntimeError("NDFD fetch failed; fixture not written.")

    with atomic_write(PARQUET_PATH) as tmp:
        query.df.to_parquet(tmp)

    print(
        f"Wrote {PARQUET_PATH}: shape={query.df.shape}, "
        f"index={query.df.index.dtype}, {PARQUET_PATH.stat().st_size / 1024:.1f} KB",
        flush=True,
    )


def load() -> dict[str, Signature]:
    if not PARQUET_PATH.exists():
        return {}
    return {"ndfd.parquet": dataframe_signature(pd.read_parquet(PARQUET_PATH))}


SOURCE = FixtureSource("ndfd", refresh=refresh, load=load)
