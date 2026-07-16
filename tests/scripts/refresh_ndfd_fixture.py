"""Record the parsed NDFD forecast DataFrame for the offline test fixture.

Run from the repo root with ``uv run python tests/scripts/refresh_ndfd_fixture.py``.
Requires network access to the NRAO forecast endpoint (see ``ndfd.URL``).

The offline test replays the *parsed* DataFrame (the text parser is exercised
only by the live ``schema`` lane), so this script fetches once via the real
``NdfdQuery`` and snapshots ``query.df`` to
``tests/fixtures/dataframes/ndfd.parquet``. Because the snapshot is frozen
rather than reparsed, its dtypes track whatever pandas produced when it was
written — rerun this after a pandas upgrade if the live ``schema`` lane reports
an index-resolution drift against ``NdfdSchema``.
"""

from __future__ import annotations

from pathlib import Path

from site_forecast.query.ndfd import NdfdQuery
from site_forecast.query import to_parquet


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "dataframes"


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    query = NdfdQuery()
    if not query.okay:
        raise RuntimeError("NDFD fetch failed; fixture not written.")

    # `to_parquet` appends the `.parquet` suffix to the given stem.
    to_parquet(query.df, FIXTURES_DIR / "ndfd")
    out_path = FIXTURES_DIR / "ndfd.parquet"

    print(f"Wrote {out_path}: shape={query.df.shape}, index={query.df.index.dtype}")
    print(f"      {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
