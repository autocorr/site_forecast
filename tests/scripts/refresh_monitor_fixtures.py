"""Record raw rowsets from the VLA monitor DB for offline test fixtures.

Run from the repo root with ``uv run python tests/scripts/refresh_monitor_fixtures.py``.
Requires a ``[Monitor]`` section in ``site_forecast.ini`` with reachable host
credentials.

Captures the rowsets that ``ApiQuery`` and ``WeatherStationQuery`` issue
(matching the call order in ``MonitorConnection.query_phases`` and
``query_weather`` respectively) and writes two pickle files under
``tests/fixtures/dataframes/``. The fixtures embed the ``mjd_start``/
``mjd_end`` window so the offline tests can reproduce ``forecast_time``
and ``is_recent`` deterministically.

Window size is chosen to satisfy both ``is_complete`` (>12 h span) and
``is_recent`` (forecast_time within 1 h of ``df.index.max()``), with a
small enough cadence to keep each pickle under ~200 KB.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from astropy.time import Time

from site_forecast.query.monitor import (
    MonitorConnection,
    N_BASELINES,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "dataframes"

LOOKBACK_DAYS = 0.6  # ~14.4 h — passes is_complete (>12 h) for ApiQuery/WSQuery


def fetch_api_rowsets(conn: MonitorConnection, mjd_start: float, mjd_end: float):
    """Mirror ``MonitorConnection.query_phases`` order: RMS_PHASE0..5 then
    RMS_PHASE_FOR_OST on (host='evla-m360-1', device='API')."""
    host = "evla-m360-1"
    device = "API"
    mon_points = [f"RMS_PHASE{n}" for n in range(N_BASELINES)] + ["RMS_PHASE_FOR_OST"]
    return [conn.fetch_rows(host, device, mp, mjd_start, mjd_end) for mp in mon_points]


def fetch_weather_rowsets(conn: MonitorConnection, mjd_start: float, mjd_end: float):
    """Mirror ``MonitorConnection.query_weather`` order on host='evla-m352'."""
    items = [
        ("HMT337", "Temperature"),
        ("HMT337", "Dewpoint_Temperature"),
        ("HMT337", "Relative_Humidity"),
        ("WXT520", "Pressure"),
        ("WXT520", "Wind_Speed_Minimum"),
        ("WXT520", "Wind_Speed_Average"),
        ("WXT520", "Wind_Speed_Maximum"),
        ("WXT520", "Wind_Direction_Minimum"),
        ("WXT520", "Wind_Direction_Average"),
        ("WXT520", "Wind_Direction_Maximum"),
        ("M352", "Pyranometer_2"),
    ]
    return [
        conn.fetch_rows("evla-m352", device, mp, mjd_start, mjd_end)
        for device, mp in items
    ]


def _summarize(rowsets):
    return [len(rs) for rs in rowsets]


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    mjd_end = float(Time.now().mjd)
    mjd_start = mjd_end - LOOKBACK_DAYS

    conn = MonitorConnection()

    api_rowsets = fetch_api_rowsets(conn, mjd_start, mjd_end)
    weather_rowsets = fetch_weather_rowsets(conn, mjd_start, mjd_end)

    api_path = FIXTURES_DIR / "monitor_api_rows.pkl.gz"
    weather_path = FIXTURES_DIR / "monitor_weather_rows.pkl.gz"

    for path, rowsets in ((api_path, api_rowsets), (weather_path, weather_rowsets)):
        with gzip.open(path, "wb", compresslevel=9) as f:
            pickle.dump(
                {
                    "mjd_start": mjd_start,
                    "mjd_end": mjd_end,
                    "rowsets": rowsets,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    print(f"Wrote {api_path}: rowset sizes={_summarize(api_rowsets)}")
    print(f"      {api_path.stat().st_size / 1024:.1f} KB")
    print(f"Wrote {weather_path}: rowset sizes={_summarize(weather_rowsets)}")
    print(f"      {weather_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
