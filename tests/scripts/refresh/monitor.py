"""Monitor fixture source: raw VLA-monitor rowsets at ``monitor_*_rows.pkl.gz``.

Requires a ``[Monitor]`` section in ``site_forecast.ini`` with reachable host
credentials. Captures the rowsets that ``ApiQuery`` and ``WeatherStationQuery``
issue (matching the call order in ``MonitorConnection.query_phases`` /
``query_weather``) and writes two pickle files. The fixtures embed the
``mjd_start``/``mjd_end`` window so the offline tests reproduce ``forecast_time``
and ``is_recent`` deterministically.

``load()`` replays the pickled rowsets back through ``ApiQuery`` /
``WeatherStationQuery`` (mocking ``psycopg2.connect`` exactly as
``tests/query/test_monitor.py`` does) so the diff is over the *parsed* frames —
where a dtype shift would actually bite the offline tests.
"""

from __future__ import annotations

import gzip
import pickle
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

from astropy.time import Time

import site_forecast.query.monitor as monitor_mod
from site_forecast.query.monitor import (
    ApiQuery,
    MonitorConnection,
    N_BASELINES,
    WeatherStationQuery,
)

from . import FixtureSource, Signature, atomic_write, dataframe_signature, FIXTURES_DIR


DATAFRAMES_DIR = FIXTURES_DIR / "dataframes"
API_PATH = DATAFRAMES_DIR / "monitor_api_rows.pkl.gz"
WEATHER_PATH = DATAFRAMES_DIR / "monitor_weather_rows.pkl.gz"

LOOKBACK_DAYS = 0.6  # ~14.4 h — passes is_complete (>12 h) for ApiQuery/WSQuery


def _fetch_api_rowsets(conn, mjd_start, mjd_end):
    """Mirror ``MonitorConnection.query_phases`` order: RMS_PHASE0..5 then
    RMS_PHASE_FOR_OST on (host='evla-m360-1', device='API')."""
    mon_points = [f"RMS_PHASE{n}" for n in range(N_BASELINES)] + ["RMS_PHASE_FOR_OST"]
    return [
        conn.fetch_rows("evla-m360-1", "API", mp, mjd_start, mjd_end)
        for mp in mon_points
    ]


def _fetch_weather_rowsets(conn, mjd_start, mjd_end):
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


def refresh() -> None:
    mjd_end = float(Time.now().mjd)
    mjd_start = mjd_end - LOOKBACK_DAYS
    conn = MonitorConnection()

    for path, rowsets in (
        (API_PATH, _fetch_api_rowsets(conn, mjd_start, mjd_end)),
        (WEATHER_PATH, _fetch_weather_rowsets(conn, mjd_start, mjd_end)),
    ):
        payload = {"mjd_start": mjd_start, "mjd_end": mjd_end, "rowsets": rowsets}
        with atomic_write(path) as tmp:
            with gzip.open(tmp, "wb", compresslevel=9) as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        sizes = [len(rs) for rs in rowsets]
        print(
            f"Wrote {path}: rowset sizes={sizes}, {path.stat().st_size / 1024:.1f} KB",
            flush=True,
        )


@contextmanager
def _mock_connection(rowsets):
    """Wire ``psycopg2.connect`` to a mock yielding ``rowsets`` from successive
    ``cursor.fetchall()`` calls, matching ``tests/query/test_monitor.py``."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = list(rowsets)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    real_connect = monitor_mod.psycopg2.connect
    monitor_mod.psycopg2.connect = lambda **kwargs: mock_conn
    try:
        yield
    finally:
        monitor_mod.psycopg2.connect = real_connect


def _replay(path: Path, query_cls):
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    with _mock_connection(payload["rowsets"]):
        q = query_cls(mjd_start=payload["mjd_start"], mjd_end=payload["mjd_end"])
    return q.df


def load() -> dict[str, Signature]:
    out: dict[str, Signature] = {}
    for path, query_cls in ((API_PATH, ApiQuery), (WEATHER_PATH, WeatherStationQuery)):
        if path.exists():
            out[path.name] = dataframe_signature(_replay(path, query_cls))
    return out


SOURCE = FixtureSource("monitor", refresh=refresh, load=load)
