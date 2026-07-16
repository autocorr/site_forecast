"""Skip guards for the live ``schema`` lane.

Each fixture checks that its service is reachable and calls ``pytest.skip``
otherwise, so ``pytest -m schema`` reports skips (never failures) on a machine
where a service is down or off-network. Reachability uses a bare TCP connect —
no request body — so the guard is cheap and side-effect free.
"""

import socket

import pytest

from site_forecast import CONFIG


def _reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        # Real offline failures — DNS (gaierror), refused, timeout — all
        # subclass OSError.
        return False
    except RuntimeError:
        # pytest-recording's `--block-network` patches the socket to raise
        # `RuntimeError("Network is disabled")`; treat that as unreachable so
        # the lane skips rather than errors under a network-blocked run.
        return False


@pytest.fixture
def require_open_meteo():
    # Covers all four open-meteo classes; the ensemble endpoint shares infra.
    if not _reachable("api.open-meteo.com", 443):
        pytest.skip("open-meteo API unreachable")


@pytest.fixture
def require_ndfd():
    if not _reachable("mcmonitor.evla.nrao.edu", 80):
        pytest.skip("NDFD forecast host (mcmonitor.evla.nrao.edu) unreachable")


@pytest.fixture
def require_internet():
    # Herbie pulls HRRR from cloud buckets with no single stable host, so gate
    # on a generic reachability check rather than a specific data endpoint.
    if not _reachable("1.1.1.1", 443):
        pytest.skip("no internet connectivity for Herbie HRRR download")


@pytest.fixture
def require_monitor():
    if not CONFIG.has_section("Monitor"):
        pytest.skip("no [Monitor] section in site_forecast.ini")
    host = CONFIG.get("Monitor", "host", fallback=None)
    if not host or not _reachable(host, 5432):
        pytest.skip("monitor PostgreSQL database unreachable")
