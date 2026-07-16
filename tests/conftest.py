from pathlib import Path

import pytest
import requests


TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
CASSETTES_DIR = FIXTURES_DIR / "cassettes"
DATAFRAMES_DIR = FIXTURES_DIR / "dataframes"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def cassettes_dir() -> Path:
    return CASSETTES_DIR


@pytest.fixture
def dataframes_dir() -> Path:
    return DATAFRAMES_DIR


@pytest.fixture
def vcr_cassette_dir() -> str:
    """Store cassettes alongside the other replay fixtures rather than in the
    pytest-recording default (`tests/query/cassettes/`)."""
    return str(CASSETTES_DIR)


@pytest.fixture
def default_cassette_name(request) -> str:
    """Let a test pick a shared cassette via ``@pytest.mark.cassette("name")``
    so several tests exercising the same request replay one recording. Falls
    back to pytest-recording's default (the test name) when unmarked."""
    marker = request.node.get_closest_marker("cassette")
    if marker is not None:
        return marker.args[0]
    from pytest_recording.plugin import get_default_cassette_name

    return get_default_cassette_name(request.cls, request.node.name)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "match_on": ["method", "scheme", "host", "path", "query"],
        "decode_compressed_response": True,
        "filter_headers": ["authorization", "cookie", "set-cookie"],
    }


@pytest.fixture
def bypass_openmeteo_cache(monkeypatch):
    """Replace the module-level cached/retry sessions in open_meteo so VCR
    can intercept HTTP. requests_cache short-circuits at the requests layer,
    before urllib3 — VCR never sees the call unless we bypass it."""
    from site_forecast.query import open_meteo

    session = requests.Session()
    monkeypatch.setattr(open_meteo, "CACHE_SESSION", session)
    monkeypatch.setattr(open_meteo, "RETRY_SESSION", session)
    yield session
