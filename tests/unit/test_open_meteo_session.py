from site_forecast.query.open_meteo import API_URL, ENSEMBLE_API_URL, RETRY_SESSION


def _forcelist():
    """The urllib3 Retry status_forcelist mounted on the open-meteo session."""
    adapter = RETRY_SESSION.get_adapter(API_URL)
    return adapter.max_retries.status_forcelist


def test_retry_session_retries_503_and_429():
    """Open-meteo intermittently returns 503 (and could return 429); both must
    be retried, not just the retry_requests default of (500, 502, 504)."""
    forcelist = _forcelist()
    assert 503 in forcelist
    assert 429 in forcelist
    # Regression guard: the retry_requests defaults must not silently return.
    for code in (500, 502, 504):
        assert code in forcelist


def test_retry_session_budget():
    """Five retries so transient blips are ridden out within the hourly cadence."""
    assert RETRY_SESSION.get_adapter(API_URL).max_retries.total == 5


def test_ensemble_endpoint_shares_retry_config():
    """The ensemble endpoint uses the same session, so it inherits the forcelist."""
    ensemble_forcelist = RETRY_SESSION.get_adapter(
        ENSEMBLE_API_URL
    ).max_retries.status_forcelist
    assert 503 in ensemble_forcelist
