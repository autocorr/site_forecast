# Test framework for `site_forecast`

> **Revision note (2026-07-15).** Rewritten from the original proposal into a
> status tracker: the framework is roughly half implemented (commits
> `f4e84ae`, `0c703a1`) and the suite passes — **62 tests, offline, ~7 s**.
> Scope clarification: these tests are a **safety net for the migration in
> DESIGN.md**, written against the *current* API (`okay`, `save_data`, `.df`).
> They will be revised alongside the migration and do not need to anticipate
> the future `Result`/`Op` API; a more complete test suite gets built out
> after the migration. NOMADS tests are deferred (see below).

## Context

Two constraints from the user:

- **No network on every test run.** Fixtures replay cached query results so
  the default `pytest` is offline and fast.
- **Detect schema changes infrequently.** A separate, opt-in run hits the
  real services and validates response structure.

Standing decisions:

- **Lean two-layer fixtures.** VCR cassette for `open_meteo` (the only query
  with non-trivial custom parsing of a binary FlatBuffers wire format);
  parquet/netcdf/rowset snapshots for the rest. For `monitor`, snapshot the
  raw rowset so `parse_rows`/`rowset_to_dataframe` get exercised too.
- **pandera schemas** as the single source of truth for query DataFrame
  contracts — reused by live schema tests and as documentation.
- **pytest markers** (`schema`, `slow`) gate live/long tests; default
  `pytest` is offline.
- **Thin assertion layer on query tests.** The fixtures and pandera schemas
  are the durable assets — they validate the *data*, which survives the
  DESIGN.md refactor unchanged. The assertions on the wrapper (`okay`,
  `save_data`) are deliberately thin and against the current API; they get
  rewritten to `Result` status-gate assertions during DESIGN migration
  steps 2–6. Do not pre-build tests against the future API.

## Status

### Done

- [x] Test deps in `pyproject.toml` `test` extra: `pytest-recording`,
      `pytest-cov`, `pandera[pandas]`
- [x] `[tool.pytest.ini_options]`: `testpaths`, `pythonpath = ["tests"]`,
      `schema`/`slow` markers, `addopts = "-m 'not schema and not slow'"`
- [x] `tests/conftest.py` — path fixtures, `vcr_config`
      (`match_on`, `decode_compressed_response`, header filtering), and
      `bypass_openmeteo_cache` (monkeypatches `CACHE_SESSION`/`RETRY_SESSION`
      to a plain `requests.Session` so VCR can intercept below
      requests-cache)
- [x] `tests/schemas/queries.py` — schemas for `OpenMeteoVla`,
      `OpenMeteoMultiSite`, `OpenMeteoVlaPressure`, `OpenMeteoVlaEnsemble`,
      `Ndfd`, `ApiQuery`, `WeatherStation`; `assert_herbie_dataset` helper
- [x] Pure-function unit tests: `test_station.py`, `test_readdata.py`,
      `test_sensitivity_pure.py`, `test_plot_helpers.py`,
      `test_query_helpers.py`
- [x] `unit/test_open_meteo_session.py` (unplanned; regression tests for the
      503/429 retry config added in `5fb3dc6`)
- [x] `query/test_ndfd.py` + `fixtures/dataframes/ndfd.parquet`
- [x] `query/test_monitor.py` + `monitor_api_rows.pkl.gz` /
      `monitor_weather_rows.pkl.gz` (mocked `psycopg2.connect`, rowsets
      replayed by `side_effect` call order)
- [x] Fixture refresh via `scripts/refresh_fixtures.py <source>` (see the
      unified-script item below; per-source logic in `scripts/refresh/`)
- [x] `query/test_open_meteo.py` + four cassettes under `fixtures/cassettes/`.
      Conftest adds a `default_cassette_name` override keyed off a
      `@pytest.mark.cassette("name")` marker so the VLA tests share one
      recording. Recording revealed the date index is `datetime64[s, UTC]`,
      not the `[ns]` the schemas assumed — the four open-meteo schemas were
      corrected to match.
- [x] `schema_live/` lane (`test_live_{open_meteo,ndfd,herbie_maps,monitor}.py`,
      all `@pytest.mark.schema`) + `schema_live/conftest.py` socket-reachability
      skip guards. Herbie parametrized to `tcolw`/`mcc`; monitor gates on
      `df is not None`. The live run caught that `ndfd.parquet` was stale
      (`[ns]` vs the current parser's `[us, UTC]`); regenerated it and set
      `NdfdSchema` to `[us]` (now refreshed via `refresh_fixtures.py ndfd`).
- [x] **CI: `.github/workflows/test.yml`** — remote is GitHub
      (`autocorr/site_forecast`), so this applies as planned: PR/push
      trigger, `uv sync --all-extras`, `pytest --cov=site_forecast`,
      offline lane only. Independent of the migration.
- [x] `query/test_herbie_maps.py` + `fixtures/dataframes/herbie_tcolw_tiny.nc`
      (524 KB) + `refresh_fixtures.py herbie`. Two layers: known-answer
      unit tests for the pure reductions (`get_var_names`, `geodetic_to_number`,
      `subset_rectangular_region`, `add_coverage`) and a fixture-replay
      integration test that mocks `get_tcolw` and runs the real
      `subset`/`extract_*`/`add_coverage` pipeline offline (`pick_points`
      confirmed to work on a round-tripped netcdf). Integration assertions are
      structural (`assert_herbie_dataset`) + invariants, not golden constants.
- [x] `unit/test_sensitivity.py` + `fixtures/sensitivity/om_{surface,pressure}.parquet`
      + `refresh_fixtures.py sensitivity`. Covers `AmModelPredictor`
      profile prep, real-`am` smoke runs (`AmModelPredictor.run`,
      `VlaSeasonalPredictor.run`, and a truncated clear-sky `compute()` — all in
      the default lane, ~3 s), `_add_derived_columns` known-answer, `_get_tcolw_df`,
      the cloud/ensemble `_run_series` MultiIndex assembly (with `_run_am_model`
      mocked), and the real `multiprocessing.Pool` path at `n_workers=2` (exercises
      `_worker_init` in forked workers + arg pickling + aggregation). Reuses the
      open-meteo schemas and the Herbie fixture.
      **Caught two production bugs from the amwrap `main` bump:**
      `sensitivity.py:129` (`.m`→`.value`) and `:418` (`str`→`Path(...)`), both of
      which had silently zeroed out the sensitivity forecast; fixed and now guarded.
- [x] **Unified `scripts/refresh_fixtures.py`** — single manual CLI over a
      shared `scripts/refresh/` library. One `FixtureSource` per source
      (`refresh()` fetches live + writes atomically; `load()` signatures the
      current on-disk fixture). The CLI captures the signature before/after and
      prints a per-source diff — column/data_var adds/drops, **dtype and
      index-resolution changes**, and shape delta — the recurring
      `datetime64[s]`/`[ns]`/`[us]` drift being the real target. `open_meteo`
      has no `load()`: its `refresh()` shells out to
      `pytest --record-mode=rewrite` then replays the cassettes as the check.
      `--all` / named subset / `--list`; no-arg is a safe no-op (never silently
      hits the network + DB).

### Remaining, in priority order

_None — the offline-suite milestone is complete. The next build-out (fuller
plotter/orchestrator tests, nomads, the `slow` end-to-end lane) happens
alongside/after the DESIGN.md migration._

### Deferred

- **`nomads_cutout`** — the module was written some time ago but is not yet
  run in operations or wired to any consumer. It will eventually be fully
  implemented (see DESIGN.md Section 10); defer its offline test, fixture,
  pandera schema, and live schema test until it is brought into service
  after the migration. Do not build fixtures for it speculatively.
- **`slow` end-to-end lane** (instantiating the orchestrator with mocked
  queries and asserting outputs) — defer to the post-migration build-out,
  where it targets `Forecast(profile, ctx)` directly.

## Interaction with the DESIGN.md migration

- **Purpose.** This suite exists so that DESIGN migration steps 2–10 can
  ship incrementally with a regression net under them. Coverage of the
  current behavior is the goal; elegance of the assertions is not.
- **What survives the migration unchanged:** replay fixtures (cassettes,
  parquet/netcdf snapshots, pickled rowsets), the pandera schemas, the
  pure-function unit tests, and the live `schema` lane.
- **What gets rewritten during the migration:** the thin wrapper assertions
  in `tests/query/` (`assert q.okay` → status-gate assertions on
  `q.result`, then on operation `Result`s; `save_data()` checks → generic
  serializer checks). This happens naturally as each DESIGN step lands; it
  is expected churn, not rework to avoid.
- **After the migration**, build the fuller suite: plotters unit-tested
  with hand-built `dict[str, Result]`, `validate_profile` tests, the `slow`
  end-to-end lane, and nomads once implemented.

## Fixture strategy per query

**`open_meteo` (cassette, done)**: `pytest-recording` records via
`@pytest.mark.vcr`. The module-level
`CACHE_SESSION = requests_cache.CachedSession(".cache", expire_after=3600)`
short-circuits HTTP at the requests-cache layer, so VCR (at urllib3) never
sees the call — the `bypass_openmeteo_cache` conftest fixture swaps in a
plain `requests.Session` for the duration of each test. Note this also
bypasses the retry adapter; the retry *configuration* is covered separately
by `unit/test_open_meteo_session.py`. Cassettes store the FlatBuffers binary
as base64 in YAML — opaque but functional.

**`ndfd` (parquet, done)**: stores the parsed DataFrame. The text parser is
dead in offline tests — its schema-change detection is the responsibility
of the live schema test.

**`herbie_maps` (netcdf, done)**: the refresh script produces a tiny
subset — ~50×50 grid centered on the VLA (all sub-hourly steps kept),
524 KB. `get_tcolw` is mocked to return the cached Dataset; the real
reduction pipeline runs offline. The quantile/mean-at-radius reductions are
checked as known-answer unit tests on synthetic grids (exact values) and as
invariants on the fixture-driven `HerbieQuery`. (The
exploratory full-size netcdfs — `test_tcolw.netcdf` ~403 MB and
`test_tcolw_subset.netcdf` — now live under `data/old_data/`, still
untracked; the committed fixture is produced fresh by the refresh script,
never copied from working data.)

**`monitor` (pickled rowset, done)**: `psycopg2.connect` monkeypatched to a
mock whose cursor `fetchall()` returns pre-pickled raw rows by call order,
so `parse_rows` and `rowset_to_dataframe` — the real schema-bearing
parsers — execute against representative data.

## Schema validation (pandera)

`tests/schemas/queries.py` declares one `DataFrameSchema` per in-service
query class, plus an `assert_herbie_dataset` structural check for the
xarray-based Herbie query (pandera targets DataFrames). Two consumers:

1. **Offline test** loads the fixture, asserts it satisfies the schema —
   catches accidental fixture mutation and serves as a documented contract.
2. **Live test** (`-m schema`) hits the real service, asserts the result
   satisfies the schema — primary mechanism for detecting upstream
   structural change.

Schemas declare columns, dtypes, nullability, and index/MultiIndex names.
Value ranges are deliberately not locked (those drift), only structural
shape. These schemas are also the foothold for the schema-validated
`Result` evolution sketched in DESIGN.md Section 10.

## Run modes

- `pytest` — offline, fast, default (and the CI lane)
- `pytest -m schema` — live, run before releases or on a weekly cron
- `pytest -m slow` — end-to-end (post-migration)
- `pytest -m ""` — everything

## Known issues

The suite surfaces ~21 warnings in two families, both worth fixing before
upstream deprecations become errors:

- `Pandas4Warning: 'd' is deprecated` from `Station.sun_rise_and_sets` and
  from the tests themselves (`pd.Timedelta("1d")` → `"1D"`), plus one
  `concat` default-sorting deprecation.
- `DeprecationWarning: get_cmap ... removed in 3.11` from the Matplotlib
  colormap helpers.

## Verification (for each remaining slice)

1. `uv sync --all-extras` installs cleanly
2. `ruff check . && ruff format --check .` passes
3. `pytest` — all offline tests pass in well under a minute with no
   network traffic
4. `pytest -m schema` — run manually once after adding a live test
5. `pytest --cov=site_forecast` — coverage report generated
6. CI green once `.github/workflows/test.yml` lands
