# site_forecast — Architecture Redesign

This document describes a target architecture for `site_forecast` and an incremental migration path from the current code. It is a design proposal, not a code change.

> **Revision note (2026-07-15).** Revised after a review against the current codebase. Substantive changes from the first draft: the `Result.require`/`usable` contract was made coherent (three consumer tiers: `okay`, `usable`, `has_data`); the string-keyed `OperationRegistry` was replaced with direct `Op` references held by profiles; operations now declare `requires` and profiles are validated at startup; the phase-prediction sketch preserves the existing four-model fallback ladder (including the `seasonal` model that runs when every input fails); profiles declare their own writers so the VLA `forecast.tsv` does not leak into the site-agnostic orchestrator; `SiteProfile` reuses the existing `Station` class instead of a new `SiteLocation`; the migration steps were reordered (Context extraction before `safe_run`); and the inventory of operations was synced to the current code (ensemble query, four Herbie cutouts, `previous_phase_forecasts`).

The redesign is driven by two pressures:

1. **Multi-site expansion.** The codebase started as a VLA-only forecaster and is being extended to "shallow" multi-site VLBA analyses and (eventually) deeper analyses for sites such as the GBT. Site-specific behavior is currently scattered across module-level globals (`SITE_LAT`, `SITE_LON`), VLA-only modules (`query/monitor.py`), and unconditional construction inside `Forecast.__init__`.
2. **Query/result entanglement.** Today a "query" class owns the network call, the error swallowing (`except: self.df = None`), the result, and a thin `okay` flag whose semantics differ between subclasses. Validity is implicit in the DataFrame's contents. Plotters and the phase predictor reach into `.df` directly, so a missing column or a partial time range silently breaks downstream code or hides inside a broad `try/except`. A concrete example of the failure class: `OpenMeteoVlaPressureQuery.__init__` calls `unpivot_pressure_levels(self.df, ...)` unconditionally after the parent constructor's `try/except` may have set `self.df = None`, so a failed pressure query raises `AttributeError` out of `Forecast.__init__` and kills the entire hourly run.

The goal is an architecture where:

- Sites are composed, not subclassed.
- Failures are first-class data on a result object, not exceptions or `None`s.
- Adding a new analysis or a new site is a local change, not a core refactor.

## 1. Goals and non-goals

### Goals

- **Composable sites.** Each site (VLA, VLBA-multi, individual VLBA stations, future GBT) is described by a `SiteProfile` that lists which operations to run, which plotters to render, and which writers to run. The orchestrator does not branch on site identity.
- **Explicit result validity.** Every operation returns a `Result` whose `status` field is the source of truth for "can downstream code use this?". DataFrame state is no longer a hidden contract. Because different consumers have different validity thresholds (the phase model must not train on stale monitor data, but the phase-history plot should still draw it), `Result` exposes a small ladder of checks (`okay` / `usable` / `has_data`) rather than a single boolean.
- **Best-effort orchestration.** Each operation runs independently; one failure does not abort the run. Downstream operations and plotters check the appropriate `Result` gate and skip cleanly with a logged reason.
- **Fail fast on misconfiguration.** Data failures degrade gracefully at runtime; *wiring* failures (a profile pipeline that omits or misorders a dependency) are caught by validation at startup, not disguised as missing data an hour into operations.
- **Preserve the fallback ladder.** The current system always produces *some* phase forecast — down to the `seasonal` model driven by synthetic hour/day-of-year covariates when every external input fails. The redesign keeps that guarantee.
- **Incremental migration.** Every step of the migration leaves the hourly operational pipeline working. There is no flag day.

### Non-goals

- Rewriting plotting from scratch. Existing Matplotlib and Plotly figure code is preserved; only its inputs change.
- Full per-column schema validation (pandera/pydantic). The `Result` wrapper is forward-compatible with that, but it is not part of this design.
- Hiding the underlying DataFrame/Dataset behind named domain accessors. `.data` is still exposed.
- Generalizing the LightGBM phase RMS predictor to non-VLA sites. It stays a VLA-only operation.
- Parallel execution within a pipeline. Sequential is fine and matches current behavior.

## 2. Architectural pillars

### A. Operations as first-class values

Every unit of work — querying Open-Meteo, fetching HRRR via Herbie, running the LightGBM phase predictor, computing AM sensitivity — is an `Op`: a small frozen dataclass holding a `name`, the callable, and the names of the results it `requires`. Profiles reference `Op` objects **directly** (ordinary imports), not through a string-keyed registry.

An earlier draft used a global `OperationRegistry` populated by import side effects. That was dropped: for roughly a dozen operations with no plugin requirement, a registry costs more than it pays — registration-by-import forces importing every heavy dependency (psycopg2, Herbie, darts) even for a shallow per-station profile, module re-imports under pytest trip duplicate-registration errors, and string names lose IDE navigation and typo safety. Direct references give identical composability (the results dict is still keyed by `op.name`) while being statically checkable and free of global mutable state.

### B. Lightweight `Result`

Operations return a `Result`: a small dataclass carrying a `status` enum, the underlying data (DataFrame, Dataset, or `None`), an `issues` list of human-readable strings, the `coverage` time window of the data, and the `forecast_time`. The DataFrame stays accessible via `result.data`. No schema validation, no domain-specific accessors.

### C. Site profiles

A `SiteProfile` is a pure-data description of a site: its code, its `Station`(s), the ordered pipeline of `Op`s, the plotters to render, and the writers to run. There is no `VlaSite` subclass, and the orchestrator contains no site-specific output logic — the VLA `forecast.tsv` is a writer declared on the VLA profile.

### D. Best-effort orchestration

The orchestrator runs each pipeline step inside a `safe_run` wrapper that converts any uncaught exception into a `Result(status=MISSING, …)`. The pipeline runner never raises mid-pipeline. Plotters and downstream operations are responsible for checking the appropriate `Result` gate and degrading gracefully when an upstream input is missing.

## 3. Core abstractions

All of the following live under `src/site_forecast/core/`. Sketches below; types are illustrative, not final.

### 3.1 `Status` and `Result`

```python
# core/result.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import pandas as pd


class Status(str, Enum):
    OK      = "ok"        # data present, passes validity checks
    PARTIAL = "partial"   # data present but missing columns or has gaps
    STALE   = "stale"     # data present but too old to be useful
    MISSING = "missing"   # query failed; no data


@dataclass
class Result:
    """Standard exchange type returned by every operation."""
    name: str
    status: Status
    forecast_time: pd.Timestamp
    data: Any = None                       # DataFrame, xarray Dataset, or None
    issues: list[str] = field(default_factory=list)
    coverage: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

    @property
    def okay(self) -> bool:
        """Strict gate: data present and complete. Model inputs use this."""
        return self.status is Status.OK

    @property
    def usable(self) -> bool:
        """Tolerant gate: some current data; consumer handles gaps."""
        return self.status in (Status.OK, Status.PARTIAL)

    @property
    def has_data(self) -> bool:
        """Weakest gate: anything at all, even stale. History plots use this."""
        return self.status is not Status.MISSING

    def has(self, *names: str) -> bool:
        """has_data plus the named columns/variables present. For consumers
        that accept stale data but still need specific columns."""
        if self.data is None:
            return False
        have = getattr(self.data, "columns", None)
        if have is None:
            have = getattr(self.data, "data_vars", ())
        return all(n in have for n in names)

    def with_issue(self, msg: str) -> "Result":
        return Result(self.name, self.status, self.forecast_time,
                      self.data, [*self.issues, msg], self.coverage)

    def degraded_to(self, status: Status, msg: str) -> "Result":
        return Result(self.name, status, self.forecast_time,
                      self.data, [*self.issues, msg], self.coverage)

    def require(self, *names: str) -> "Result":
        """Return self, or a PARTIAL copy if any required column/variable
        is missing. Works on DataFrame columns and xarray Dataset data_vars."""
        if self.data is None:
            return self
        have = getattr(self.data, "columns", None)
        if have is None:
            have = getattr(self.data, "data_vars", ())
        missing = [n for n in names if n not in have]
        if missing:
            return self.degraded_to(Status.PARTIAL, f"missing columns: {missing}")
        return self
```

The three gates form a ladder (`okay` ⊂ `usable` ⊂ `has_data`) and each consumer picks the weakest gate it can tolerate:

| Consumer | Gate | Rationale |
|---|---|---|
| Model input (phase predictor, AM sensitivity) | `require(*cols).okay` | Must not run on incomplete or stale data. |
| Typical forecast plot panel | `usable` | Can draw around gaps; must not draw stale data as if current. |
| History/context plot (e.g., recent measured phase RMS) | `has_data`, or `has(*cols)` when specific columns are needed | Stale measurements are still informative to operators. |

**`require` must be gated with `.okay`, not `.usable`.** `require` degrades a result with missing columns to `PARTIAL`, and `PARTIAL` still satisfies `usable` — checking `usable` after `require` would pass the degraded result through and reintroduce the downstream `KeyError` this design exists to prevent. The canonical pattern is:

```python
weather = results["open_meteo_surface"].require("wind_speed_10m", "temperature_2m")
if not weather.okay:
    logger.warning("skipping wind plot: %s", weather.issues)
    return
```

A consumer that genuinely tolerates missing columns checks `usable` and then guards its own column accesses — `require` is for consumers that need all the named columns.

Note the asymmetry: `require(…).okay` also rejects `STALE`, which is right for model inputs but too strict for history plots. A history plotter that needs specific columns on possibly-stale data uses `result.has("phase_rms")` instead — `require`'s `PARTIAL` degradation would slip past both `usable` and `has_data`.

### 3.2 `Context`

```python
# core/context.py
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .profile import SiteProfile


@dataclass(frozen=True)
class Context:
    """Per-run context. Replaces module-level CONFIG, SITE_LAT, SITE_LON."""
    profile: SiteProfile
    forecast_time: pd.Timestamp
    forecast_root: Path                    # root of all forecast output
    forecast_dir: Path                     # output dir for this run + profile
    cache_dir: Path
    config: ConfigParser                   # parsed site_forecast.ini snapshot

    @property
    def station(self) -> "Station":
        """The single station for a single-site profile. Raises for
        multi-site profiles, which must iterate ctx.profile.stations."""
        (station,) = self.profile.stations
        return station
```

The `Context` carries the bits a query needs that today are pulled from globals or the `CONFIG` singleton. Operations receive a `Context` argument; they do not import `SITE_LAT`/`SITE_LON` or read `CONFIG` directly. `forecast_root` is included (not just `forecast_dir`) because the `previous_phase_forecasts` operation reads prior runs' output from disk.

There is deliberately no `ctx.latitude` shortcut: multi-site profiles have no single location, and coordinates should come from a `Station` (with astropy units) rather than bare floats.

### 3.3 `Op` and `safe_run`

```python
# core/operation.py
from dataclasses import dataclass
from typing import Callable
import logging

from .result import Result, Status
from .context import Context

logger = logging.getLogger(__name__)

OperationFn = Callable[[Context, dict[str, Result]], Result]


@dataclass(frozen=True)
class Op:
    name: str
    fn: OperationFn
    requires: tuple[str, ...] = ()         # names of upstream results consumed


def safe_run(op: Op, ctx: Context, results: dict[str, Result]) -> Result:
    """Execute op.fn; convert any uncaught exception into a MISSING Result."""
    try:
        return op.fn(ctx, results)
    except Exception as exc:
        logger.exception("operation %s failed", op.name)
        return Result(
            name=op.name,
            status=Status.MISSING,
            forecast_time=ctx.forecast_time,
            issues=[f"{type(exc).__name__}: {exc}"],
        )
```

`safe_run` is the single place where exceptions are caught and translated into `Status.MISSING`. The bare-`except` blocks scattered across the query classes (`OpenMeteoQuery.__init__`, `OpenMeteoVlaEnsembleQuery.__init__`, `ApiQuery.__init__`, `WeatherStationQuery.__init__`, `HerbieQuery.__init__`, `NdfdQuery`, `ModelPhaseForecast.__init__`) are replaced by this one wrapper.

An operation is defined as a function plus an `Op` wrapper, imported directly by profiles:

```python
# operations/open_meteo.py
def _open_meteo_surface(ctx: Context, results: dict[str, Result]) -> Result:
    ...

open_meteo_surface = Op("open_meteo_surface", _open_meteo_surface)
```

`requires` mirrors the `Plotter.requires` field (Section 3.4) and exists so profiles can be **validated at startup** (Section 3.5): a pipeline that omits or misorders a dependency is a wiring bug and should fail at import time, not surface an hour later as a `KeyError` inside `safe_run` dressed up as missing data.

`requires` declares *wiring*, not *validity*: a required operation is guaranteed to have run earlier and placed a `Result` in the dict, but that result may be `MISSING`. Validity is always a runtime check via the status gates. This is what lets an operation like `am_sensitivity` list optional inputs (cloud ice, the PWV ensemble) in `requires` and still degrade gracefully when they carry no data.

### 3.4 `Plotter` and `Writer`

```python
# core/plotter.py
from typing import Protocol
from .result import Result
from .context import Context


class Plotter(Protocol):
    name: str
    requires: tuple[str, ...]              # operation names

    def render(self, ctx: Context, results: dict[str, Result]) -> None: ...


class Writer(Protocol):
    name: str
    requires: tuple[str, ...]

    def write(self, ctx: Context, results: dict[str, Result]) -> None: ...
```

A plotter declares which operation results it consumes. The orchestrator passes the full `results` dict; the plotter's `render` is responsible for checking the appropriate gate (`okay`/`usable`/`has_data`, per the table in Section 3.1) on its requirements and skipping cleanly. Plotters no longer take a `Forecast` object — that decouples them from the orchestrator and lets a plotter be unit-tested with a hand-built `dict[str, Result]`.

`Writer` is the same shape for non-plot outputs. The VLA `forecast.tsv` is a writer on the VLA profile — the orchestrator does not know it exists. Generic serialization of raw results (each result's DataFrame to parquet, Dataset to netcdf) stays in the orchestrator because it is genuinely site-agnostic.

### 3.5 `SiteProfile`

```python
# core/profile.py
from dataclasses import dataclass
from .. import Station                    # existing class; reused, not duplicated
from .operation import Op
from .plotter import Plotter, Writer


@dataclass(frozen=True)
class SiteProfile:
    code: str                              # "Y1", "PT", "VLBA_MULTI", ...
    name: str                              # human-readable
    stations: tuple[Station, ...]          # one entry for single-site profiles
    pipeline: tuple[Op, ...]               # ordered operations
    plotters: tuple[Plotter, ...] = ()
    writers: tuple[Writer, ...] = ()


def validate_profile(profile: SiteProfile) -> None:
    """Fail at startup if the pipeline is mis-wired."""
    seen: set[str] = set()
    for op in profile.pipeline:
        missing = set(op.requires) - seen
        if missing:
            raise ValueError(
                f"{profile.code}: op {op.name!r} requires {sorted(missing)} "
                "which do not run earlier in the pipeline"
            )
        if op.name in seen:
            raise ValueError(f"{profile.code}: duplicate op {op.name!r}")
        seen.add(op.name)
    for consumer in (*profile.plotters, *profile.writers):
        missing = set(consumer.requires) - seen
        if missing:
            raise ValueError(
                f"{profile.code}: {consumer.name!r} requires {sorted(missing)}"
            )
```

`SiteProfile` reuses the existing `Station` class rather than introducing a parallel `SiteLocation` of bare floats: `Station` already carries astropy-unit coordinates, `earth_location`, and `sun_rise_and_sets`, which the night-shading and sunrise/sunset plot code actively uses. Duplicating the location as unitless floats would lose that behavior and invite unit bugs.

Every site-specific decision is encoded in the `SiteProfile`'s `pipeline`, `plotters`, and `writers`. Adding a site is writing a new profile; it is never a change to the core. All fields are tuples so profiles are genuinely immutable and hashable.

### 3.6 `Forecast` orchestrator

```python
# orchestrator.py
from .core.context import Context
from .core.profile import SiteProfile, validate_profile
from .core.operation import safe_run
from .core.result import Result


class Forecast:
    def __init__(self, profile: SiteProfile, ctx: Context):
        self.profile = profile
        self.ctx = ctx
        self.results: dict[str, Result] = {}

    def run(self) -> None:
        for op in self.profile.pipeline:
            self.results[op.name] = safe_run(op, self.ctx, self.results)

    def write_outputs(self) -> None:
        for plotter in self.profile.plotters:
            try:
                plotter.render(self.ctx, self.results)
            except Exception:
                logger.exception("plotter %s failed", plotter.name)
        for writer in self.profile.writers:
            try:
                writer.write(self.ctx, self.results)
            except Exception:
                logger.exception("writer %s failed", writer.name)
        write_serialized_results(self.ctx, self.results)   # generic parquet/netcdf
```

The orchestrator is small and site-agnostic. `loop()` schedules `Forecast(profile, ctx).run()` for each profile that needs to run hourly; `validate_profile` is called on every profile at startup, before the first scheduled run.

## 4. Data flow

```
                      SiteProfile
                          │
                          ▼
                Pipeline (ordered Ops)
                          │
                          ▼
              safe_run(op, ctx, results)
                          │   (single exception-catch site)
                          ▼
                Result(status, data, issues, coverage)
                          │
                          ▼
                results: dict[str, Result]
              │           │              │
              ▼           ▼              ▼
          Plotters     Writers     Generic serializers
                                   (parquet / netcdf)
```

A pipeline step that depends on an earlier step reads it from `results`. The phase-prediction operation is the most involved consumer; the sketch below preserves today's four-model fallback ladder from `ModelPhaseForecast.model_types`, including the guarantee that a forecast is produced even when *every* external input fails:

```python
# operations/phase_predict.py
MODEL_TYPES = {
    # (weather_okay, phase_okay): model_name
    (True,  True):  "full",
    (False, True):  "no_weather",
    (True,  False): "no_phase",
    (False, False): "seasonal",
}


def _phase_rms_lightgbm(ctx: Context, results: dict[str, Result]) -> Result:
    weather = results["open_meteo_surface"].require(*MODEL_WEATHER_COLUMNS)
    phase   = results["vla_phase_db"]
    w_okay = weather.okay
    p_okay = phase.okay      # STALE or PARTIAL monitor data is not a model input
    model = get_model(MODEL_TYPES[(w_okay, p_okay)])
    # weather_series() builds the covariate TimeSeries from weather.data when
    # okay, and otherwise synthesizes hour/day-of-year covariates on a
    # 15-minute grid (today: OpenMeteoVlaQuery.to_model_series). phase_series()
    # builds the target series from phase.data when okay, and otherwise derives
    # a stand-in from the covariates (today: the p_ts fallback branch in
    # ModelPhaseForecast.__init__). Both helpers live in this module.
    w_ts = weather_series(weather, ctx)
    p_ts = phase_series(phase, w_ts, ctx)
    df = predict_model(model, p_ts, w_ts, n=48)
    result = Result(
        name="phase_rms_lightgbm",
        status=Status.OK,
        forecast_time=ctx.forecast_time,
        data=df,
        coverage=(df.index.min(), df.index.max()),
    )
    if model.name != "full":
        result = result.with_issue(f"fallback model: {model.name}")
    return result


phase_rms_lightgbm = Op(
    "phase_rms_lightgbm",
    _phase_rms_lightgbm,
    requires=("open_meteo_surface", "vla_phase_db"),
)
```

The branching logic that today lives in `predict_phase.py` (model-key tuple from `(w_query.okay, p_query.okay_for_model)`) becomes ordinary code inside the operation, with two mappings made explicit:

- `okay_for_model` (present, recent, complete) maps to `phase.okay` — a monitor result that is old maps to `Status.STALE`, and one with non-finite values maps to `Status.PARTIAL`; neither satisfies `okay`.
- The `to_model_series` methods on the query classes, **including their synthetic-covariate fallbacks**, move into `operations/phase_predict.py` as the `weather_series`/`phase_series` helpers. They are part of the prediction operation's policy, not part of querying.
- `phase_rms_lightgbm_long` (the 12-day forecast) has no fallback ladder: it needs days of real weather covariates, so when the weather result is not `okay` it returns `MISSING` with an issue rather than attempting a prediction — matching today's clean skip in `LongModelPhaseForecast.__init__`.

`previous_phase_forecasts` — today `Forecast.get_previous_phase_forecasts`, which reads the last 12 runs' parquet output from disk for the overplot panels — also becomes an ordinary operation. It fits the model naturally: it has a data source (the filesystem under `ctx.forecast_root`), can fail partially (some hours missing), and is consumed by plotters.

## 5. Site profiles (concrete sketches)

```python
# profiles/vla.py
from ..operations.open_meteo import (
    open_meteo_surface, open_meteo_pressure, open_meteo_ensemble,
)
from ..operations.monitor import vla_phase_db, vla_weather_station
from ..operations.herbie import herbie_tcolw, herbie_tcoli, herbie_mcc, herbie_veril
from ..operations.ndfd import ndfd
from ..operations.phase_predict import (
    phase_rms_lightgbm, phase_rms_lightgbm_long, previous_phase_forecasts,
)
from ..operations.sensitivity import am_sensitivity

VLA = SiteProfile(
    code="Y1",
    name="VLA",
    stations=(SITES_BY_NAME["Y1"],),
    pipeline=(
        open_meteo_surface,
        open_meteo_pressure,
        open_meteo_ensemble,          # ECMWF IFS ensemble PWV; feeds sensitivity
        vla_phase_db,
        vla_weather_station,
        herbie_tcolw,
        herbie_tcoli,                 # cloud ice; feeds sensitivity
        herbie_mcc,
        herbie_veril,
        ndfd,
        phase_rms_lightgbm,
        phase_rms_lightgbm_long,
        previous_phase_forecasts,
        am_sensitivity,
    ),
    plotters=(
        VlaOperatorSummary(),
        VlaWeatherSummary(),
        VlaSensitivityPlot(),
    ),
    writers=(
        VlaForecastTsv(),             # today: Forecast.write_results
    ),
)


# profiles/vlba.py
from ..operations.open_meteo import open_meteo_surface, open_meteo_multi_site

VLBA_MULTI = SiteProfile(
    code="VLBA_MULTI",
    name="VLBA (multi-site)",
    stations=tuple(VLBA_SITES),
    pipeline=(open_meteo_multi_site,),
    plotters=(VlbaMultiSitePlots(),),
)

def vlba_individual(station: Station) -> SiteProfile:
    """Shallow per-station profile."""
    return SiteProfile(
        code=station.name,
        name=f"VLBA {station.name}",
        stations=(station,),
        pipeline=(open_meteo_surface,),
        plotters=(BasicWeatherPanel(),),
    )


# profiles/gbt.py — placeholder
GBT = SiteProfile(
    code="GB",
    name="GBT",
    stations=(SITES_BY_NAME["GB"],),
    pipeline=(),   # operations to be defined when GBT analyses are designed
)
```

(The `am_sensitivity` operation consumes `open_meteo_surface`, `open_meteo_pressure`, `herbie_tcolw`, `herbie_tcoli`, and `open_meteo_ensemble` — the same inputs `VlaSensitivityEstimator` takes today — and its `requires` tuple lists all five, so `validate_profile` guarantees they run first.)

The hourly loop becomes:

```python
ALL_PROFILES = [VLA, VLBA_MULTI, *(vlba_individual(s) for s in VLBA_SITES)]

def generate() -> None:
    now = pd.Timestamp.now(tz="utc")
    for profile in ALL_PROFILES:
        ctx = build_context(profile, forecast_time=now)
        fc = Forecast(profile, ctx)
        fc.run()
        fc.write_outputs()
```

## 6. Module reorganization

Target layout:

```
src/site_forecast/
  core/
    __init__.py
    result.py            # Status, Result
    context.py           # Context, build_context
    operation.py         # Op, OperationFn, safe_run
    plotter.py           # Plotter, Writer protocols
    profile.py           # SiteProfile, validate_profile
  operations/
    __init__.py
    open_meteo.py        # surface, pressure, ensemble, multi_site
    monitor.py           # vla_phase_db, vla_weather_station (VLA-only)
    herbie.py            # tcolw, tcoli, mcc, veril
    ndfd.py
    phase_predict.py     # phase_rms_lightgbm, phase_rms_lightgbm_long,
                         # previous_phase_forecasts, weather/phase series helpers
    sensitivity.py       # am_sensitivity
  profiles/
    __init__.py          # ALL_PROFILES; validates every profile on import
    vla.py
    vlba.py
    gbt.py
  plotting/
    __init__.py
    mpl/                 # existing matplotlib figure code
    plotly/              # existing plotly figure code
    plotters.py          # Plotter classes (thin adapters over figure code)
  orchestrator.py        # Forecast, generate, loop, SafeScheduler
  __init__.py            # Station, config loader; no SITE_LAT/SITE_LON globals
```

What moves vs. stays:

- `query/` → `operations/`. `QueryBase` is replaced by `Op` functions; the abstract `forecast_time`/`okay`/`save_data` properties become fields on `Result` and a serialization concern in the orchestrator.
- `forecast.py` → `orchestrator.py`. Significantly smaller; per-site wiring lives in `profiles/`, and `write_results` becomes the `VlaForecastTsv` writer.
- `plotting/plot_mpl.py` and `plotting/plot_plotly.py` keep their figure-construction code; new `Plotter` classes wrap them and translate `dict[str, Result]` into the figure functions' arguments.
- `predict_phase.py` and `sensitivity.py` become operations. The model-selection tuple in `ModelPhaseForecast` becomes ordinary code inside the operation function, and the `to_model_series` fallbacks move with it (Section 4).
- `__init__.py` loses `SITE_LAT`, `SITE_LON`, and `VLA_SITE`. `Station`, `SITES`, `SITES_BY_NAME`, and the config loader stay.
- `query/nomads_cutout.py` is currently imported by nothing in the pipeline. It is **not** migrated; see Section 10.

## 7. Error handling principles

- **One catch site.** `safe_run` is the only place that converts an uncaught exception into a `Result`. The bare `except:` blocks in the current query constructors (`query/open_meteo.py`, `query/monitor.py`, `query/herbie_maps.py`, `query/ndfd.py`, `predict_phase.py`) and the broad `try/except` blocks inside plotting go away.
- **Status is the contract; the gate is per-consumer.** Validity is on `Result.status`, not on whether `data is None`, not on `df.attrs["has_bad"]`, not on a custom `okay_for_model` property defined only on `ApiQuery`. But "can I use this?" has more than one right answer, so consumers pick a gate from the ladder in Section 3.1: model inputs demand `okay`, ordinary plots accept `usable`, history plots accept `has_data`. A single blanket boolean was tried in the first draft and could not express the existing (and correct) behavior that stale phase measurements are plotted but never fed to the model.
- **`require` for column dependencies, gated with `okay`.** When a downstream consumer needs specific columns, it calls `result.require("col_a", "col_b")` and then checks `.okay` — not `.usable`, which a `require`-degraded `PARTIAL` result still satisfies.
- **Three failure modes, named.** `MISSING` (no data — typically a network/timeout/exception), `PARTIAL` (data present but incomplete — missing columns, non-finite values, or short coverage; today's `df.attrs["has_bad"]` and `is_complete`), `STALE` (data present but its newest timestamp is too old to be useful; today's `is_recent`). Today these are conflated. Note one deliberate behavior change: today `has_bad` makes the monitor queries entirely not-`okay`, so a single non-finite phase value suppresses the phase plots; under this design the same data is `PARTIAL` — still plotted, still excluded from the model.
- **Wiring errors are not data errors.** A profile that references an operation whose dependencies don't run earlier fails in `validate_profile` at startup. `safe_run` should essentially never see a `KeyError` from the results dict.
- **No silent skips in plotters.** A plotter that skips because an input fails its gate logs a single warning naming the operation and the issues from its `Result`. It does not log the same warning for every panel.

## 8. Migration plan

The migration keeps the hourly `generate()` cycle running at every step. Each step is independently shippable.

1. **Add `core/`.** Land `core/result.py`, `core/context.py`, `core/operation.py`, `core/plotter.py`, `core/profile.py`. Nothing imports them yet. No behavior change.
2. **Dual-emit `Result` from queries.** Every existing `QueryBase` subclass gains a `.result: Result` property derived from its current `.df` / `.ds` / `.okay` state. Call sites are unchanged. Status mapping: `df is None` → `MISSING`; `df.attrs["has_bad"]` or a failed `is_complete` → `PARTIAL`; a failed `is_recent` (monitor queries) → `STALE`; otherwise `OK`.
3. **Migrate consumers to `.result`.** One consumer at a time — `Forecast.write_results`, then each plotter — switches from `query.df`/`query.okay` to `query.result.data` plus the appropriate gate. This step changes only the *validity checks* (plotters keep their `def plot_x(fc)` signatures; those change in step 9). `predict_phase` keeps calling `to_model_series()` on the query objects for now; it migrates in step 6 when the series helpers move.
4. **Extract globals to `Context`.** `SITE_LAT`, `SITE_LON`, `VLA_SITE`, and the module-level `CONFIG` singleton are pulled into `Context` (including `forecast_root`). Query classes accept `ctx: Context` in their constructor. Old globals stay as deprecated re-exports for one cycle (a `__getattr__` on `__init__.py` that warns on access). This step precedes `safe_run` adoption because `safe_run`'s failure path needs `ctx.forecast_time` to build the `MISSING` result.
5. **Replace bare excepts with `safe_run`.** Inside each query class, the `try: … except: self.df = None` block is replaced by routing the work through `safe_run`. The `.result` property now derives from a real `Result` produced inside `safe_run`. Behavior is preserved; failures now produce `Result(status=MISSING, issues=[…])` instead of silent `None`. This also fixes the `OpenMeteoVlaPressureQuery` crash described in the introduction.
6. **Define `Op` functions.** Each existing query class is wrapped by an operation function with a declared `requires` tuple:
   ```python
   def _open_meteo_surface(ctx, results):
       return OpenMeteoVlaQuery(ctx).result

   open_meteo_surface = Op("open_meteo_surface", _open_meteo_surface)
   ```
   The phase-prediction ops absorb the `to_model_series` helpers and the four-model fallback ladder from `predict_phase.py` at this step, and `previous_phase_forecasts` is extracted from `Forecast.get_previous_phase_forecasts`. `Forecast` still wires everything by hand — the ops just exist alongside.
7. **Introduce `SiteProfile` for VLA.** `Forecast.__init__` is refactored to take a `SiteProfile` and iterate `profile.pipeline`, storing results in a `dict[str, Result]`. During this step each op continues to construct the underlying query object, and `Forecast` keeps the legacy attributes (`self.weather`, `self.phase`, …) pointing at those **query objects** — not at `Result`s — so any plotter not yet migrated in step 3, and anything else touching `.df`/`.okay`/`to_model_series()`, keeps working. `Forecast.save_data` (which delegates to each query's `save_data`) is replaced by the generic `write_serialized_results` over the results dict. `validate_profile` runs at startup from this step on.
8. **Introduce `VLBA_MULTI` and decide the output layout.** Multi-site Open-Meteo and Plotly plots move from "always on inside `Forecast.__init__`" to a separate profile, and the hourly loop invokes both profiles. This is the point where the multi-profile output layout **must** be decided — two profiles writing into the same `forecasts/<date>/<hour>/` collide. Leading candidate: a per-profile subdirectory, `forecasts/<date>/<hour>/<profile_code>/`, with the VLA profile writing at the top level for backward compatibility (or symlinked). `Context.forecast_dir` is the single point that encodes the convention.
9. **Refactor plotters to consume `(ctx, results)`.** Each plotter is converted from `def plot_x(fc):` to a `Plotter` class with a `render(ctx, results)` method, replacing `fc.weather.df...` access with `results["open_meteo_surface"]` plus the appropriate gate. This is the **largest step by volume**: `plot_all_weather` fans out to roughly 20 `plot_*` functions plus a dozen `draw_*` helpers in `plot_mpl.py`, and `plot_plotly.py` has its own entry point. It should be split into several shippable slices by figure family (phase panels, wind/weather panels, sensitivity panels, Herbie maps, VLBA Plotly). The Herbie plot functions that today take a query object (`plot_herbie_maps(hq)` etc.) take the Herbie `Result`s instead.
10. **Cleanup.** Remove deprecated aliases: `SITE_LAT`/`SITE_LON`, the `Forecast.weather` legacy attributes, `df.attrs["has_bad"]`, and the raw `.df`/`.okay`/`save_data` members on the former query classes. Add the stub `GBT` profile. Delete `QueryBase` once nothing inherits from it. `query/nomads_cutout.py` stays as-is, to be ported as an `Op` post-migration (Section 10).

After step 10, the architecture is the one described in Sections 2–6. The operational pipeline has been running successfully throughout.

## 9. Worked example: adding GBT

Adding a "deep" GBT analysis touches only `operations/` and `profiles/`:

```
src/site_forecast/
  operations/
    gbt_weather.py        # new — gbt_weather = Op("gbt_weather", ...)
    gbt_opacity.py        # new — gbt_opacity = Op("gbt_opacity", ...,
                          #        requires=("gbt_weather",))
  profiles/
    gbt.py                # populated — pipeline = (gbt_weather, gbt_opacity)
  plotting/
    plotters.py           # new GbtSummaryPlot Plotter class
```

No edits to `core/`. No edits to `vla.py` or `vlba.py`. No new branches anywhere in the orchestrator. The hourly loop picks up GBT once `GBT` is added to `ALL_PROFILES`, and `validate_profile` confirms the wiring at startup.

The same shape applies to "I want a new plot type for the VLA": write a new `Plotter` class, append it to `VLA.plotters`, done — no changes elsewhere.

## 10. Open questions and deferred decisions

- **`query/nomads_cutout.py`.** ~500 lines, currently imported by nothing in the operational pipeline; not yet run or tested. It will eventually be fully implemented, but it is ported as an `Op` only after this migration, when a consumer exists. It should not be migrated speculatively, and no tests are written for it until then (see `test_plan.md`).
- **Schema-validated `Result`.** A future evolution could attach an expected schema to each operation (column names, dtypes, minimum coverage) and have `safe_run` verify the schema before returning `OK`. The current `Result.require(*cols)` is a foothold for this; nothing in the design precludes adding it later.
- **Generalizing the LightGBM phase predictor.** The current model is VLA-specific. If a similar predictor is needed for other sites, the path is a new operation (`phase_rms_lightgbm_<site>`) with its own model pickle, not a generalized `phase_rms_lightgbm` that branches on site.
- **Parallel pipeline execution.** The `dict[str, Result]` flow — and now the explicit `requires` tuples, which form a dependency DAG — is compatible with running independent operations concurrently, but the design is sequential and there is no current need for parallelism.
- **The Open-Meteo column lists.** Three module-level lists (`COLUMNS_VLA_HR`, `COLUMNS_VLA_15`, `COLUMNS_MULTI_HR`) plus the pressure-level column list built as class attributes on `OpenMeteoVlaPressureQuery`. They become arguments to operation functions rather than module-level constants; their content is unchanged.

## 11. Original prompt

Ask for clarifying questions before giving detailed answers. The current task is to re-design the architecture of the codebase to improve separation of concerns, extensibility, and maintability, and then describe the general principles and outline an action plan into a new file, "DESIGN.md". The goal is simply to describe the proposed architecture and write it into this file, not to change the codebase.

There are two primary concerns:
(1) The current codebase initially focused on creating forecasts just for the VLA site but I am now extending it other sites, e.g., the VLBA and GBT (Green Bank Telescope).
(2) The original separation of concerns between classes and their effectiveness at encapsulation can be improved, especially for error handling and robustness.

For (1), the sites have different datasets that are available and relevant to them. The VLA has a local database on the network for current weather station measurements and the API values (found in `query.monitor.py`) as well as the NWP product queries are tuned to the forecast of the phase RMS from LightGBM (found in `predict_phase.py`). The VLA was the initial focus, but I am now expanding it to "shallower" analyses of other sites, like for the VLBA and related "multi-site" queries and plots, and potentially "deeper" analysis of other specific locations, like for the GBT (not started yet). There should likely be a more robust encapsulation of what gets queried and predicted for specific sites and what gets produced for different sites.

For (2), most code is organized into classes and modules, but a significant difficulty of using the codebase in operations is that there is not a good separation of concerns between (a) the process of querying a network endpoint, and (b) encapsulation of the result. Problems can arise both if a query fails to run at all (e.g., timeout) or it *does* complete but the data is partly invalid (e.g., missing columns, missing values, incomplete timerange). Once the query is complete, most information on the state of the result is implicit in the pandas DataFrame or xarray DataSet themselves, but not encapsulated in any way, meaning that things like plotting functions primarily operate on the DataFrames themselves, and these can be brittle. In order for the design to be robust, there should be more complete error handling. It also seems likely to me that the design should have at least some kind of separation between queries and results, but I am not sure of the best path forward. It will likely be a lot of work to provide complete encapsulation of all the information in the DataFrames and DataSets, so don't focus on encapsulating all the information on all the columns. The key is that the objects have ways of cleanly determining whether they are valid or not, and if they are not valid, then don't do things or raise errors that can be cleanly handled.
