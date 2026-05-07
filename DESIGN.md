# site_forecast — Architecture Redesign

This document describes a target architecture for `site_forecast` and an incremental migration path from the current code. It is a design proposal, not a code change.

The redesign is driven by two pressures:

1. **Multi-site expansion.** The codebase started as a VLA-only forecaster and is being extended to "shallow" multi-site VLBA analyses and (eventually) deeper analyses for sites such as the GBT. Site-specific behavior is currently scattered across module-level globals (`SITE_LAT`, `SITE_LON`), VLA-only modules (`query/monitor.py`), and unconditional construction inside `Forecast.__init__`.
2. **Query/result entanglement.** Today a "query" class owns the network call, the error swallowing (`except: self.df = None`), the result, and a thin `okay` flag whose semantics differ between subclasses. Validity is implicit in the DataFrame's contents. Plotters and the phase predictor reach into `.df` directly, so a missing column or a partial time range silently breaks downstream code or hides inside a broad `try/except`.

The goal is an architecture where:

- Sites are composed, not subclassed.
- Failures are first-class data on a result object, not exceptions or `None`s.
- Adding a new analysis or a new site is a local change, not a core refactor.

## 1. Goals and non-goals

### Goals

- **Composable sites.** Each site (VLA, VLBA-multi, individual VLBA stations, future GBT) is described by a `SiteProfile` that lists which named operations to run and which plotters to render. The orchestrator does not branch on site identity.
- **Explicit result validity.** Every operation returns a `Result` whose `status` field is the source of truth for "can downstream code use this?". DataFrame state is no longer a hidden contract.
- **Best-effort orchestration.** Each operation runs independently; one failure does not abort the run. Downstream operations and plotters check `Result.usable` and skip cleanly with a logged reason.
- **Incremental migration.** Every step of the migration leaves the hourly operational pipeline working. There is no flag day.

### Non-goals

- Rewriting plotting from scratch. Existing Matplotlib and Plotly figure code is preserved; only its inputs change.
- Full per-column schema validation (pandera/pydantic). The `Result` wrapper is forward-compatible with that, but it is not part of this design.
- Hiding the underlying DataFrame/Dataset behind named domain accessors. `.data` is still exposed.
- Generalizing the LightGBM phase RMS predictor to non-VLA sites. It stays a VLA-only registered operation.
- Parallel execution within a pipeline. Sequential is fine and matches current behavior.
- Choosing the on-disk output layout for multi-site forecasts. Deferred (see Section 10).

## 2. Architectural pillars

### A. Operation registry

Every unit of work — querying Open-Meteo, fetching HRRR via Herbie, running the LightGBM phase predictor, computing AM sensitivity — is a **named operation** registered into a shared registry. Operations are addressed by string names (e.g., `"open_meteo_surface"`, `"vla_phase_db"`, `"phase_rms_lightgbm"`).

### B. Lightweight `Result`

Operations return a `Result`: a small dataclass carrying a `status` enum, the underlying data (DataFrame, Dataset, or `None`), an `issues` list of human-readable strings, the `coverage` time window of the data, and the `forecast_time`. The DataFrame stays accessible via `result.data`. No schema validation, no domain-specific accessors.

### C. Site profiles

A `SiteProfile` is a pure-data description of a site: its code, location, the ordered list of operation names that make up its pipeline, and the list of plotters to render. There is no `VlaSite` subclass.

### D. Best-effort orchestration

The orchestrator runs each pipeline step inside a `safe_run` wrapper that converts any uncaught exception into a `Result(status=MISSING, …)`. The orchestrator never raises mid-pipeline. Plotters and downstream operations are responsible for checking `Result.usable` and degrading gracefully when an upstream input is missing.

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
        return self.status is Status.OK

    @property
    def usable(self) -> bool:
        # OK or PARTIAL: there is some data; downstream may choose to consume it.
        return self.status in (Status.OK, Status.PARTIAL)

    def with_issue(self, msg: str) -> "Result":
        return Result(self.name, self.status, self.forecast_time,
                      self.data, [*self.issues, msg], self.coverage)

    def degraded_to(self, status: Status, msg: str) -> "Result":
        return Result(self.name, status, self.forecast_time,
                      self.data, [*self.issues, msg], self.coverage)

    def require(self, *columns: str) -> "Result":
        """Return self, or a degraded copy if any required column is missing."""
        if self.data is None:
            return self
        missing = [c for c in columns if c not in self.data.columns]
        if missing:
            return self.degraded_to(Status.PARTIAL, f"missing columns: {missing}")
        return self
```

`Result.require` is the standard pattern for downstream consumers to declare what they need:

```python
weather = results["open_meteo_surface"].require("wind_speed_10m", "temperature_2m")
if not weather.usable:
    logger.warning("skipping wind plot: %s", weather.issues)
    return
```

### 3.2 `Context`

```python
# core/context.py
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .profile import SiteProfile


@dataclass(frozen=True)
class Context:
    """Per-run context. Replaces module-level CONFIG, SITE_LAT, SITE_LON."""
    profile: SiteProfile
    forecast_time: pd.Timestamp
    forecast_dir: Path                     # output dir for this run
    cache_dir: Path
    config: "Config"                       # parsed site_forecast.ini snapshot

    @property
    def latitude(self):
        return self.profile.location.latitude

    @property
    def longitude(self):
        return self.profile.location.longitude
```

The `Context` carries the bits a query needs that today are pulled from globals or the `CONFIG` singleton. Operations receive a `Context` argument; they do not import `SITE_LAT`/`SITE_LON` or read `CONFIG` directly.

### 3.3 `Operation`, `OperationRegistry`, `safe_run`

```python
# core/operation.py
from typing import Callable, Protocol
import logging
import traceback

from .result import Result, Status
from .context import Context

logger = logging.getLogger(__name__)

OperationFn = Callable[[Context, dict[str, Result]], Result]


class Operation(Protocol):
    name: str
    def run(self, ctx: Context, results: dict[str, Result]) -> Result: ...


class OperationRegistry:
    def __init__(self):
        self._fns: dict[str, OperationFn] = {}

    def register(self, name: str):
        def decorator(fn: OperationFn) -> OperationFn:
            if name in self._fns:
                raise ValueError(f"operation {name!r} already registered")
            self._fns[name] = fn
            return fn
        return decorator

    def invoke(self, name: str, ctx: Context,
               results: dict[str, Result]) -> Result:
        fn = self._fns[name]
        return safe_run(name, fn, ctx, results)


def safe_run(name: str, fn: OperationFn, ctx: Context,
             results: dict[str, Result]) -> Result:
    """Execute fn; convert any uncaught exception into a MISSING Result."""
    try:
        return fn(ctx, results)
    except Exception as exc:
        logger.exception("operation %s failed", name)
        return Result(
            name=name,
            status=Status.MISSING,
            forecast_time=ctx.forecast_time,
            issues=[f"{type(exc).__name__}: {exc}"],
        )


REGISTRY = OperationRegistry()
```

`safe_run` is the single place where exceptions are caught and translated into `Status.MISSING`. The bare-except blocks scattered across `query/open_meteo.py`, `query/herbie_maps.py`, etc. are replaced by this one wrapper.

### 3.4 `Plotter`

```python
# core/plotter.py
from typing import Protocol
from .result import Result
from .context import Context


class Plotter(Protocol):
    name: str
    requires: list[str]                    # operation names

    def render(self, ctx: Context, results: dict[str, Result]) -> None: ...
```

A plotter declares which operation results it consumes. The orchestrator passes the full `results` dict; the plotter's `render` is responsible for checking `usable` on its requirements and skipping cleanly. Plotters no longer take a `Forecast` object — that decouples them from the orchestrator and lets a plotter be unit-tested with a hand-built `dict[str, Result]`.

### 3.5 `SiteProfile`

```python
# core/profile.py
from dataclasses import dataclass, field
from .plotter import Plotter


@dataclass(frozen=True)
class SiteLocation:
    latitude: float        # decimal degrees
    longitude: float
    elevation: float       # metres


@dataclass(frozen=True)
class SiteProfile:
    code: str              # "Y1", "PT", "VLBA_MULTI", ...
    name: str              # human-readable
    location: SiteLocation | None          # None for multi-site profiles
    pipeline: list[str]                    # ordered operation names
    plotters: list[Plotter] = field(default_factory=list)
```

Every site-specific decision is encoded in the `SiteProfile`'s `pipeline` and `plotters`. Adding a site is writing a new profile; it is never a change to the core.

### 3.6 `Forecast` orchestrator

```python
# orchestrator.py
from .core.context import Context
from .core.profile import SiteProfile
from .core.operation import REGISTRY
from .core.result import Result


class Forecast:
    def __init__(self, profile: SiteProfile, ctx: Context):
        self.profile = profile
        self.ctx = ctx
        self.results: dict[str, Result] = {}

    def run(self) -> None:
        for op_name in self.profile.pipeline:
            self.results[op_name] = REGISTRY.invoke(op_name, self.ctx, self.results)

    def write_outputs(self) -> None:
        for plotter in self.profile.plotters:
            try:
                plotter.render(self.ctx, self.results)
            except Exception:
                logger.exception("plotter %s failed", plotter.name)
        write_serialized_results(self.ctx, self.results)
        write_forecast_tsv(self.ctx, self.results)
```

The orchestrator is small and site-agnostic. `loop()` schedules `Forecast(profile, ctx).run()` for each profile that needs to run hourly.

## 4. Data flow

```
                      SiteProfile
                          │
                          ▼
                Pipeline (ordered op names)
                          │
                          ▼
            REGISTRY.invoke(op, ctx, results)
                          │   (each call wrapped in safe_run)
                          ▼
                Result(status, data, issues, coverage)
                          │
                          ▼
                results: dict[str, Result]
                  │                 │
                  ▼                 ▼
              Plotters       Serializers (parquet / tsv / netcdf)
                  │
                  ▼
                Outputs
```

A pipeline step that depends on an earlier step reads it from `results`:

```python
@REGISTRY.register("phase_rms_lightgbm")
def phase_rms_lightgbm(ctx: Context, results: dict[str, Result]) -> Result:
    weather = results["open_meteo_surface"].require("wind_speed_10m", "temperature_2m")
    phase   = results["vla_phase_db"]
    if not weather.usable and not phase.usable:
        return Result(
            name="phase_rms_lightgbm",
            status=Status.MISSING,
            forecast_time=ctx.forecast_time,
            issues=["no usable inputs"],
        )
    model_key = ("full" if weather.okay else "no_weather",
                 phase.usable)
    df = predict_phase(model_key, weather.data, phase.data, ctx)
    return Result(
        name="phase_rms_lightgbm",
        status=Status.OK,
        forecast_time=ctx.forecast_time,
        data=df,
        coverage=(df.index.min(), df.index.max()),
    )
```

The branching logic that today lives in `predict_phase.py` (model-key tuple from `(w_query.okay, p_query.okay_for_model)`) becomes ordinary code inside the operation, but its inputs are now explicit `Result` objects with explicit status, not implicit booleans.

## 5. Site profiles (concrete sketches)

```python
# profiles/vla.py
VLA = SiteProfile(
    code="Y1",
    name="VLA",
    location=SiteLocation(34.0773880, -107.6156450, 2115.0),
    pipeline=[
        "open_meteo_surface",
        "open_meteo_pressure",
        "vla_phase_db",
        "vla_weather_station",
        "herbie_tcolw",
        "herbie_mcc",
        "herbie_veril",
        "ndfd",
        "phase_rms_lightgbm",
        "phase_rms_lightgbm_long",
        "am_sensitivity",
    ],
    plotters=[
        VlaOperatorSummary(),
        VlaWeatherSummary(),
        VlaSensitivityPlot(),
    ],
)


# profiles/vlba.py
VLBA_MULTI = SiteProfile(
    code="VLBA_MULTI",
    name="VLBA (multi-site)",
    location=None,
    pipeline=["open_meteo_multi_site"],
    plotters=[VlbaMultiSitePlots()],
)

def vlba_individual(station: Station) -> SiteProfile:
    """Shallow per-station profile."""
    return SiteProfile(
        code=station.name,
        name=f"VLBA {station.name}",
        location=SiteLocation(station.lat, station.lon, station.elev),
        pipeline=["open_meteo_surface"],
        plotters=[BasicWeatherPanel()],
    )


# profiles/gbt.py — placeholder
GBT = SiteProfile(
    code="GB",
    name="GBT",
    location=SiteLocation(38.4331222, -79.8398361, 824.0),
    pipeline=[
        # operations to be defined when GBT analyses are designed
    ],
    plotters=[],
)
```

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
    operation.py         # Operation, OperationRegistry, safe_run, REGISTRY
    plotter.py           # Plotter protocol
    profile.py           # SiteLocation, SiteProfile
  operations/
    __init__.py          # imports modules to trigger @REGISTRY.register decorators
    open_meteo.py        # surface, pressure, multi_site
    monitor.py           # vla_phase_db, vla_weather_station (VLA-only)
    herbie.py            # tcolw, tcoli, mcc, veril
    ndfd.py
    phase_predict.py     # phase_rms_lightgbm, phase_rms_lightgbm_long
    sensitivity.py       # am_sensitivity
  profiles/
    __init__.py
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

- `query/` → `operations/`. `QueryBase` is replaced by the `Operation` protocol; the abstract `forecast_time`/`okay`/`save_data` properties become fields on `Result` and a serialization concern in the orchestrator.
- `forecast.py` → `orchestrator.py`. Significantly smaller; per-site wiring lives in `profiles/`.
- `plotting/plot_mpl.py` and `plotting/plot_plotly.py` keep their figure-construction code; new `Plotter` classes wrap them and translate `dict[str, Result]` into the figure functions' arguments.
- `predict_phase.py` and `sensitivity.py` become operations. The model-selection tuple in `ModelPhaseForecast` becomes ordinary code inside the operation function.
- `__init__.py` loses `SITE_LAT`, `SITE_LON`, and `VLA_SITE`. `Station`, `SITES`, `SITES_BY_NAME`, and the config loader stay.

## 7. Error handling principles

- **One catch site.** `safe_run` is the only place that converts an uncaught exception into a `Result`. The bare `except:` blocks in the current `query/open_meteo.py:212`, `query/herbie_maps.py:361`, `predict_phase.py`, and the broad `try/except` in `plotting/plot_mpl.py:draw_band_limit_strip` go away.
- **Status is the contract.** Validity is on `Result.status`, not on whether `data is None`, not on `df.attrs["has_bad"]`, not on a custom `okay_for_model` property defined only on `ApiQuery`. Any caller that wants to know "can I use this?" reads `result.usable`.
- **`require` for column dependencies.** When a downstream consumer needs specific columns, it calls `result.require("col_a", "col_b")` and gets back either the original `OK` result or a degraded copy with `Status.PARTIAL` and an issue describing what's missing. Plotters then check `usable` and skip cleanly.
- **Three failure modes, named.** `MISSING` (no data — typically a network/timeout/exception), `PARTIAL` (data present but incomplete — missing columns or short coverage), `STALE` (data present but its newest timestamp is too old to be useful, e.g., the VLA monitor DB last reported six hours ago). Today these are conflated.
- **No silent skips in plotters.** A plotter that skips because an input is not usable logs a single warning naming the missing operation and the issues from its `Result`. It does not log the same warning for every panel.

## 8. Migration plan

The migration keeps the hourly `generate()` cycle running at every step. Each step is independently shippable.

1. **Add `core/`.** Land `core/result.py`, `core/context.py`, `core/operation.py`, `core/plotter.py`, `core/profile.py`. Nothing imports them yet. No behavior change.
2. **Dual-emit `Result` from queries.** Every existing `QueryBase` subclass gains a `.result: Result` property derived from its current `.df` / `.ds` / `.okay` state. Call sites are unchanged. `df.attrs["has_bad"]` becomes the source for `Status.PARTIAL` vs `Status.OK`.
3. **Migrate consumers to `.result`.** One consumer at a time — `Forecast.write_results`, `predict_phase.ModelPhaseForecast`, then each plotter — switches from `query.df`/`query.okay` to `query.result.data`/`query.result.usable`. Old fields stay in place.
4. **Replace bare excepts with `safe_run`.** Inside each query class, the `try: … except: self.df = None` block is replaced by routing the work through `safe_run`. The `.result` property now derives from a real `Result` produced inside `safe_run`. Behavior is preserved; failures now produce `Result(status=MISSING, issues=[…])` instead of silent `None`.
5. **Extract globals to `Context`.** `SITE_LAT`, `SITE_LON`, `VLA_SITE`, and the module-level `CONFIG` singleton are pulled into `Context`. Query classes accept `ctx: Context` in their constructor. Old globals stay as deprecated re-exports for one cycle (a `__getattr__` on `__init__.py` that warns on access).
6. **Introduce `OperationRegistry`.** Each existing query class is wrapped by an operation function:
   ```python
   @REGISTRY.register("open_meteo_surface")
   def open_meteo_surface(ctx, results):
       return OpenMeteoVlaQuery(ctx).result
   ```
   `Forecast` still wires queries by hand — the registry just exists alongside.
7. **Introduce `SiteProfile` for VLA.** `Forecast.__init__` is refactored to take a `SiteProfile` and iterate `profile.pipeline`, storing results in a `dict[str, Result]`. The old attributes (`self.weather`, `self.phase`, …) stay as views over the dict (`@property` accessors that read `self.results["open_meteo_surface"]`). Existing plotter code is unaffected.
8. **Introduce `VLBA_MULTI_PROFILE`.** Multi-site Open-Meteo and Plotly plots move from "always on inside `Forecast.__init__`" to a separate profile. The hourly loop now invokes both profiles.
9. **Refactor plotters to consume `(ctx, results)`.** Each plotter is converted from `def plot_x(fc):` to a `Plotter` class with a `render(ctx, results)` method. The `fc.weather.df.wind_speed_10m * KMHOUR_TO_MS` style accesses are replaced with `results["open_meteo_surface"].require("wind_speed_10m").data["wind_speed_10m"] * KMHOUR_TO_MS` and a `usable` check at the top.
10. **Cleanup.** Remove deprecated aliases: `SITE_LAT`/`SITE_LON`, the `Forecast.weather` view properties, `df.attrs["has_bad"]`, and the raw `.df`/`.okay` attributes on operation classes. Add the stub `GBT_PROFILE`. Delete `QueryBase` once nothing inherits from it.

After step 10, the architecture is the one described in Sections 2–6. The operational pipeline has been running successfully throughout.

## 9. Worked example: adding GBT

Adding a "deep" GBT analysis touches only `operations/` and `profiles/`:

```
src/site_forecast/
  operations/
    gbt_weather.py        # new — @REGISTRY.register("gbt_weather")
    gbt_opacity.py        # new — @REGISTRY.register("gbt_opacity")
  profiles/
    gbt.py                # populated — pipeline = ["gbt_weather", "gbt_opacity"]
  plotting/
    plotters.py           # new GbtSummaryPlot Plotter class
```

No edits to `core/`. No edits to `vla.py` or `vlba.py`. No new branches anywhere in the orchestrator. The hourly loop picks up GBT once `GBT` is added to `ALL_PROFILES`.

The same shape applies to "I want a new plot type for the VLA": register a new `Plotter` class, append it to `VLA.plotters`, done — no changes elsewhere.

## 10. Open questions and deferred decisions

- **On-disk output layout for multi-site forecasts.** Per-site subdirectory (`forecasts/<timestamp>/<site_code>/…`) vs. site-prefixed flat files. Deferred to implementation time. The `Context.forecast_dir` field is the single point that needs to encode the chosen convention.
- **Schema-validated `Result`.** A future evolution could attach an expected schema to each operation (column names, dtypes, minimum coverage) and have `safe_run` verify the schema before returning `OK`. The current `Result.require(*cols)` is a foothold for this; nothing in the design precludes adding it later.
- **Generalizing the LightGBM phase predictor.** The current model is VLA-specific. If a similar predictor is needed for other sites, the path is a new operation (`phase_rms_lightgbm_<site>`) with its own model pickle, not a generalized `phase_rms_lightgbm` that branches on site.
- **Parallel pipeline execution.** The `dict[str, Result]` flow is compatible with running independent operations concurrently, but the design is sequential and there is no current need for parallelism.
- **The four Open-Meteo column lists** (`COLUMNS_VLA_HR`, `COLUMNS_VLA_15`, `COLUMNS_MULTI_HR`, `COLUMNS_MULTI_PR`). They become arguments to operation functions rather than module-level constants, but their content is unchanged.

## 10. Original prompt

Ask for clarifying questions before giving detailed answers. The current task is to re-design the architecture of the codebase to improve separation of concerns, extensibility, and maintability, and then describe the general principles and outline an action plan into a new file, "DESIGN.md". The goal is simply to describe the proposed architecture and write it into this file, not to change the codebase.

There are two primary concerns:
(1) The current codebase initially focused on creating forecasts just for the VLA site but I am now extending it other sites, e.g., the VLBA and GBT (Green Bank Telescope).
(2) The original separation of concerns between classes and their effectiveness at encapsulation can be improved, especially for error handling and robustness.

For (1), the sites have different datasets that are available and relevant to them. The VLA has a local database on the network for current weather station measurements and the API values (found in `query.monitor.py`) as well as the NWP product queries are tuned to the forecast of the phase RMS from LightGBM (found in `predict_phase.py`). The VLA was the initial focus, but I am now expanding it to "shallower" analyses of other sites, like for the VLBA and related "multi-site" queries and plots, and potentially "deeper" analysis of other specific locations, like for the GBT (not started yet). There should likely be a more robust encapsulation of what gets queried and predicted for specific sites and what gets produced for different sites.

For (2), most code is organized into classes and modules, but a significant difficulty of using the codebase in operations is that there is not a good separation of concerns between (a) the process of querying a network endpoint, and (b) encapsulation of the result. Problems can arise both if a query fails to run at all (e.g., timeout) or it *does* complete but the data is partly invalid (e.g., missing columns, missing values, incomplete timerange). Once the query is complete, most information on the state of the result is implicit in the pandas DataFrame or xarray DataSet themselves, but not encapsulated in any way, meaning that things like plotting functions primarily operate on the DataFrames themselves, and these can be brittle. In order for the design to be robust, there should be more complete error handling. It also seems likely to me that the design should have at least some kind of separation between queries and results, but I am not sure of the best path forward. It will likely be a lot of work to provide complete encapsulation of all the information in the DataFrames and DataSets, so don't focus on encapsulating all the information on all the columns. The key is that the objects have ways of cleanly determining whether they are valid or not, and if they are not valid, then don't do things or raise errors that can be cleanly handled.
