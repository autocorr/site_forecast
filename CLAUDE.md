# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Commands

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Fix lint**: `ruff check --fix .`
- **Test**: `uv run pytest`
- **Test (single file)**: `uv run pytest path/to/test_file.py -v`
- **Test (single test)**: `uv run pytest path/to/test_file.py::test_name -v`
- **Install dev deps**: `uv sync --all-extras`
- **Build package**: `uv build`
- **Build docs**: `cd docs && make html`
- **Clean docs**: `cd docs && make clean`

Optional dependency groups: `test` (pytest), `dev` (flake8), `doc` (Sphinx),
`train` (optuna).

## Architecture

`site_forecast` is an operational weather forecasting system for radio telescope sites, with a focus on the National Radio Astronomy Observatory's Very Lage Array (VLA) and Very Long Baseline Array (VLBA).  It ingests numerical weather prediction (NWP) data from multiple sources, runs a machine learning model to predict the atmospheric phase stability, writes data out in parquet, TSV, and netcdf formats, and generatesdiagnostic plots using matplotlib and Plotly. Forecasts are generated on an hourly schedule using the `schedule` package.

### Data Flow

```
Weather Sources → Forecast (orchestrator) → ModelPhaseForecast (LightGBM) → TSV + plots
```

**Query modules** (`src/site_forecast/query/`) each subclass `QueryBase` and expose a `.query()` method returning a pandas DataFrame:
- `open_meteo.py` — Open-Meteo API; `OpenMeteoVlaQuery` for the VLA site, `OpenMeteoMultiSiteQuery` for all VLBA sites
- `herbie_maps.py` — HRRR/GFS cutout maps via Herbie (cloud water, ice, cover fields)
- `monitor.py` — Real-time phase RMS from the VLA monitor PostgreSQL database
- `ndfd.py` — NOAA National Digital Forecast Database
- `nomads_cutout.py` — NOMADS model output

**Orchestration** (`forecast.py`): `Forecast` instantiates all queries, calls `ModelPhaseForecast` and `LongModelPhaseForecast`, writes `forecast.tsv`, saves `.parquet` files, generates plots, and creates symlinks to the latest outputs. `generate()` runs a single full cycle; `loop()` runs `generate()` on a `SafeScheduler` every hour at `:30`.

**Phase prediction** (`predict_phase.py`): `ModelPhaseForecast` produces a 48-step (12-hour) forecast; `LongModelPhaseForecast` produces 288 steps (12 days). Both use pre-trained LightGBM models from `models/`. Model selection is automatic based on data availabilitys.

**Atmospheric radiative transfer** (`sensitivity.py`): wraps the external `am` model via `amwrap` to compute opacity and sky brightness temperature.

**Plotting** (`plotting/`): `plot_mpl.py` generates static Matplotlib figures; `plot_plotly.py` generates interactive HTML figures for multi-site VLBA data.

**Station definitions** (`__init__.py`): defines 12 `Station` objects (VLA Y1 + 10 VLBA sites) with geodetic coordinates, elevation, and sunrise/sunset helpers. Also handles loading `site_forecast.ini` config (searched in cwd, `~/.site_forecast.ini`, `~/.config/site_forecast.ini`).

### Runtime configuration

`site_forecast.ini` (not committed) specifies output directories (`forecasts/`, `plots/`, `logs/`) and `log_level`. Do not commit this file.

### Pre-trained models

Four `.pkl` LightGBM model files live in `models/` and are tracked in git.  Training utilities are in `src/site_forecast/train/` (`readdata.py` for preprocessing, `__init__.py` for `DataSet`/`TrainingData` classes; hyperparameter search uses Optuna via the `train` optional dependency group).

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
