"""DataFrame contracts for each query class.

Two consumers:
1. Offline tests validate cached fixtures against these schemas.
2. Live schema tests (`-m schema`) validate real upstream responses against
   the same schemas — primary detection mechanism for upstream drift.

Schemas describe structural shape (columns, dtypes, index) only. Value
ranges are intentionally not constrained.
"""

from typing import Iterable

import pandera.pandas as pa
import xarray as xr


_OPEN_METEO_HOURLY_COLS = [
    "total_column_integrated_water_vapour",
    "boundary_layer_height",
    "lifted_index",
    "convective_inhibition",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "weather_code",
    "rain",
    "showers",
    "snowfall",
    "precipitation_probability",
]

_OPEN_METEO_MINUTELY_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_gusts_10m",
    "precipitation",
    "freezing_level_height",
    "cape",
    "visibility",
    "direct_radiation",
    "diffuse_radiation",
]

_OPEN_METEO_MULTI_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation_probability",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "boundary_layer_height",
    "total_column_integrated_water_vapour",
]

_DERIVED_TIME_COLS = {
    "mjd": pa.Column("float64"),
    "hour": pa.Column("float32"),
    "day_of_year": pa.Column("int32"),
}


def _float32_cols(names: Iterable[str]) -> dict:
    return {name: pa.Column("float32") for name in names}


OpenMeteoVlaSchema = pa.DataFrameSchema(
    columns={
        "site": pa.Column(str),
        **_float32_cols(_OPEN_METEO_HOURLY_COLS),
        **_float32_cols(_OPEN_METEO_MINUTELY_COLS),
        **_DERIVED_TIME_COLS,
    },
    index=pa.Index("datetime64[ns, UTC]", name="date"),
    strict=True,
)


OpenMeteoMultiSiteSchema = pa.DataFrameSchema(
    columns={
        **_float32_cols(_OPEN_METEO_MULTI_COLS),
        **_DERIVED_TIME_COLS,
    },
    index=pa.MultiIndex(
        [
            pa.Index("datetime64[ns, UTC]", name="date"),
            pa.Index(str, name="site"),
        ]
    ),
    strict=True,
)


OpenMeteoVlaPressureSchema = pa.DataFrameSchema(
    columns={
        "temperature": pa.Column("float32"),
        "relative_humidity": pa.Column("float32"),
    },
    index=pa.MultiIndex(
        [
            pa.Index("datetime64[ns, UTC]", name="date"),
            pa.Index("int64", name="pressure"),
        ]
    ),
    strict=True,
)


OpenMeteoVlaEnsembleSchema = pa.DataFrameSchema(
    columns={
        "total_column_integrated_water_vapour": pa.Column("float32"),
    },
    index=pa.MultiIndex(
        [
            pa.Index("datetime64[ns, UTC]", name="date"),
            pa.Index("int64", name="member"),
        ]
    ),
    strict=True,
)


NdfdSchema = pa.DataFrameSchema(
    columns={
        "wind_direction": pa.Column("float64"),
        "wind_gust": pa.Column("float64"),
        "wind_speed": pa.Column("float64"),
        "dewpoint_temperature": pa.Column("float64"),
        "temperature": pa.Column("float64"),
    },
    index=pa.Index("datetime64[ns, UTC]", name="hour"),
    strict=True,
)


ApiQuerySchema = pa.DataFrameSchema(
    columns={
        "mjd": pa.Column("float64"),
        **{f"rms_phase{n}": pa.Column("float64") for n in range(6)},
        "rms_phase_for_ost": pa.Column("float64"),
        "phase_rms_avg": pa.Column("float64"),
        "phase_rms_med": pa.Column("float64"),
        "phase_rms": pa.Column("float64"),
    },
    index=pa.Index("datetime64[ns]", name="time"),
    strict=True,
)


WeatherStationSchema = pa.DataFrameSchema(
    columns={
        "mjd": pa.Column("float64"),
        "temperature": pa.Column("float64"),
        "dewpoint_temperature": pa.Column("float64"),
        "pressure": pa.Column("float64"),
        "pyranometer_2": pa.Column("float64"),
        "relative_humidity": pa.Column("float64"),
        "wind_direction_average": pa.Column("float64"),
        "wind_direction_maximum": pa.Column("float64"),
        "wind_direction_minimum": pa.Column("float64"),
        "wind_speed_average": pa.Column("float64"),
        "wind_speed_maximum": pa.Column("float64"),
        "wind_speed_minimum": pa.Column("float64"),
    },
    index=pa.Index("datetime64[ns]", name="time"),
    strict=True,
)


def assert_herbie_dataset(ds: xr.Dataset, var: str) -> None:
    """Assert that an xarray Dataset matches the HerbieQuery contract for the
    given variable name (e.g., 'tcolw', 'tcoli', 'mcc', 'veril')."""
    expected_dims = {"date", "step", "y", "x", "quantile", "radius"}
    missing_dims = expected_dims - set(ds.sizes)
    if missing_dims:
        raise AssertionError(f"Missing dims: {missing_dims}; have {dict(ds.sizes)}")
    if ds.sizes["quantile"] != 21:
        raise AssertionError(f"quantile dim should be 21, got {ds.sizes['quantile']}")
    if ds.sizes["radius"] != 2:
        raise AssertionError(f"radius dim should be 2, got {ds.sizes['radius']}")

    expected_vars = {var, f"{var}_q", f"{var}_p", f"{var}_m", f"{var}_c"}
    missing_vars = expected_vars - set(ds.data_vars)
    if missing_vars:
        raise AssertionError(
            f"Missing data_vars for {var}: {missing_vars}; have {list(ds.data_vars)}"
        )

    expected_dtypes = {
        var: "float32",
        f"{var}_q": "float64",
        f"{var}_p": "float64",
        f"{var}_m": "float32",
        f"{var}_c": "float64",
    }
    for name, expected in expected_dtypes.items():
        actual = ds[name].dtype.name
        if actual != expected:
            raise AssertionError(f"{name} dtype should be {expected}, got {actual}")
