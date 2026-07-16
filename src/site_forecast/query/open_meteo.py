from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from darts import TimeSeries
from pandas import Timestamp
from astropy.time import Time

import requests_cache
import openmeteo_requests
from retry_requests import retry

from . import (
    QueryBase,
    timeseries_from_dataframe,
    to_parquet,
)
from .. import SITES, SITES_BY_NAME, Station, logger
from ..train import to_training_subset


CACHE_SESSION = requests_cache.CachedSession(".cache", expire_after=3600)
RETRY_SESSION = retry(
    CACHE_SESSION,
    retries=5,
    backoff_factor=1,
    status_to_retry=(429, 500, 502, 503, 504),
)
API_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
VLA_SITE = SITES_BY_NAME["Y1"]

PRESSURE_LEVELS = [
    1000,
    975,
    950,
    925,
    900,
    875,
    850,
    825,
    800,
    775,
    750,
    725,
    700,
    675,
    650,
    625,
    600,
    575,
    550,
    525,
    500,
    475,
    450,
    425,
    400,
    375,
    350,
    325,
    300,
    275,
    250,
    225,
    200,
    175,
    150,
    125,
    100,
    70,
    50,
    40,
    30,
    20,
    15,
    10,
]

COLUMNS_VLA_HR = [
    # Quantities used in phase forecast
    "total_column_integrated_water_vapour",
    "boundary_layer_height",
    "lifted_index",
    "convective_inhibition",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # "sensible_heat_flux",
    # "latent_heat_flux",
    # Precipitation, storm quantities
    "weather_code",
    "rain",
    "showers",
    "snowfall",
    "precipitation_probability",
]

COLUMNS_VLA_15 = [
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

COLUMNS_MULTI_HR = [
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


def date_range_from_response_interval(interval):
    return pd.date_range(
        start=pd.to_datetime(interval.Time(), unit="s", utc=True),
        end=pd.to_datetime(interval.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=interval.Interval()),
        inclusive="left",
    )


def response_to_pandas(response, columns, kind="hourly") -> pd.DataFrame:
    if kind == "hourly":
        wrapped = response.Hourly()
    elif kind == "minutely":
        wrapped = response.Minutely15()
    else:
        raise ValueError(f"Invalid {kind=}")
    data = {
        label: wrapped.Variables(i).ValuesAsNumpy() for i, label in enumerate(columns)
    }
    data.update({"date": date_range_from_response_interval(wrapped)})
    return pd.DataFrame(data).set_index("date")


def parse_response(
    responses,
    names,
    hourly_columns=None,
    minutely_columns=None,
) -> pd.DataFrame:
    if hourly_columns is None and minutely_columns is None:
        raise ValueError(
            "At least one of either hourly or minutely columns must be specified."
        )
    all_df = []
    for response, name in zip(responses, names):
        # Parse data from the response.
        if hourly_columns:
            h_df = response_to_pandas(response, hourly_columns, kind="hourly")
        if minutely_columns:
            m_df = response_to_pandas(response, minutely_columns, kind="minutely")
        # Merge time series data as needed.
        if hourly_columns and minutely_columns:
            h_df = h_df.resample("15min").asfreq().interpolate()
            df = m_df.merge(h_df, left_index=True, right_index=True)
        elif hourly_columns:
            df = h_df
        elif minutely_columns:
            df = m_df
        # Add time quantities
        time = Time(df.index)
        df["mjd"] = time.mjd
        df["site"] = name
        df["hour"] = (df.index.hour + df.index.minute / 60).astype(np.float32)
        df["day_of_year"] = df.index.dayofyear.astype(np.int32)
        all_df.append(df)
    m_df = pd.concat(all_df).reset_index().set_index(["date", "site"])
    m_df.attrs["has_bad"] = int(np.any(~np.isfinite(m_df)))
    return m_df


def request_data(
    n_days: int = 5,
    sites: List[Station] = [VLA_SITE],
    hourly_columns: List[str] = COLUMNS_VLA_HR,
    minutely_columns: List[str] = COLUMNS_VLA_15,
    n_past_days: int = 1,
):
    if n_days > 16:
        raise ValueError("Forecast days must be 16 days or fewer.")
    lats = [s.latitude.to("deg").value for s in sites]
    lons = [s.longitude.to("deg").wrap_at("180 deg").value for s in sites]
    names = [s.name for s in sites]
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": hourly_columns,
        "minutely_15": minutely_columns,
        "models": "gfs_seamless",
        "forecast_days": n_days,
        "past_days": n_past_days,
    }
    openmeteo = openmeteo_requests.Client(session=RETRY_SESSION)
    responses = openmeteo.weather_api(API_URL, params=params)
    df = parse_response(
        responses,
        names,
        hourly_columns=hourly_columns,
        minutely_columns=minutely_columns,
    )
    logger.info(f"Weather: (N={df.shape[0]}, has_bad={df.attrs['has_bad']})")
    return df


def unpivot_pressure_levels(
    df: pd.DataFrame,
    pressure_columns: List[str],
) -> pd.DataFrame:
    keep_cols = ["date"] + [c for c in df.columns if "hPa" in c]
    p_df = pd.wide_to_long(
        df.reset_index()[keep_cols],
        stubnames=pressure_columns,
        i=["date"],
        j="pressure",
        sep="_",
        suffix=r"\d+hPa",
    )
    pressures = (
        pd.to_numeric(p_df.index.get_level_values("pressure").str.replace("hPa", ""))
        .unique()
        .sort_values()
    )  # ascending 10 -> 825 hPa
    p_df.index = p_df.index.set_levels(pressures, level="pressure")
    return p_df.sort_index()


def parse_ensemble_response(responses, names, variable: str) -> pd.DataFrame:
    site_dfs = []
    for response, name in zip(responses, names):
        hourly = response.Hourly()
        dates = date_range_from_response_interval(hourly)
        member_data = {}
        for i in range(hourly.VariablesLength()):
            var = hourly.Variables(i)
            member_data[var.EnsembleMember()] = var.ValuesAsNumpy()
        df = pd.DataFrame(member_data, index=dates)
        df.index.name = "date"
        df.columns.name = "member"
        site_dfs.append(df.stack().rename(variable).to_frame())
    df = (
        pd.concat(site_dfs, keys=names, names=["site"])
        .reorder_levels(["date", "site", "member"])
        .sort_index()
    )
    df.attrs["has_bad"] = int(np.any(~np.isfinite(df)))
    return df


def request_ensemble_data(
    n_days: int = 5,
    sites: List[Station] = [VLA_SITE],
    variable: str = "total_column_integrated_water_vapour",
    model: str = "ecmwf_ifs025_ensemble",
    n_past_days: int = 1,
) -> pd.DataFrame:
    lats = [s.latitude.to("deg").value for s in sites]
    lons = [s.longitude.to("deg").wrap_at("180 deg").value for s in sites]
    names = [s.name for s in sites]
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": variable,
        "models": model,
        "forecast_days": n_days,
        "past_days": n_past_days,
    }
    openmeteo = openmeteo_requests.Client(session=RETRY_SESSION)
    responses = openmeteo.weather_api(ENSEMBLE_API_URL, params=params)
    df = parse_ensemble_response(responses, names, variable)
    logger.info(f"Weather: (N={df.shape[0]}, has_bad={df.attrs['has_bad']})")
    return df


class OpenMeteoQuery(QueryBase):
    delta = pd.Timedelta("12.5h")
    freq = "15min"
    query_type = "open-meteo"
    outname = "weather"

    def __init__(self, **kwargs):
        """
        Query the open-meteo API for GFS/HRRR forecast values at one or more
        site locations.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments are passed to the ``request_data`` function.
        """
        self._time = pd.Timestamp.now(tz="utc")
        try:
            self.df = request_data(**kwargs)
        except Exception:
            logger.exception(f"Error retrieving {self.query_type} forecast data.")
            self.df = None

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.df is not None

    def save_data(self, outname: Union[Path, str, None] = None) -> None:
        if outname is None:
            outname = self.outname
        if not self.okay:
            logger.warn(f"Could not save data for {self.query_type} query.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)


class OpenMeteoVlaQuery(OpenMeteoQuery):
    delta = pd.Timedelta("12.5h")
    freq = "15min"
    query_type = "open-meteo vla"
    outname = "weather"

    def __init__(self, **kwargs):
        """
        Query the open-meteo API for HRRR forecast values at the VLA site.
        This class is tailored to the parameters used for the VLA phase RMS
        model.  If open-meteo cannot be reached, then minimal data products are
        constructed using day of year and hour of day on a 15-minutely time
        cadence.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments are passed to the ``request_data`` function.
        """
        super().__init__(
            n_days=5,
            hourly_columns=COLUMNS_VLA_HR,
            minutely_columns=COLUMNS_VLA_15,
            **kwargs,
        )
        if self.okay:
            self.df.reset_index(level="site", inplace=True)

    def to_model_series(self) -> TimeSeries:
        if self.okay:
            df_subset = to_training_subset(self.df)
            return timeseries_from_dataframe(df_subset, freq=self.freq)
        else:
            start = self.forecast_time - self.delta
            end = self.forecast_time + self.delta
            dates = pd.date_range(
                start=start,
                end=end,
                freq=self.freq,
            )
            df = pd.DataFrame(
                {
                    "hour": dates.hour + dates.minute / 60,
                    "day_of_year": dates.dayofyear,
                },
                index=dates,
            )
            return timeseries_from_dataframe(df, freq=self.freq)


class OpenMeteoVlaPressureQuery(OpenMeteoQuery):
    delta = pd.Timedelta("12.5h")
    freq = "1h"
    query_type = "open-meteo pressure"
    outname = "weather_pr"
    pressure_columns = ["temperature", "relative_humidity"]
    hourly_columns = [
        f"{column}_{level}hPa"
        for column in pressure_columns
        for level in PRESSURE_LEVELS
        if level <= 825
    ]

    def __init__(self, **kwargs):
        """
        Query the open-meteo API for GFS/HRRR forecast values for pressure
        level quantities at the VLA site.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments are passed to the ``request_data`` function.
        """
        super().__init__(
            n_days=5,
            n_past_days=0,
            hourly_columns=self.hourly_columns,
            minutely_columns=None,
            **kwargs,
        )
        self.df = unpivot_pressure_levels(
            self.df,
            self.pressure_columns,
        )


class OpenMeteoVlaEnsembleQuery(OpenMeteoQuery):
    delta = pd.Timedelta("12.5h")
    freq = "1h"
    query_type = "open-meteo ensemble"
    outname = "weather_pwv_ensemble"

    def __init__(self, **kwargs):
        """
        Query the open-meteo ensemble API for ECMWF IFS 0.25° ensemble forecasts
        of total column integrated water vapour at the VLA site.

        Returns a DataFrame with a MultiIndex of (date, member) where member is
        the integer ensemble member index (0 = control, 1–50 = perturbed).

        Parameters
        ----------
        kwargs :
            Additional keyword arguments are passed to ``request_ensemble_data``.
        """
        self._time = pd.Timestamp.now(tz="utc")
        try:
            self.df = request_ensemble_data(
                n_days=5,
                n_past_days=0,
                variable="total_column_integrated_water_vapour",
                model="ecmwf_ifs025_ensemble",
                **kwargs,
            )
            self.df = self.df.reset_index(level="site", drop=True)
        except Exception:
            logger.exception(f"Error retrieving {self.query_type} forecast data.")
            self.df = None


class OpenMeteoMultiSiteQuery(OpenMeteoQuery):
    delta = pd.Timedelta("12.5h")
    freq = "1h"
    query_type = "open-meteo multi-site"
    outname = "weather_ms"

    def __init__(self, **kwargs):
        """
        Query the open-meteo API for GFS/HRRR forecast values at multiple
        site locations for the VLA, GBT, and VLBA stations.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments are passed to the ``request_data`` function.
        """
        super().__init__(
            n_days=5,
            n_past_days=0,
            sites=SITES,
            hourly_columns=COLUMNS_MULTI_HR,
            minutely_columns=None,
            **kwargs,
        )
