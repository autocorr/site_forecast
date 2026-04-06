
from pathlib import Path
from typing import (Optional, Union, List, Iterable)
from numbers import Real

import numpy as np
import pandas as pd
from darts import TimeSeries
from pandas import Timestamp
from astropy.time import Time
from astropy.coordinates import (Latitude, Longitude)

import requests_cache
import openmeteo_requests
from retry_requests import retry

from . import (
        QueryBase,
        timeseries_from_dataframe,
        to_parquet,
        wrap_coordinates,
)
from .. import (SITE_LAT, SITE_LON, SITES, SITES_BY_NAME, Station, logger)
from ..train import to_training_subset


CACHE_SESSION = requests_cache.CachedSession(".cache", expire_after=3600)
RETRY_SESSION = retry(CACHE_SESSION, retries=5, backoff_factor=0.2)
URL = "https://api.open-meteo.com/v1/forecast"
VLA_SITE = SITES_BY_NAME["Y1"]


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
        #"sensible_heat_flux",
        #"latent_heat_flux",
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
            label: wrapped.Variables(i).ValuesAsNumpy()
            for i, label in enumerate(columns)
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
        raise ValueError("At least one of either hourly or minutely columns must be specified.")
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
        df["hour"] = (df.index.hour + df.index.minute/60).astype(np.float32)
        df["day_of_year"] = df.index.dayofyear.astype(np.int32)
        all_df.append(df)
    m_df = pd.concat(all_df).reset_index().set_index(["date", "site"])
    m_df.attrs["has_bad"] = int(np.any(~np.isfinite(m_df)))
    return m_df


def request_data(
        n_days: int=5,
        sites: List[Station]=[VLA_SITE],
        hourly_columns: List[str]=COLUMNS_VLA_HR,
        minutely_columns: List[str]=COLUMNS_VLA_15,
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
            "past_days": 1,
    }
    openmeteo = openmeteo_requests.Client(session=RETRY_SESSION)
    responses = openmeteo.weather_api(URL, params=params)
    df = parse_response(
            responses,
            names,
            hourly_columns=hourly_columns,
            minutely_columns=minutely_columns,
    )
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
        except:
            logger.exception(f"Error retrieving {self.query_type} forecast data.")
            self.df = None

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.df is not None

    def save_data(self, outname: Union[Path, str, None]=None) -> None:
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
        Query the open-meteo API for HRRR forecast values at a site location.
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
                **kwargs
        )
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
                        "hour": dates.hour + dates.minute/60,
                        "day_of_year": dates.dayofyear,
                    },
                    index=dates,
            )
            return timeseries_from_dataframe(df, freq=self.freq)


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
                n_days=3,
                hourly_columns=COLUMNS_MULTI_HR,
                minutely_columns=None,
                **kwargs
        )

