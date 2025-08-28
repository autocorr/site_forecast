
from pathlib import Path
from typing import (Optional, Union)
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
from .. import (SITE_LAT, SITE_LON, logger)
from ..train import to_training_subset


CACHE_SESSION = requests_cache.CachedSession(".cache", expire_after=3600)
RETRY_SESSION = retry(CACHE_SESSION, retries=5, backoff_factor=0.2)
URL = "https://api.open-meteo.com/v1/forecast"


COLUMNS_HR = [
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

COLUMNS_15 = [
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


def date_range_from_response_interval(interval):
    return pd.date_range(
            start=pd.to_datetime(interval.Time(), unit="s", utc=True),
            end=pd.to_datetime(interval.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=interval.Interval()),
            inclusive="left",
    )


def parse_response(response):
    # Hourly variables
    hourly = response.Hourly()
    h_data = {
            label: hourly.Variables(i).ValuesAsNumpy()
            for i, label in enumerate(COLUMNS_HR)
    }
    h_data.update({"date": date_range_from_response_interval(hourly)})
    h_df = pd.DataFrame(h_data).set_index("date")
    h_df = h_df.resample("15min").asfreq().interpolate()
    # interpolate onto 15 minute grid
    # 15-minutely variables
    minutely = response.Minutely15()
    m_data = {
            label: minutely.Variables(i).ValuesAsNumpy()
            for i, label in enumerate(COLUMNS_15)
    }
    m_data.update({"date": date_range_from_response_interval(minutely)})
    m_df = pd.DataFrame(m_data).set_index("date")
    df = m_df.merge(h_df, left_index=True, right_index=True)
    # Add time quantities
    time = Time(df.index)
    df["mjd"] = time.mjd
    df["hour"] = (df.index.hour + df.index.minute/60).astype(np.float32)
    df["day_of_year"] = df.index.dayofyear.astype(np.int32)
    # Site attributes
    df.attrs["latitude"] = response.Latitude()
    df.attrs["longitude"] = response.Longitude()
    df.attrs["timezone"] = response.Timezone()
    df.attrs["timezone_abbrev"] = response.TimezoneAbbreviation()
    df.attrs["utc_offset"] = response.UtcOffsetSeconds()
    return df


def request_data(
        n_days=2,
        lat: Union[Latitude, Real]=SITE_LAT,
        lon: Union[Longitude, Real]=SITE_LON,
    ):
    lat, lon = wrap_coordinates(lat, lon)
    if n_days > 16:
        raise ValueError("Forecast days must be 16 days or fewer.")
    params = {
            "latitude": lat.to("deg").value,
            "longitude": lon.to("deg").wrap_at("180 deg").value,
            "hourly": COLUMNS_HR,
            "minutely_15": COLUMNS_15,
            "models": "gfs_seamless",
            "forecast_days": n_days,
            "past_days": 1,
    }
    openmeteo = openmeteo_requests.Client(session=RETRY_SESSION)
    response = openmeteo.weather_api(URL, params=params)
    df = parse_response(response[0])
    df = df.reset_index().set_index("date")
    df.attrs["has_bad"] = int(np.any(~np.isfinite(df)))
    logger.info(f"Weather: (N={df.shape[0]}, has_bad={df.attrs['has_bad']})")
    return df


class OpenMeteoQuery(QueryBase):
    delta = pd.Timedelta("12.5h")
    freq = "15min"

    def __init__(self, **kwargs):
        """
        Query the open-meteo API for HRRR forecast values at a site location.
        If open-meteo cannot be reached, then minimal data products are
        constructed using day of year and hour of day on a 15-minutely time
        cadence.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments are passed to the ``request_data`` function.
        """
        self._time = pd.Timestamp.now(tz="utc")
        try:
            self.df = request_data(**kwargs)
        except:
            logger.exception("Error retrieving open-meteo forecast data.")
            self.df = None

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.df is not None

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

    def save_data(self, outname: Union[Path, str]="weather") -> None:
        if not self.okay:
            logger.warn("Could not save data for open-meteo query.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)

