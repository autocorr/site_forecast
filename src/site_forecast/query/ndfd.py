
from io import StringIO
from typing import Union
from pathlib import Path

import requests

import pandas as pd
from pandas import (DataFrame, Timestamp)

from . import (QueryBase, to_parquet)
from .. import logger


URL = "http://mcmonitor.evla.nrao.edu/evla/ostweather/forecast.txt"


def query_weather(url=URL) -> DataFrame | None:
    """
    Query the NRAO's internal forecast generated from the XML REST API of the
    NOAA's National Digital Forecast Database (NDFD).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.split("\n")
        header = lines[0]
        body = lines[1:]
        df = pd.read_csv(StringIO("\n".join(body)), sep=r"\s+")
        df.rename(
                columns={
                        "HR": "hour",
                        "WndDir": "wind_direction",
                        "WndGust": "wind_gust",
                        "WndSpeed": "wind_speed",
                        "DewPt": "dewpoint_temperature",
                        "Temp": "temperature",
                },
                inplace=True,
        )
        date = pd.to_datetime(header).normalize()
        hours = pd.to_timedelta(df.hour, unit="hr")
        df.set_index(date+hours, inplace=True)
        df.drop(columns="hour", inplace=True)
        return df
    except:
        logger.warn("Could not retrieve NDFD weather forecast.")
        return None


class NdfdQuery(QueryBase):
    def __init__(self, time=None, url: str=URL):
        self._time = pd.Timestamp.now(tz="utc") if time is None else time
        try:
            self.df = query_weather(url=url)
        except:
            logger.exception("Error retrieving NDFD weather forecast.")
            self.df = None

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.df is not None

    def save_data(self, outname: Union[Path, str]="ndfd") -> None:
        if not self.okay:
            logger.warn(f"Could not save data for NDFD forecast.")
            return
        to_parquet(self.df, self.forecast_dir / outname)

