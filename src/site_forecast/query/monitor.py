
from pathlib import Path
from typing import (Optional, Union)
from numbers import Real

import numpy as np
import pandas as pd
from darts import TimeSeries
from pandas import Timestamp
from astropy.time import Time

import psycopg2

from . import (
        QueryBase,
        timeseries_from_dataframe,
        to_parquet,
)
from .. import (CONFIG, logger)


N_BASELINES = 6
SQL_QUERY = """
SELECT * FROM mcdata
WHERE
  hostname = %(host)s AND
  devicename = %(device)s AND
  monpointname = %(mon_point)s AND
  timestamp BETWEEN %(mjd_start)s AND
                    %(mjd_end)s
  ORDER BY timestamp
"""


def column_wise_mean(df, name):
    return df[[f"{name}{n}" for n in range(N_BASELINES)]].mean(axis=1)


def column_wise_median(df, name):
    return df[[f"{name}{n}" for n in range(N_BASELINES)]].median(axis=1)


def parse_rows(rows):
    items = []
    for row in rows:
        _, _, name, time, alert, value, _, _ = row
        if alert:
            value = np.nan
        items.append([float(time), name.lower(), float(value)])
    df = pd.DataFrame(
            items,
            columns=["mjd", "column", "value"],
    )
    return df


def rowset_to_dataframe(rowset):
    df = pd.concat(rowset)
    # Use `pivot_table` since there are occassional double entries for a given
    # time near the of a month (e.g., 2022-06). The default is to take the
    # mean of duplicate entries.
    df = (df
            .sort_values("mjd")
            .pivot_table(columns="column", index="mjd", values="value")
            .reset_index()
    )
    # Set the index as a UTC time from the recorded MJD
    datetimes = Time(df.mjd.values, format="mjd", scale="utc").datetime
    df.index = pd.DatetimeIndex(datetimes)
    df.index.name = "time"
    df.sort_index(inplace=True)
    return df


class MonitorConnection:
    def __init__(self,
                host=None,
                user=None,
                password=None,
                dbname=None,
                timeout_length=15,
        ):
        if timeout_length <= 0:
            raise ValueError(f"Timeout duration must be greater than zero: {timeout_length}")
        self.timeout_length = timeout_length
        def get(s):
            return CONFIG.get("Monitor", s)
        host = get("host") if host is None else host
        user = get("user") if user is None else user
        password = get("password") if password is None else password
        dbname = get("dbname") if dbname is None else dbname
        self.connection = psycopg2.connect(
                host=host,
                user=user,
                password=password,
                dbname=dbname,
                connect_timeout=self.timeout_length,
        )

    def __del__(self):
        self.connection.close()

    @property
    def timeout_length_ms(self):
        return int(self.timeout_length * 1_000)

    def fetch_rows(self, mon_host, device, mon_point, mjd_start, mjd_end):
        if mjd_start > mjd_end:
            raise ValueError(f"MJD start must come before end: {mjd_start=}, {mjd_end=}")
        params = {
                "host": mon_host,
                "device": device,
                "mon_point": mon_point,
                "mjd_start": mjd_start,
                "mjd_end": mjd_end,
        }
        with self.connection.cursor() as cursor:
            cursor.execute(f"SET statement_timeout = {self.timeout_length_ms}")
            cursor.execute(SQL_QUERY, params)
            rows = cursor.fetchall()
            return rows

    def query_phases(self, mjd_start, mjd_end, n_smooth=3):
        if n_smooth < 1:
            raise ValueError(f"Smoothing number must be greater than zero: {n_smooth}")
        mon_host = "evla-m360-1"
        device = "API"
        all_mon_points = [f"RMS_PHASE{n}" for n in range(N_BASELINES)]
        all_mon_points.append("RMS_PHASE_FOR_OST")
        baseline_dfs = []
        for mon_point in all_mon_points:
            rows = self.fetch_rows(mon_host, device, mon_point, mjd_start, mjd_end)
            baseline_dfs.append(parse_rows(rows))
        df = rowset_to_dataframe(baseline_dfs)
        # For some time periods the values become unsynchronized and are offset
        # between the values for the six baselines. Resample onto a uniform 10 min
        # grid using a mean. This does shift the values by +/- 5 min from their
        # reported time.
        df = (
                df
                .resample("10min")
                .median()
                .dropna(how="all")
                .sort_values(by="mjd")
        )
        df["phase_rms_avg"] = column_wise_mean(df, "rms_phase")
        df["phase_rms_med"] = column_wise_median(df, "rms_phase")
        df["phase_rms"] = df.phase_rms_med.rolling(n_smooth, center=True, min_periods=1).mean()
        df.attrs["has_bad"] = int(np.any(~np.isfinite(df.phase_rms)))
        df.attrs["rms_max"] = df.phase_rms_med.max()
        df.attrs["rms_min"] = df.phase_rms_med.min()
        logger.info(f"API: (N={df.shape[0]}, has_bad={df.attrs['has_bad']}, min={df.attrs['rms_min']:.3f}, max={df.attrs['rms_max']:.3f})")
        return df

    def query_weather(self, mjd_start, mjd_end, **kwargs):
        """
        See Butler et al. (2014) EVLA Memo #179 for details of the weather station:
          https://library.nrao.edu/public/memos/evla/EVLAM_179.pdf
        """
        items = [
                ("HMT337", "Temperature"),
                ("HMT337", "Dewpoint_Temperature"),
                ("HMT337", "Relative_Humidity"),
                ("WXT520", "Pressure"),
                ("WXT520", "Wind_Speed_Minimum"),
                ("WXT520", "Wind_Speed_Average"),
                ("WXT520", "Wind_Speed_Maximum"),
                ("WXT520", "Wind_Direction_Minimum"),
                ("WXT520", "Wind_Direction_Average"),
                ("WXT520", "Wind_Direction_Maximum"),
                (  "M352", "Pyranometer_2"),
        ]
        row_dfs = []
        for device, mon_point in items:
            rows = self.fetch_rows("evla-m352", device, mon_point, mjd_start, mjd_end)
            row_dfs.append(parse_rows(rows))
        df = rowset_to_dataframe(row_dfs)
        df = df.interpolate("time", limit_direction="both").resample("10min").mean()
        df.attrs["has_bad"] = int(df.isnull().any().any())
        logger.info(f"WS: (N={df.shape[0]}, has_bad={df.attrs['has_bad']})")
        return df


class MonitorPointDbQuery(QueryBase):
    lookback = 0.8  # ~19.2 hour
    min_lag = pd.Timedelta("1h")
    min_span = pd.Timedelta("12h")

    def __init__(self, mjd_start=None, mjd_end=None):
        if mjd_end is None:
            mjd_end = float(Time.now().mjd)
        if mjd_start is None:
            mjd_start = mjd_end - self.lookback
        if mjd_start > mjd_end:
            raise ValueError(f"MJD start must come before end: {mjd_start=}, {mjd_end=}")
        self.mjd_start = mjd_start
        self.mjd_end = mjd_end
        self._time = pd.Timestamp(Time(mjd_end, format="mjd").to_datetime(), tz="utc")

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.df is not None and not self.df.attrs["has_bad"]

    @property
    def is_recent(self) -> bool:
        try:
            t_last = self.df.index.tz_localize("utc").max()
            return abs(self.forecast_time - t_last) < self.min_lag
        except:
            return False

    @property
    def is_complete(self) -> bool:
        try:
            t_span = abs(self.df.index.min() - self.df.index.max())
            return t_span > self.min_span
        except:
            return False


class ApiQuery(MonitorPointDbQuery):
    def __init__(self, mjd_start=None, mjd_end=None, **kwargs):
        super().__init__(mjd_start=mjd_start, mjd_end=mjd_end)
        try:
            df = MonitorConnection().query_phases(
                    mjd_start=self.mjd_start,
                    mjd_end=self.mjd_end,
                    **kwargs
            )
            self.df = df
        except:
            logger.exception("Error retrieving API data.")
            self.df = None

    @property
    def okay_for_model(self) -> bool:
        return self.okay and self.is_recent and self.is_complete

    def to_model_series(self) -> Optional[TimeSeries]:
        if self.okay_for_model:
            return (
                    timeseries_from_dataframe(self.df[["phase_rms"]], freq="10min")
                    .resample("15min", method="interpolate")
            )
        else:
            return None

    def save_data(self, outname: Union[Path, str]="api") -> None:
        if self.df is None:
            logger.warn("Could not save data for API monitor point query.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)


class WeatherStationQuery(MonitorPointDbQuery):
    def __init__(self, mjd_start=None, mjd_end=None, **kwargs):
        super().__init__(mjd_start=mjd_start, mjd_end=mjd_end)
        try:
            self.df = MonitorConnection().query_weather(
                    mjd_start=self.mjd_start,
                    mjd_end=self.mjd_end,
                    **kwargs
            )
        except:
            logger.exception("Error retrieving site weather station data.")
            self.df = None

    def save_data(self, outname: Union[Path, str]="station") -> None:
        if not self.okay:
            logger.warn("Could not save data for site weather station query.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)

