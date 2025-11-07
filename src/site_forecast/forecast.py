
import os
os.environ["DARTS_CONFIGURE_MATPLOTLIB"] = "0"

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import Timestamp
from schedule import Scheduler

from . import (
        CONFIG,
        KMHOUR_TO_MS,
        logger,
        _now_dir,
)
from .plotting import plot_all_weather
from .predict_phase import (ModelPhaseForecast, LongModelPhaseForecast)
from .query.herbie_maps import HerbieQuery
from .query.monitor import (ApiQuery, WeatherStationQuery)
from .query.ndfd import NdfdQuery
from .query.open_meteo import OpenMeteoQuery


OUT_COLS = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "wind_gusts_10m",
        "wind_speed_10m",
        "wind_direction_10m",
        "total_column_integrated_water_vapour",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "precipitation_probability",
        "precipitation",
        "rain",
        "showers",
        "snowfall",
        "phase_rms",
]


class Forecast:
    def __init__(self):
        self.forecast_time = pd.Timestamp.now(tz="utc")
        self.weather = OpenMeteoQuery()
        self.phase   = ApiQuery()
        self.station = WeatherStationQuery()
        self.ndfd    = NdfdQuery()
        self.predict = ModelPhaseForecast(self.weather, self.phase)
        self.predict_long = LongModelPhaseForecast(self.weather, self.phase)
        self.herbie_queries = {
                q: HerbieQuery(query_type=q)
                for q in ("veril", "tcolw", "mcc")
        }

    @property
    def now_dir(self) -> Path:
        date = self.forecast_time.date().isoformat().replace("-", "/")
        hour = f"{self.forecast_time.hour:02d}"
        return Path(date) / hour

    @property
    def forecast_root(self) -> Path:
        return Path(CONFIG.get("Paths", "forecasts", fallback="./forecasts")).expanduser()

    @property
    def forecast_dir(self) -> Path:
        return self.forecast_root / self.now_dir

    def save_data(self):
        queries = [
                self.weather,
                self.phase,
                self.station,
                self.ndfd,
                self.predict,
                self.predict_long,
        ]
        queries.extend(self.herbie_queries.values())
        for query in queries:
            query.save_data()

    def write_results(self, out_cols=None) -> None:
        if not self.weather.okay and not self.predict.okay:
            logger.warn("Skipping TSV out file.")
            return
        out_cols = OUT_COLS if out_cols is None else out_cols
        w_df = self.weather.df
        f_df = self.predict.df
        out_dir = self.forecast_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "forecast.tsv"
        df = pd.concat([f_df, w_df], axis="columns", join="inner")
        df.index.name = "utc_time"
        # FIXME Most columns may not exist if the open-meteo query wasn't successful.
        # So this file will have a variable number of columns depending on the success.
        # This should be reformatted so these other columns have NaNs when no valid
        # data is present.
        for col in out_cols:
            if col not in df.columns:
                df[col] = np.nan
        try:
            df["wind_speed_10m"] *= KMHOUR_TO_MS
            df["wind_gusts_10m"] *= KMHOUR_TO_MS
        except KeyError:
            pass
        with out_path.open("w") as f:
            f.write(f"# Generated at {self.forecast_time.isoformat()}\n")
            f.write(f"# Phase model: {self.predict.model.name}\n")
            df.to_csv(f, sep="\t")

    def get_previous_phase_forecasts(self, filen="predicted_phase", n_forecasts=12):
        steps = np.arange(-n_forecasts, 0)
        deltas = pd.to_timedelta(steps, unit="hr")
        timestamps = self.forecast_time + deltas
        all_df = []
        for ts in timestamps:
            date = ts.date().isoformat().replace("-", "/")
            hour = f"{ts.hour:02d}"
            path = self.forecast_root / date / hour / f"{filen}.parquet"
            try:
                df = pd.read_parquet(path)
                df.attrs["offset"] = ts
                all_df.append(df)
            except FileNotFoundError:
                logger.warn(f"Prior phase forecast not found: {date}/{hour}")
                all_df.append(None)
        return all_df, deltas

    def plot_all(self) -> None:
        plot_all_weather(self)

    def link_latest(self) -> None:
        plot_root = Path(CONFIG.get("Paths", "plots", fallback="./plots")).expanduser()
        fcst_root = self.forecast_root
        for root in (plot_root, fcst_root):
            now_dir = root / self.now_dir
            latest_dir = root / "latest"
            if latest_dir.is_symlink():
                latest_dir.unlink()
            if not latest_dir.exists():
                latest_dir.symlink_to(now_dir, target_is_directory=True)
            else:
                raise RuntimeError(f"Directory exists: {latest_dir}")


def generate() -> Forecast:
    fc = Forecast()
    fc.save_data()
    fc.write_results()
    fc.plot_all()
    fc.link_latest()
    return fc


class SafeScheduler(Scheduler):
    def __init__(self, reschedule_on_failure=True):
        """
        Parameters
        ----------
        reschedule_on_failure : bool
            If True, jobs will be rescheduled for their next run as if they had
            completed successfully. If False, they'll run on the next
            ``run_pending()`` tick.

        Adapted from Matt Lewis's implementation here:
            https://gist.github.com/mplewis/8483f1c24f2d6259aef6#file-test_safe_schedule-py-L15
        """
        self.reschedule_on_failure = reschedule_on_failure
        super().__init__()

    def _run_job(self, job):
        try:
            super()._run_job(job)
        except Exception:
            logger.exception("Unhandled exception raised during forecast.")
            job.last_run = datetime.now()
            job._schedule_next_run()


def loop() -> None:
    scheduler = SafeScheduler()
    scheduler.clear()
    scheduler.every().hour.at(":30").do(generate)
    while True:
        scheduler.run_pending()
        time.sleep(10)  # sec

