
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, Optional
from numbers import Real

from pandas import (DataFrame, Timestamp)
from darts import TimeSeries
from xarray import Dataset

from astropy.coordinates import Latitude, Longitude

from .. import (CONFIG, _now_dir)


class QueryBase(ABC):
    @property
    @abstractmethod
    def forecast_time(self) -> Timestamp:
        raise NotImplementedError()

    @property
    @abstractmethod
    def okay(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def save_data(self, outname: Optional[Union[Path, str]]=None) -> None:
        raise NotImplementedError()

    @property
    def _now_dir(self) -> Path:
        path = _now_dir(self.forecast_time)
        return path

    @property
    def _forecast_root(self) -> Path:
        return Path(CONFIG.get("Paths", "forecasts", fallback="./forecasts")).expanduser()

    @property
    def forecast_dir(self) -> Path:
        return self._forecast_root / self._now_dir


def to_parquet(df: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_path = path.parent / f"{path.name}.parquet"
    df.to_parquet(out_path)


def to_netcdf(ds: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_path = path.parent / f"{path.name}.netcdf"
    ds.to_netcdf(out_path)


def timeseries_from_dataframe(df: DataFrame) -> TimeSeries:
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_convert(None)
    return TimeSeries.from_dataframe(df, fill_missing_dates=True)


def normalize_time(time: Timestamp) -> Timestamp:
    t = time.tz_convert("utc").tz_convert(None)
    return Timestamp(t.year, t.month, t.day, t.hour)


def wrap_coordinates(lat, lon) -> (Latitude, Longitude):
    if isinstance(lat, Real):
        lat = Latitude(lat, unit="deg")
    if isinstance(lon, Real):
        lon = Longitude(lon, unit="deg")
    return Latitude(lat), Longitude(lon)


