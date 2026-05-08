"""
Query HRRR forecast data from NOMADS using the filter CGI endpoints
and parse the GRIB2 responses with cfgrib.

This module is functionally equivalent to ``query.herbie_maps`` but
replaces the Herbie library with direct ``requests`` downloads from
NOMADS, making it suitable for environments where Herbie cannot be
installed or configured.

The NOMADS sub-hourly endpoint (``filter_hrrr_sub.pl``) is used for
TCOLW, TCOLI, and VIL (15-min resolution, 0-18 h horizon), mirroring
Herbie's ``product="subh"`` path.  The hourly surface endpoint
(``filter_hrrr_2d.pl``) is used for MCDC and TCDC, mirroring
``product="sfc"``.

Because cfgrib requires a real file path and does not support reading
from an in-memory file-like object, temporary GRIB2 files are written
to ``/dev/shm`` (a RAM-backed tmpfs on Linux) to avoid spindle disk I/O.
"""

import os
import time as time_module
import tempfile
import warnings
from numbers import Real
from pathlib import Path
from typing import Optional, Union

import cfgrib
import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from xarray import Dataset
from pandas import Timestamp
from astropy.coordinates import Latitude, Longitude

from . import (
    QueryBase,
    to_netcdf,
    normalize_time,
    wrap_coordinates,
)
from .. import CONFIG, SITE_LAT, SITE_LON, logger
from .herbie_maps import (
    subset_rectangular_region,
    extract_position,
    extract_mean,
    extract_quantiles,
    add_coverage,
    geodetic_to_number,
    get_var_names,
)


warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    message=".*xarray decode_timedelta will default to False.*",
)

# NOMADS CGI filter endpoints, keyed by HRRR product type.
NOMADS_FILTER = {
    "subh": "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_sub.pl",
    "sfc": "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl",
}
# File naming pattern for each product type.
NOMADS_FILE_PAT = {
    "subh": "hrrr.t{hour:02d}z.wrfsubhf{fxx:02d}.grib2",
    "sfc": "hrrr.t{hour:02d}z.wrfsfcf{fxx:02d}.grib2",
}
DEFAULT_FXX = list(range(13))  # forecast hours 0..12
HRRR_PROD_LAG_HOURS = 2  # typical HRRR upload lag behind real time


def _make_session(retries: int = 3, backoff_factor: float = 2.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _latest_hrrr_run() -> pd.Timestamp:
    """Estimate the most recently available HRRR model run."""
    now = pd.Timestamp.now(tz="utc")
    return (now - pd.Timedelta(hours=HRRR_PROD_LAG_HOURS)).floor("h").tz_convert(None)


def _build_url(
    date_str: str,
    run_hour: int,
    fxx: int,
    quantity: str,
    layer: str,
    product: str = "subh",
    region: Optional[tuple] = None,
) -> str:
    """Construct a NOMADS filter CGI URL for a single HRRR GRIB2 file."""
    filter_url = NOMADS_FILTER[product]
    filename = NOMADS_FILE_PAT[product].format(hour=run_hour, fxx=fxx)
    lev_key = "lev_" + layer.replace(" ", "_")
    params = [
        f"dir=/hrrr.{date_str}/conus",
        f"file={filename}",
        f"var_{quantity}=on",
        f"{lev_key}=on",
    ]
    if region is not None:
        llon, rlon, blat, tlat = region
        params += [
            "subregion=",
            f"leftlon={llon}",
            f"rightlon={rlon}",
            f"bottomlat={blat}",
            f"toplat={tlat}",
        ]
    return filter_url + "?" + "&".join(params)


def _fetch_grib_datasets(
    session: requests.Session,
    url: str,
    conn_timeout: float = 4,
    read_timeout: float = 60,
    shm_dir: str = "/dev/shm",
) -> list[Dataset]:
    """Download a GRIB2 response from NOMADS and parse with cfgrib.

    Writes to ``/dev/shm`` (RAM-backed tmpfs on Linux) to avoid spindle
    disk I/O.  cfgrib cannot read from a file-like object; it requires a
    real file path.
    """
    resp = session.get(url, timeout=(conn_timeout, read_timeout))
    resp.raise_for_status()
    tmpdir = shm_dir if Path(shm_dir).is_dir() else None
    with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".grib2", delete=False) as f:
        f.write(resp.content)
        tmppath = f.name
    try:
        return cfgrib.open_datasets(tmppath)
    finally:
        os.unlink(tmppath)


def _rename_unknown(ds: Dataset, target: str) -> Dataset:
    """Rename an ``"unknown"`` data variable to ``target`` if present."""
    if "unknown" in ds.data_vars and target not in ds.data_vars:
        ds = ds.rename_vars({"unknown": target})
    return ds


def get_quantity(
    quantity: str = "TCOLW",
    layer: str = "entire atmosphere",
    product: str = "subh",
    time: Optional[Timestamp] = None,
    fxx: Optional[list] = None,
    lat: Optional[Latitude] = None,
    lon: Optional[Longitude] = None,
    lat_margin: float = 0.5,
    lon_margin: float = 0.6,
    conn_timeout: float = 4,
    read_timeout: float = 60,
    retry_delay: float = 60,
    max_tries: int = 3,
    shm_dir: str = "/dev/shm",
) -> Dataset:
    """Fetch a single HRRR quantity from NOMADS across multiple forecast steps.

    Parameters
    ----------
    quantity : str
        GRIB2 variable abbreviation (e.g. ``"TCOLW"``, ``"TCOLI"``,
        ``"VIL"``, ``"MCDC"``, ``"TCDC"``).
    layer : str
        Level name matching the NOMADS ``lev_`` parameter with spaces
        instead of underscores (e.g. ``"entire atmosphere"``).
    product : str
        ``"subh"`` for the sub-hourly HRRR product (15-min resolution,
        0-18 h, ``filter_hrrr_sub.pl``); ``"sfc"`` for the hourly
        surface product (0-48 h, ``filter_hrrr_2d.pl``).
    time : pandas.Timestamp, optional
        Model run time.  Defaults to the latest estimated run.
    fxx : list of int, optional
        Forecast hours to retrieve.  Defaults to 0-12.
    lat, lon : Latitude, Longitude, optional
        Site coordinates used for server-side subregion filtering to
        reduce download size.  If ``None`` the full CONUS grid is
        downloaded.
    lat_margin, lon_margin : float
        Degrees of margin around the site for the bounding box.
    conn_timeout, read_timeout : float
        Connection and read timeout in seconds per request.
    retry_delay : float
        Seconds between retries.  NOAA requests a minimum of 60 s.
    max_tries : int
        Maximum download attempts per forecast step.
    shm_dir : str
        Directory for temporary GRIB2 files (default ``/dev/shm``).
    """
    if retry_delay < 60:
        raise ValueError(f"NOAA requests retry delay must be >=60 s: {retry_delay=}")
    if fxx is None:
        fxx = DEFAULT_FXX

    run_time = _latest_hrrr_run() if time is None else normalize_time(time)
    date_str = run_time.strftime("%Y%m%d")
    run_hour = run_time.hour

    region = None
    if lat is not None and lon is not None:
        lat_val, lon_val = geodetic_to_number(lat, lon)
        region = (
            round(lon_val - lon_margin, 2),  # leftlon
            round(lon_val + lon_margin, 2),  # rightlon
            round(lat_val - lat_margin, 2),  # bottomlat
            round(lat_val + lat_margin, 2),  # toplat
        )

    session = _make_session()
    datasets = []
    for fh in fxx:
        url = _build_url(date_str, run_hour, fh, quantity, layer, product, region)
        for attempt in range(max_tries):
            try:
                dsets = _fetch_grib_datasets(
                    session, url, conn_timeout, read_timeout, shm_dir
                )
                if dsets:
                    datasets.append(dsets[0])
                break
            except Exception:
                logger.warning(
                    "NOMADS fetch attempt %d/%d failed for %s fxx=%d.",
                    attempt + 1,
                    max_tries,
                    quantity,
                    fh,
                )
                if attempt < max_tries - 1:
                    time_module.sleep(retry_delay)
        else:
            logger.warning(
                "Skipping %s fxx=%d after %d failed attempts.", quantity, fh, max_tries
            )

    if not datasets:
        raise RuntimeError(
            f"No data retrieved for {quantity} (product={product!r}, run={run_time})"
        )

    ds = xr.concat(datasets, dim="step")
    ds = ds.assign_coords(date=ds.valid_time)
    ds.attrs["model"] = "hrrr"
    return ds


def get_tcolw(**kwargs) -> Dataset:
    quantity = "TCOLW"
    varname = "tcolw"
    ds = get_quantity(
        quantity=quantity, layer="entire atmosphere", product="subh", **kwargs
    )
    ds = _rename_unknown(ds, varname)
    ds[varname] = ds[varname].assign_attrs(
        {
            "GRIB_name": quantity,
            "GRIB_shortName": quantity,
            "GRIB_units": "kg/m^2",
            "long_name": "Total column-integrated cloud water",
            "units": "kg/m^2",
            "standard_name": quantity,
            "plot_log": 1,
            "plot_min": 1e-4,
            "plot_max": 1e1,
            "plot_split": 0.1,
        }
    )
    return ds


def get_tcoli(**kwargs) -> Dataset:
    quantity = "TCOLI"
    varname = "tcoli"
    ds = get_quantity(
        quantity=quantity, layer="entire atmosphere", product="subh", **kwargs
    )
    ds = _rename_unknown(ds, varname)
    ds[varname] = ds[varname].assign_attrs(
        {
            "GRIB_name": quantity,
            "GRIB_shortName": quantity,
            "GRIB_units": "kg/m^2",
            "long_name": "Total column-integrated cloud ice",
            "units": "kg/m^2",
            "standard_name": quantity,
            "plot_log": 1,
            "plot_min": 1e-4,
            "plot_max": 1e1,
            "plot_split": 0.01,
        }
    )
    return ds


def get_vil(**kwargs) -> Dataset:
    # HRRR GRIB2 VIL shortName is "veril" in the NCEP param table; cfgrib
    # may return the variable as "veril" directly, or as "unknown" if the
    # eccodes table is missing it.  Handle both cases.
    ds = get_quantity(
        quantity="VIL", layer="entire atmosphere", product="subh", **kwargs
    )
    ds = _rename_unknown(ds, "veril")
    ds["veril"] = ds["veril"].assign_attrs(
        {
            "GRIB_units": "kg/m^2",
            "GRIB_shortName": "VERIL",
            "units": "kg/m^2",
            "plot_log": 1,
            "plot_min": 1e-4,
            "plot_max": 1e1,
            "plot_split": 0.1,
        }
    )
    return ds


def get_mcdc(**kwargs) -> Dataset:
    ds = get_quantity(
        quantity="MCDC", layer="middle cloud layer", product="sfc", **kwargs
    )
    ds = _rename_unknown(ds, "mcc")
    ds["mcc"] = ds["mcc"].assign_attrs(
        {
            "plot_log": 0,
            "plot_min": 0,
            "plot_max": 100,
        }
    )
    return ds


def get_tcdc(**kwargs) -> Dataset:
    ds = get_quantity(
        quantity="TCDC", layer="entire atmosphere", product="sfc", **kwargs
    )
    ds = _rename_unknown(ds, "tcc")
    ds["tcc"] = ds["tcc"].assign_attrs(
        {
            "plot_log": 0,
            "plot_min": 0,
            "plot_max": 100,
        }
    )
    return ds


class NomadsQuery(QueryBase):
    def __init__(
        self,
        lat: Union[Latitude, Real] = SITE_LAT,
        lon: Union[Longitude, Real] = SITE_LON,
        time: Optional[Timestamp] = None,
        query_type: Optional[str] = "tcolw",
        **kwargs,
    ):
        """Query HRRR forecast data from NOMADS via the filter CGI endpoints.

        Functionally equivalent to
        :class:`~site_forecast.query.herbie_maps.HerbieQuery` but uses
        direct ``requests`` downloads instead of the Herbie library.

        Parameters
        ----------
        lat : number
            Site latitude.
        lon : number
            Site longitude.
        time : pandas.Timestamp, optional
            Forecast time.  If ``None``, the latest available HRRR run is
            used (estimated as ~2 hours behind the current UTC time).
        query_type : str, optional
            Convenience name for the HRRR variable to retrieve:

            - ``"tcolw"``  total column-integrated cloud liquid water (subh)
            - ``"tcoli"``  total column-integrated cloud ice (subh)
            - ``"tcc"``    total cloud coverage (sfc)
            - ``"mcc"``    medium cloud coverage (sfc)
            - ``"veril"``  vertically integrated liquid water (subh)
        """
        lat, lon = wrap_coordinates(lat, lon)
        self.lat = lat
        self.lon = lon
        self._query_type = query_type
        self._time = pd.Timestamp.now(tz="utc") if time is None else time
        try:
            match query_type:
                case "tcolw":
                    ds = get_tcolw(lat=lat, lon=lon, **kwargs)
                case "tcoli":
                    ds = get_tcoli(lat=lat, lon=lon, **kwargs)
                case "tcc":
                    ds = get_tcdc(lat=lat, lon=lon, **kwargs)
                case "mcc":
                    ds = get_mcdc(lat=lat, lon=lon, **kwargs)
                case "veril":
                    ds = get_vil(lat=lat, lon=lon, **kwargs)
                case _:
                    ds = get_quantity(lat=lat, lon=lon, **kwargs)
            dss = subset_rectangular_region(ds, lat=lat, lon=lon)
            p_ds = extract_position(dss, lat=lat, lon=lon)
            m_ds = extract_mean(dss, lat=lat, lon=lon)
            q_ds = extract_quantiles(dss, lat=lat, lon=lon)
            self.ds = (
                dss.merge(p_ds, compat="override")
                .merge(q_ds, compat="override")
                .merge(m_ds, compat="override")
            )
            add_coverage(self.ds)
        except:
            logger.exception("Error retrieving HRRR data via NOMADS.")
            self.ds = None

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.ds is not None

    @property
    def query_type(self) -> str:
        if self._query_type is not None:
            return self._query_type
        return list(self.ds.data_vars.keys())[0]

    @property
    def label_unit(self) -> str:
        s = self.ds.data_vars[self.query_type].attrs["GRIB_units"]
        return s.replace("%", r"\%")

    @property
    def label_name(self) -> str:
        return self.ds.data_vars[self.query_type].attrs["GRIB_shortName"].upper()

    @property
    def label(self) -> str:
        return rf"$\mathrm{{{self.label_name}}}\ [\mathrm{{{self.label_unit}}}]$"

    @property
    def data(self) -> Dataset:
        return self.ds[self.query_type]

    @property
    def n_steps(self) -> Real:
        return len(self.ds.step)

    @property
    def attrs(self) -> dict:
        return self.data.attrs

    @property
    def plot_min(self) -> Optional[Real]:
        return self.attrs.get("plot_min", None)

    @property
    def plot_max(self) -> Optional[Real]:
        return self.attrs.get("plot_max", None)

    @property
    def plot_split(self) -> Optional[Real]:
        return self.attrs.get("plot_split", None)

    @property
    def plot_log(self) -> Optional[bool]:
        return bool(self.attrs.get("plot_log", 0))

    @property
    def plot_norm_type(self) -> str:
        return "log" if self.plot_log else "linear"

    def save_data(self, outname: Optional[Union[Path, str]] = None) -> None:
        if not self.okay:
            logger.warn(f"Could not save data for: {self.query_type}")
            return
        if outname is None:
            model = self.ds.attrs.get("model", "hrrr")
            outname = f"{model}_{self.query_type}"
        outpath = self.forecast_dir / Path(outname)
        to_netcdf(self.ds, outpath)
