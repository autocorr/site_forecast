
import io
import warnings
import multiprocessing
from numbers import Real
from pathlib import Path
from contextlib import redirect_stdout
from typing import (Optional, Union)
from numbers import Real

import herbie
import numpy as np
import pandas as pd
import xarray as xr
from xarray import Dataset
from pandas import (DataFrame, Timestamp)
from astropy.coordinates import (Latitude, Longitude)

from . import (
        QueryBase,
        to_parquet,
        to_netcdf,
        normalize_time,
        wrap_coordinates,
)
from .. import (CONFIG, SITE_LAT, SITE_LON, logger)


warnings.filterwarnings(action="ignore", category=FutureWarning,
        message=".*xarray decode_timedelta will default to False.*")


def geodetic_to_number(lat: Latitude, lon: Longitude) -> (Real, Real):
    return (
            lat.to("deg").value,
            lon.to("deg").wrap_at("360 deg").value,
    )


def get_var_names(ds):
    return list(
            q for q in ds.data_vars.keys()
            if not q.endswith(("_c", "_m", "_p", "_q"))
    )


def get_quantity(
            quantity="TCOLW",
            layer="entire atmosphere",
            model="hrrr",
            product="subh",
            time=None,
            fxx=None,
            save_dir=None,
            timeout=300,  # sec
    ) -> Dataset:
    if fxx is None:
        fxx = list(range(13))  # 0..12 hours
    if time is None:
        try:
            last_fxx = max(fxx)
        except TypeError:
            last_fxx = fxx
        # Forecasts are uploaded sequentially, so check the most recent valid
        # time of the last forecast needed.
        hl = herbie.HerbieLatest(
                model=model,
                product=product,
                fxx=last_fxx,
                overwrite=True,
                verbose=False,
                save_dir="/dev/shm",
        )
        time = hl.date
    else:
        time = normalize_time(time)
    if save_dir is None:
        save_dir = Path(CONFIG.get("Paths", "herbie", fallback="/dev/shm")).expanduser()
    # Use HerbieLatest to get the latest forecast run and then use FastHerbie
    # to download all the forecast steps for that run.
    h = herbie.FastHerbie(
            [time],
            model=model,
            product=product,
            fxx=fxx,
            overwrite=True,
            verbose=False,
            save_dir=save_dir,
    )
    if len(h.objects) == 0:
        raise RuntimeError(f"Empty query: ({quantity=}, {layer=}, {fxx=})")
    search = rf":{quantity}:{layer}:(?:anl|\d+ min fcst|\d+ hour fcst)"
    df = h.inventory(search)
    if df.shape[0] == 0:
        raise RuntimeError(f"Empty inventory: ({quantity=}, {layer=}, {fxx=})")
    with multiprocessing.Pool(processes=1) as pool:
        ds = pool.apply_async(h.xarray, (search,)).get(timeout=timeout)
    return ds


def get_tcolw(**kwargs) -> Dataset:
    quantity = "TCOLW"
    ds = get_quantity(
            quantity=quantity,
            layer="entire atmosphere",
            model="hrrr",
            product="subh",
            **kwargs
    )
    ds = ds.rename_vars({"unknown": quantity.lower()})
    ds['tcolw'] = ds.tcolw.assign_attrs({
        "GRIB_name": quantity,
        "GRIB_shortName": quantity,
        "GRIB_units": "kg/m^2",
        "long_name": "Total column-integrated cloud water",
        "units": "kg/m^2",
        "standard_name": quantity,
        "plot_log": 1,
        "plot_min": 1e-4,
        "plot_max": 1e+1,
        "plot_split": 0.1,
    })
    return ds


def get_tcoli(**kwargs) -> Dataset:
    quantity = "TCOLI"
    ds = get_quantity(
            quantity=quantity,
            layer="entire atmosphere",
            model="hrrr",
            product="subh",
            **kwargs
    )
    ds = ds.rename_vars({"unknown": quantity.lower()})
    ds['tcoli'] = ds.tcoli.assign_attrs({
        "GRIB_name": quantity,
        "GRIB_shortName": quantity,
        "GRIB_units": "kg/m^2",
        "long_name": "Total column-integrated cloud ice",
        "units": "kg/m^2",
        "standard_name": quantity,
        "plot_log": 1,
        "plot_min": 1e-4,
        "plot_max": 1e+1,
        "plot_split": 0.01,
    })
    return ds


def get_vil(**kwargs) -> Dataset:
    ds = get_quantity(
            quantity="VIL",
            layer="entire atmosphere",
            model="hrrr",
            product="subh",
            **kwargs
    )
    quantity = "VERIL"
    ds['veril'] = ds.veril.assign_attrs({
        "GRIB_units": "kg/m^2",
        "GRIB_shortName": quantity,
        "units": "kg/m^2",
        "plot_log": 1,
        "plot_min": 1e-4,
        "plot_max": 1e+1,
        "plot_split": 0.1,
    })
    return ds


def get_mcdc(**kwargs) -> Dataset:
    ds = get_quantity(
            quantity="MCDC",
            layer="middle cloud layer",
            model="hrrr",
            product="sfc",
            **kwargs
    )
    ds['mcc'] = ds.mcc.assign_attrs({
        "plot_log": 0,
        "plot_min": 0,
        "plot_max": 100,
    })
    return ds


def get_tcdc(**kwargs) -> Dataset:
    ds = get_quantity(
            quantity="TCDC",
            layer="entire atmosphere",
            model="hrrr",
            product="sfc",
            **kwargs
    )
    ds['tcc'] = ds.tcc.assign_attrs({
        "plot_log": 0,
        "plot_min": 0,
        "plot_max": 100,
    })
    return ds


def subset_rectangular_region(
            ds,
            lat: Latitude=SITE_LAT,
            lon: Longitude=SITE_LON,
            lat_size=0.450,
            lon_size=0.543,
    ) -> Dataset:
    lat, lon = geodetic_to_number(lat, lon)
    if lat_size <= 0 or lon_size <= 0:
        raise ValueError(f"Invalid sizes: {lat_size=}, {lon_size=}")
    # FIXME This selection method will break around the wrap-around point 0/360,
    # but the HRRR is limited to the range 225-300 deg so will be okay for now.
    mask = (
            (lat-lat_size/2 <= ds.latitude)  & (ds.latitude  <= lat+lat_size/2) &
            (lon-lon_size/2 <= ds.longitude) & (ds.longitude <= lon+lon_size/2)
    )
    return ds.where(mask, drop=True)


def pick_points(ds, points, method="nearest", k=None):
    with redirect_stdout(io.StringIO()) as f:
        return ds.herbie.pick_points(
                points,
                method=method,
                k=k,
                use_cached_tree=False,
                verbose=False,
        )


def extract_position(
            ds,
            lat: Latitude=SITE_LAT,
            lon: Longitude=SITE_LON,
    ) -> DataFrame:
    lat, lon = geodetic_to_number(lat, lon)
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        points = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    else:
        if (n_lat := len(lat)) != (n_lon := len(lon)):
            raise ValueError(f"Latitude and longitude of unequal length: {n_lat=}, {n_lon=}")
        points = pd.DataFrame({"latitude": lat, "longitude": lon})
    p_ds = pick_points(ds, points, method="weighted")
    # Select all time steps for point 0 and neighbor/k 0. Data points have
    # already been weighted and averaged and all k values are identical.
    df = p_ds.to_dataframe().loc[:,0,0].reset_index()
    df["date"] = df.step + df.time
    df.set_index("date", inplace=True)
    data_cols = list(ds.data_vars.keys())
    df = df[[*data_cols]].rename(columns={c: f"{c}_p" for c in data_cols})
    df.attrs["radius"] = 0.0
    return df


def extract_circular_region(
            ds,
            lat: Latitude=SITE_LAT,
            lon: Longitude=SITE_LON,
            radius=23.0,  # km
    ) -> DataFrame:
    lat, lon = geodetic_to_number(lat, lon)
    points = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    if radius <= 0:
        raise ValueError(f"Invalid radius: {radius=}")
    # Determine number of pixels in a circle with the given radius. Then
    # select this number of numbers.
    # FIXME Hard-coded to HRRR's 3 km spatial resolution.
    k = int(np.pi * (radius / 3)**2)
    k = 1 if k == 0 else k
    # Redirect IO from Herbie's print statement "Growing BallTree".
    # Should probably file a PR so that function respects the `verbose`
    # argument too.
    p_ds = pick_points(ds, points, method="nearest", k=k)
    # Drop the "point" part of the index and make the major index the
    # time-step rather than the nearest neighbor index, k.
    df = (
            p_ds.to_dataframe()
            .droplevel("point")
            .swaplevel()  # step|time & k
            .sort_index()
    )
    # Convert the step index to date in order to match results from
    # `extract_position`.
    df = df.reset_index()
    dates = df.step + df.time
    df.index = pd.MultiIndex.from_arrays([dates, df.k], names=["date", "k"])
    data_cols = list(ds.data_vars.keys())
    df = df[[*data_cols]]
    df.attrs["radius"] = radius
    return df


def extract_quantiles(
            ds,
            lat=SITE_LAT,
            lon=SITE_LON,
            radii=(10, 20),
            n_steps=21,
    ) -> DataFrame:
    quantiles = np.linspace(0, 1, n_steps)
    data_cols = get_var_names(ds)
    to_merge = []
    for radius in radii:
        df = (
                extract_circular_region(ds, lat=lat, lon=lon, radius=radius)
                .groupby(level="date")[data_cols]
                .quantile(quantiles)
                .rename_axis(index={None: "quantile"})
                .assign(radius=float(radius))
                .set_index("radius", append=True)
                .reorder_levels(["radius", "date", "quantile"])
                .rename(columns={c: f"{c}_q" for c in data_cols})
        )
        to_merge.append(df)
    return pd.concat(to_merge)


def extract_mean(
            ds,
            lat=SITE_LAT,
            lon=SITE_LON,
            radii=(10, 20),
    ) -> DataFrame:
    data_cols = get_var_names(ds)
    to_merge = []
    for radius in radii:
        df = (
                extract_circular_region(ds, lat=lat, lon=lon, radius=radius)
                .groupby(level="date")[data_cols]
                .mean()
                .assign(radius=float(radius))
                .set_index("radius", append=True)
                .reorder_levels(["radius", "date"])
                .rename(columns={c: f"{c}_m" for c in data_cols})
        )
        to_merge.append(df)
    return pd.concat(to_merge)


def add_coverage(ds, threshold=1e-4) -> None:
    for quantity in get_var_names(ds):
        coverages = []
        for radius in ds.radius:
            s_ds = ds.sel(radius=radius)
            coverage_at_radius = []
            for d in s_ds.date:
                ss_ds = s_ds.sel(date=d)
                for q, v in zip(ss_ds["quantile"], ss_ds[f"{quantity}_q"]):
                    if v >= threshold:
                        break
                coverage_at_radius.append(1-q.item())
            coverages.append(coverage_at_radius)
        da = xr.DataArray(
                coverages,
                coords={"radius": ds.radius, "date": ds.date},
                dims=["radius", "date"],
                name=f"{quantity}_c",
        )
        ds[f"{quantity}_c"] = da


class HerbieQuery(QueryBase):
    def __init__(
                self,
                lat: Union[Latitude, Real]=SITE_LAT,
                lon: Union[Longitude, Real]=SITE_LON,
                time: Optional[Timestamp]=None,
                query_type: Optional[str]="tcolw",
                **kwargs
        ):
        """
        Query numerical weather model image data using Herbie. The default query
        is sub-hourly total cloud liquid water data from the HRRR. To specify a
        different query, additional keyword arguments are passed to the
        `get_quantity` function.

        Parameters
        ----------
        lat : number
            Site latitude.
        lon : number
            Site longitude.
        time : pandas.Timestamp, None
            Forecast time. If left as `None`, then the latest valid forecast
            data are retrieved.
        query_type : str, None
            Convenience method to specify an HRRR query. Valid names include:
              - "tcolw" : total column-integrated cloud liquid water
              - "tcoli" : total column-integrated cloud ice water
              - "tcc"   : total cloud coverage
              - "mcc"   : medium cloud coverage
              - "veril" : vertically integrated water
            If `None` or an unknown value, then a query is performed using the
            additional keyword arguments passed to the constructor.
        """
        lat, lon = wrap_coordinates(lat, lon)
        self.lat = lat
        self.lon = lon
        self._query_type = query_type
        self._time = pd.Timestamp.now(tz="utc") if time is None else time
        try:
            match query_type:
                case "tcolw":
                    ds = get_tcolw(**kwargs)
                case "tcoli":
                    ds = get_tcoli(**kwargs)
                case "tcc":
                    ds = get_tcdc(**kwargs)
                case "mcc":
                    ds = get_mcdc(**kwargs)
                case "veril":
                    ds = get_vil(**kwargs)
                case _:
                    ds = get_quantity(**kwargs)
            dss = subset_rectangular_region(ds, lat=lat, lon=lon)
            p_df = extract_position(dss, lat=lat, lon=lon).to_xarray()
            m_df = extract_mean(dss, lat=lat, lon=lon).to_xarray()
            q_df = extract_quantiles(dss, lat=lat, lon=lon).to_xarray()
            self.ds  = dss.merge(p_df).merge(m_df).merge(q_df)
            add_coverage(self.ds)
        except:
            logger.exception("Error retrieving HRRR data.")
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
        else:
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

    def save_data(self, outname: Optional[Union[Path, str]]=None) -> None:
        if not self.okay:
            logger.warn(f"Could not save data for: {self.query_type}")
            return
        if outname is None:
            model = self.ds.attrs["model"]
            outname = f"{model}_{self.query_type}"
        outpath = self.forecast_dir / Path(outname)
        to_netcdf(self.ds, outpath)

