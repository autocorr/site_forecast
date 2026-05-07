from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .query.open_meteo import OpenMeteoVlaQuery, OpenMeteoVlaPressureQuery
    from .query.herbie_maps import HerbieQuery

import multiprocessing
from numbers import Real
from pathlib import Path

import amwrap
import numpy as np
import pandas as pd
from astropy import units as u

from . import logger
from .query import QueryBase, to_parquet

FREQ_MIN = 1.0 * u.GHz
FREQ_MAX = 50.0 * u.GHz
FREQ_STEP = 0.5 * u.GHz
OUTPUT_COLUMNS = [
    "frequency",
    "brightness temperature",
    "opacity",
    "transmittance",
]

# TODO
# - Predict range of sensitivities from the ensemble IFS PWV forecast.


class AmModelPredictor:
    """
    AM atmospheric radiative transfer model predictor for a single timestep.

    Wraps ``amwrap.Model`` with atmospheric profile preparation: pressure-level
    clipping to the site surface elevation, mixing ratio computation,
    stratospheric water vapor clamping, and optional PWV scaling.
    """

    # Stratospheric water vapor mixing ratio.
    strato_mixing_ratio = 5e-6 * u.dimensionless_unscaled
    strato_pressure_limit = 30 * u.hPa
    mixing_ratio_epsilon = 1e-7 * u.dimensionless_unscaled
    relative_humidity_epsilon = 0.1 / 3 * u.dimensionless_unscaled
    output_columns = OUTPUT_COLUMNS

    @u.quantity_input
    def __init__(
        self,
        pressure: u.Quantity["pressure"],  # noqa: F821
        temperature: u.deg_C | u.Quantity["temperature"],  # noqa: F821
        relative_humidity: u.Quantity["dimensionless"],  # noqa: F821
        surface_pressure: u.Quantity["pressure"] | None = None,  # noqa: F821
        pwv: u.Quantity["length"] | None = None,  # noqa: F821
        freq_min: u.Quantity["frequency"] = 1.0 * u.GHz,  # noqa: F821
        freq_max: u.Quantity["frequency"] = 50.0 * u.GHz,  # noqa: F821
        freq_step: u.Quantity["frequency"] = 0.2 * u.GHz,  # noqa: F821
        time: pd.Timestamp | None = None,
    ):
        """
        Parameters
        ----------
        pressure : u.Quantity
            Pressure levels in hPa, ascending or descending order.
        temperature : u.Quantity
            Temperature at each pressure level, in degrees Celsius or Kelvin.
        relative_humidity : u.Quantity
            Dimensionless relative humidity at each pressure level (0 to 1).
        surface_pressure : u.Quantity, optional
            Current surface pressure used to clip the profile to the site
            elevation.
        pwv : u.Quantity, optional
            Target precipitable water vapor in mm. If provided, the
            tropospheric humidity profile is rescaled to match this value.
        freq_min : u.Quantity, optional
            Minimum output frequency for the radiative transfer calculation.
        freq_max : u.Quantity, optional
            Maximum output frequency for the radiative transfer calculation.
        freq_step : u.Quantity, optional
            Frequency resolution of the radiative transfer calculation.
        time : pd.Timestamp, optional
            Forecast valid time, attached to the output DataFrame index.
        """
        self.pwv = pwv
        self.surface_pressure = surface_pressure
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step
        self.time = time if time is not None else pd.NaT
        # Convert values to descending order in order to be used by `interp_by_pressure`
        if is_ascending := pressure[1] > pressure[0]:
            pressure = pressure[::-1]
            temperature = temperature[::-1]
            relative_humidity = relative_humidity[::-1]
        # Clip the full pressure level arrays to the current surface level.
        if self.surface_pressure is not None:
            self.pres = amwrap.interp_by_pressure(pressure, pressure, surface_pressure)
            self.temp = amwrap.interp_by_pressure(
                temperature, pressure, surface_pressure
            )
            self.relh = amwrap.interp_by_pressure(
                relative_humidity, pressure, surface_pressure
            )
        else:
            self.pres = pressure
            self.temp = temperature
            self.relh = relative_humidity
        self.mixr = amwrap.mixing_ratio_from_relative_humidity(
            self.pres, self.temp, self.relh
        )
        # Clamp stratospheric water values and small values.
        strat_mask = self.pres < self.strato_pressure_limit
        self.mixr[strat_mask] = np.maximum(
            self.strato_mixing_ratio, self.mixr[strat_mask]
        )
        self.mixr = np.maximum(self.mixing_ratio_epsilon, self.mixr)
        self.relh = np.maximum(self.relative_humidity_epsilon, self.relh)
        # Get the PWV from the vertical profiles for reference scaling.
        self.model_pwv = (
            amwrap.precipitable_water(self.pres, self.temp, self.relh).to("mm").m * u.mm
        )

    @classmethod
    def from_frames(
        cls,
        surf_row: pd.Series,
        pres_df: pd.DataFrame,
        **kwargs,
    ):
        """
        Construct from a surface weather Series and pressure-level DataFrame.

        Parameters
        ----------
        surf_row : pd.Series
            One row from an ``OpenMeteoVlaQuery`` surface DataFrame. Must
            contain ``total_column_integrated_water_vapour`` (mm) and
            ``surface_pressure`` (hPa). The Series name is used as the
            forecast timestamp.
        pres_df : pd.DataFrame
            Single-timestamp slice of an ``OpenMeteoVlaPressureQuery``
            DataFrame, indexed by pressure level (hPa). Must contain
            ``temperature`` (°C) and ``relative_humidity`` (0–100) columns.
        **kwargs
            Additional keyword arguments forwarded to the constructor.

        Returns
        -------
        AmModelPredictor
        """
        time = surf_row.name
        pwv = surf_row["total_column_integrated_water_vapour"] * u.mm
        surf_pres = surf_row["surface_pressure"] * u.hPa
        pressure = pres_df.index.get_level_values("pressure").values * u.hPa
        temperature = pres_df["temperature"].values * u.deg_C
        relative_humidity = (
            pres_df["relative_humidity"].values / 100 * u.dimensionless_unscaled
        )
        return cls(
            pressure,
            temperature,
            relative_humidity,
            surface_pressure=surf_pres,
            pwv=pwv,
            time=time,
            **kwargs,
        )

    @property
    def pwv_scale(self) -> Real:
        if self.pwv is None:
            return 1.0
        else:
            return (self.pwv / self.model_pwv).to("").value

    @u.quantity_input
    def run(
        self,
        pwv: u.Quantity["length"] | None = None,
        water_cloud: u.Quantity["surface_mass_density"] | None = None,
    ) -> pd.DataFrame:
        """
        Run the AM radiative transfer model and return spectral results.

        Parameters
        ----------
        pwv : u.Quantity, optional
            Override the PWV scaling for this run. If not provided, the
            instance's own PWV (set at construction) is used.
        water_cloud : u.Quantity, optional
            Per-layer cloud liquid water mass surface density (kg/m²). If
            not provided, no cloud liquid water is included.

        Returns
        -------
        pd.DataFrame
            Indexed by (date, frequency) with columns for brightness
            temperature, opacity, and transmittance.
        """
        if pwv is None:
            pwv_scale = self.pwv_scale
        else:
            pwv_scale = (pwv / self.model_pwv).to("").value
        model = amwrap.Model(
            self.pres,
            self.temp,
            mixing_ratio={"h2o": self.mixr},
            troposphere_h2o_scaling=pwv_scale,
            water_cloud=water_cloud,
            output_columns=self.output_columns,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_step=self.freq_step,
        )
        df = model.run()
        df["date"] = self.time
        df.set_index(["date", "frequency"], inplace=True)
        return df


class VlaSeasonalPredictor:
    """
    VLA seasonal baseline AM model for a given forecast date.

    Uses the day of year to select either the ``midlatitude_summer`` or
    ``midlatitude_winter`` AM standard atmosphere, then rescales PWV and
    surface pressure to seasonal medians derived from ERA5 re-analysis.
    """

    # Degree 20 polynomial fit coefficients to be used with `np.polyval`
    pwv_coeffs = [
        -5.462272857095006786e-43,
        1.656934508447681923e-39,
        -2.181235084452699394e-36,
        1.588452325103683404e-33,
        -6.450273847618128915e-31,
        9.031894855222803599e-29,
        5.139707673857070333e-26,
        -3.689468461067430054e-23,
        1.207770464961245261e-20,
        -2.474051429391079885e-18,
        3.325611400943010894e-16,
        -2.825185426230397888e-14,
        1.269011345482608785e-12,
        -1.716319222413035708e-12,
        -2.585565712470558030e-09,
        8.142462701345654749e-08,
        1.093890931400897358e-06,
        -6.017320608467316564e-05,
        -4.623628684919113113e-05,
        -5.705901665060668197e-03,
        4.305624353386962255e00,
    ]
    psurf_coeffs = [
        -6.142527915809436772e-45,
        1.288936993375749526e-41,
        -1.013859911478358494e-38,
        3.307747694647683131e-36,
        -3.770954139685559258e-34,
        3.070770886430273572e-31,
        -2.476054271619191759e-28,
        -2.304022541818734443e-26,
        1.062074705640575899e-22,
        -5.999988068960015943e-20,
        1.795333234696000000e-17,
        -3.290420535220588221e-15,
        3.742788146903547919e-13,
        -2.455562318917477836e-11,
        6.847489026013505731e-10,
        1.016085834773454369e-08,
        -8.250418890039004326e-07,
        7.175267419138607005e-07,
        5.242742096369586461e-04,
        -2.422924940711348313e-02,
        7.889200166542797206e02,
    ]

    @u.quantity_input
    def __init__(
        self,
        date: pd.Timestamp,
        freq_min: u.Quantity["frequency"] = FREQ_MIN,  # noqa: F821
        freq_max: u.Quantity["frequency"] = FREQ_MAX,  # noqa: F821
        freq_step: u.Quantity["frequency"] = FREQ_STEP,  # noqa: F821
    ):
        """
        Parameters
        ----------
        date : pd.Timestamp
            Date for which to compute the seasonal baseline.
        freq_min : u.Quantity, optional
            Minimum output frequency.
        freq_max : u.Quantity, optional
            Maximum output frequency.
        freq_step : u.Quantity, optional
            Frequency resolution of the output.
        """
        self.date = date
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step

    @property
    def day_of_year(self) -> Real:
        return self.date.day_of_year

    @property
    def _climatology_name(self) -> str:
        """
        Return the US standard climatology for the VLA site (midlatitude).
        Use summer values for approximately June through October. Derived based
        on times when the average PWV is less than ~7 mm from ERA5 re-analysis.
        """
        if 160 <= self.day_of_year <= 280:
            return "midlatitude_summer"
        else:
            return "midlatitude_winter"

    @property
    def _seasonal_surface_pressure(self) -> u.Quantity["pressure"]:
        """Seasonal surface pressure at the VLA site from ERA5 re-analysis."""
        return float(np.polyval(self.psurf_coeffs, self.day_of_year)) * u.hPa

    @property
    def _seasonal_pwv(self) -> u.Quantity["length"]:
        """Seasonal PWV at the VLA stie from ERA5 re-analysis."""
        return float(np.polyval(self.pwv_coeffs, self.day_of_year)) * u.mm

    def run(self, **kwargs) -> pd.DataFrame:
        """
        Run the seasonal AM model and return spectral results.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            ``amwrap.Model.from_climatology``.

        Returns
        -------
        pd.DataFrame
            Indexed by frequency (GHz) with columns for brightness
            temperature, opacity, and transmittance.
        """
        cl = amwrap.Climatology(
            self._climatology_name,
            pressure_base=self._seasonal_surface_pressure,
        )
        model = amwrap.Model.from_climatology(
            cl,
            output_columns=OUTPUT_COLUMNS,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_step=self.freq_step,
            **kwargs,
        )
        model.troposphere_h2o_scaling = (self._seasonal_pwv / cl.pwv).to("")
        return model.run().set_index("frequency")


def _band_averages(freq_df: pd.DataFrame, band_frequencies: dict) -> pd.DataFrame:
    """Average columns of a frequency-indexed DataFrame over each named band."""
    freq_axis = freq_df.index
    rows = []
    for band, (freq_lo, freq_hi) in band_frequencies.items():
        mask = (freq_axis >= freq_lo) & (freq_axis < freq_hi)
        row = freq_df.loc[mask].mean().to_dict()
        row["band"] = band
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")


@u.quantity_input
def _make_water_cloud_array(
    pressure_levels: u.Quantity["pressure"],
    tcolw: u.Quantity["surface_mass_density"],
    target_pressure_level: u.Quantity["pressure"] = 600.0 * u.hPa,
) -> u.Quantity:
    """
    Return a mass surface density array with tcolw (kg/m²) at the layer nearest
    to `target_pressure_level`.
    """
    water_cloud = np.zeros(len(pressure_levels)) * (u.kg / u.m**2)
    idx = int(np.argmin(np.abs(pressure_levels - target_pressure_level)))
    water_cloud[idx] = tcolw
    return water_cloud


def _interp_pres_to_times(
    pres_df: pd.DataFrame,
    new_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Interpolate a (date, pressure)-MultiIndex DataFrame to new_times by time."""
    unstacked = pres_df.unstack(level="pressure")
    return (
        unstacked.reindex(unstacked.index.union(new_times))
        .interpolate(method="time")
        .reindex(new_times)
        .stack(level="pressure", future_stack=True)
    )


def _run_am_model(
    time: pd.Timestamp,
    surf_row: pd.Series,
    pres_slice: pd.DataFrame,
    freq_min: u.Quantity,
    freq_max: u.Quantity,
    freq_step: u.Quantity,
    band_frequencies: dict,
    tcolw_value=None,
    tcolw_quantile: float | None = None,
    cloud_target_pressure=None,
) -> tuple[pd.Timestamp, float | None, pd.DataFrame | None]:
    """
    Run AM for a single timestep; return (time, quantile, band-indexed
    DataFrame) or (time, quantile, None). Pass tcolw and cloud_target_pressure
    to include cloud liquid water. Module-level for multiprocessing pickling.
    """
    try:
        # Initialize the model predictor for the given vertical atmospheric
        # profiles.
        predictor = AmModelPredictor.from_frames(
            surf_row,
            pres_slice,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_step=freq_step,
        )
        # Create array for cloud liquid column density if present.
        if tcolw_value is not None:
            water_cloud = _make_water_cloud_array(
                predictor.pres,
                tcolw_value,
                target_pressure_level=cloud_target_pressure,
            )
        else:
            water_cloud = None
        df = predictor.run(water_cloud=water_cloud).droplevel("date")
        return time, tcolw_quantile, _band_averages(df, band_frequencies)
    except:
        msg = f"Error in calculating sensitivity at {time}"
        if tcolw_quantile is not None:
            msg += f", q={tcolw_quantile:.2f}"
        logger.warning(msg)
        return time, tcolw_quantile, None


class VlaSensitivityEstimator(QueryBase):
    """
    Estimates VLA observing sensitivity from numerical weather forecast data.

    Runs the AM atmospheric radiative transfer model for each forecast timestep
    using Open-Meteo surface and pressure-level data. Optionally incorporates
    15-minute HRRR total column cloud liquid water from a ``HerbieQuery``.
    Produces two output DataFrames: ``clear_df`` (hourly, full forecast range)
    and ``cloud_df`` (15-min, ≤12 h, one row per TCOLW spatial quantile), each
    with effective sensitivity and effective integration time columns referenced
    to a seasonal climatological baseline.
    """

    bands = list("LSCXUKAQ")
    # Nominal lower and higher frequencies for the VLA bands in GHz.
    band_frequencies = {  # GHz
        "L": (1.0, 2.0),
        "S": (2.0, 4.0),
        "C": (4.0, 8.0),
        "X": (8.0, 12.0),
        "U": (12.0, 18.0),
        "K": (18.0, 26.5),
        "A": (26.5, 40.0),
        "Q": (40.0, 50.0),
    }
    # The receiver temperatures in K are taken from the files provided by Rich
    # Moeser in:
    #   https://mctest.evla.nrao.edu/rich/tcals/
    # by taking the median across the band for each receiver and then taking
    # the median across all antennas and polarizations.
    receiver_temperatures = {
        "L": 18.5,
        "S": 20.9,
        "C": 7.9,
        "X": 12.1,
        "U": 12.0,
        "K": 14.6,
        "A": 25.1,
        "Q": 36.8,
    }
    freq_min = 1.0 * u.GHz
    freq_max = 50.0 * u.GHz
    freq_step = 0.5 * u.GHz
    cloud_target_pressure = 600.0 * u.hPa  # layer where all TCOLW is concentrated
    cloud_radius = 20.0 * u.km  # km; spatial radius used to select TCOLW quantiles

    def __init__(
        self,
        om_query_surf: "OpenMeteoVlaQuery",
        om_query_pres: "OpenMeteoVlaPressureQuery",
        hq_query_tcolw: Optional["HerbieQuery"],
        n_workers: int = 2,
    ):
        """
        Parameters
        ----------
        om_query_surf : OpenMeteoVlaQuery
            Surface weather forecast query providing PWV and surface pressure.
        om_query_pres : OpenMeteoVlaPressureQuery
            Pressure-level forecast query providing temperature and relative
            humidity profiles.
        hq_query_tcolw : HerbieQuery or None
            15-minute HRRR cloud liquid water query. If ``None`` or not okay,
            only the clear-sky hourly forecast is produced.
        n_workers : int, optional
            Number of parallel worker processes for AM model runs.

        Raises
        ------
        ValueError
            If ``n_workers`` is less than 1.
        """
        # FIXME This class should really be refactored into two that handle the
        # "long and clear" and "short and cloudy" cases, rather than doing both
        # together.
        if n_workers < 1:
            raise ValueError(f"Invalid number of workers: {n_workers=}")
        self.surf_query = om_query_surf
        self.pres_query = om_query_pres
        self.clwp_query = hq_query_tcolw
        self.n_workers = n_workers
        self._time = om_query_surf.forecast_time
        self.clear_df = None
        self.cloud_df = None
        try:
            self.clear_df, self.cloud_df = self.compute()
        except:
            logger.exception("Error computing sensitivity estimates.")
            self.df = None
            self.cloud_df = None

    @property
    def has_cloud(self) -> bool:
        return self.clwp_query is not None and self.clwp_query.okay

    def _compute_seasonal_baseline(self) -> pd.DataFrame:
        """Run the seasonal AM model; return a (band,)-indexed baseline DataFrame."""
        clim_df = VlaSeasonalPredictor(
            self.forecast_time,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_step=self.freq_step,
        ).run()
        return _band_averages(clim_df, self.band_frequencies)

    def _add_derived_columns(
        self,
        df: pd.DataFrame,
        baseline_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add baseline-relative sensitivity columns; works for any index that includes 'band'."""
        df = df.copy()
        band_level = df.index.get_level_values("band")
        for col in baseline_df.columns:
            df[f"{col}_cl"] = band_level.map(baseline_df[col])
        # NOTE: `amwrap` inserts an underscore into column names in the DataFrame.
        df["system_temperature"] = df["brightness_temperature"] + band_level.map(
            self.receiver_temperatures
        )
        df["system_temperature_cl"] = df["brightness_temperature_cl"] + band_level.map(
            self.receiver_temperatures
        )
        df["eff_sensitivity"] = (df.transmittance_cl / df.transmittance) * (
            df.system_temperature / df.system_temperature_cl
        )
        df["eff_time"] = df.eff_sensitivity**2
        return df

    def _get_tcolw_df(self) -> pd.DataFrame:
        """Extract TCOLW spatial quantiles from HerbieQuery as a (date x quantile) DataFrame."""
        ds = self.clwp_query.ds
        tcolw_da = ds["tcolw_q"].sel(radius=self.cloud_radius, method="nearest")
        df = tcolw_da.to_pandas().T
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index.name = "date"
        return df

    def _run_series(
        self,
        surf_df: pd.DataFrame,
        pres_df: pd.DataFrame,
        tcolw_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run AM for a time series; return a (date, band)- or (date, band,
        quantile)-indexed DataFrame.

        When tcolw_df is None, runs at the hourly cadence of pres_df without
        cloud water, returning a (date, band) MultiIndex. When tcolw_df is
        provided, runs at the 15-min cadence of tcolw_df with one run per
        quantile, returning a (date, band, quantile) MultiIndex.
        """
        # FIXME This should really be better factored to avoid duplication
        # with constructing the arguments, although it is tricky because the
        # time series index for the two types of queries have different sampling.

        # Generate time series axis and arguments based on whether to use cloud
        # values or not before passing them to the runner function and
        # multiprocessing.
        if tcolw_df is not None:
            # `_interp_pres_to_times` will result in `pres_resolved` having
            # the same index as the `tcolw_df`, so a third intersection is
            # not required.
            pres_resolved = _interp_pres_to_times(pres_df, tcolw_df.index)
            times = tcolw_df.index.intersection(surf_df.index).intersection(
                pres_resolved.index.get_level_values("date").unique()
            )
            quantiles = tcolw_df.columns
            args = [
                (
                    t,
                    surf_df.loc[t],
                    pres_resolved.xs(t, level="date"),
                    self.freq_min,
                    self.freq_max,
                    self.freq_step,
                    self.band_frequencies,
                    tcolw_df.loc[t, q] * u.kg / u.m**2,
                    float(q),
                    self.cloud_target_pressure,
                )
                for t in times
                for q in quantiles
            ]
        else:
            times = pres_df.index.get_level_values("date").intersection(surf_df.index)
            pres_resolved = pres_df
            args = [
                (
                    t,
                    surf_df.loc[t],
                    pres_resolved.xs(t, level="date"),
                    self.freq_min,
                    self.freq_max,
                    self.freq_step,
                    self.band_frequencies,
                )
                for t in times
            ]

        # Run with either single or parallel execution.
        if self.n_workers == 1:
            results = [_run_am_model(*a) for a in args]
        else:
            with multiprocessing.Pool(self.n_workers) as pool:
                results = pool.starmap(_run_am_model, args)

        # Build the DataFrames and MultiIndex's depending on context.
        if tcolw_df is not None:
            by_quantile: dict = {}
            for t, q, df in results:
                if df is not None:
                    by_quantile.setdefault(q, []).append((t, df))
            if not by_quantile:
                return pd.DataFrame()
            q_frames = {}
            for q, tframes in sorted(by_quantile.items()):
                keys, dfs = zip(*tframes)
                q_frames[q] = pd.concat(dfs, keys=keys, names=["date", "band"])
            # A special key indexer has to be added otherwise the band names
            # in the index will be sorted alphabetically rather than by frequency.
            band_rank = {b: i for i, b in enumerate(self.band_frequencies)}
            return (
                pd.concat(q_frames, names=["quantile", "date", "band"])
                .reorder_levels(["date", "band", "quantile"])
                .sort_index(
                    key=lambda idx: idx.map(band_rank) if idx.name == "band" else idx
                )
            )
        else:
            frames = [(t, df) for t, _q, df in results if df is not None]
            if not frames:
                return pd.DataFrame()
            keys, valid_frames = zip(*frames)
            return pd.concat(valid_frames, keys=keys, names=["date", "band"])

    def compute(self) -> tuple[pd.DataFrame, None]:
        """
        Run all AM model forecasts and return clear-sky and cloud DataFrames.

        Returns
        -------
        clear_df : pd.DataFrame or None
            Hourly (date, band)-indexed sensitivity forecast over the full
            forecast range, or ``None`` if the run produced no results.
        cloud_df : pd.DataFrame or None
            15-minute (date, band, quantile)-indexed sensitivity forecast for
            the first 12 hours across all TCOLW spatial quantiles, or ``None``
            if no cloud data are available or the run produced no results.
        """
        baseline_df = self._compute_seasonal_baseline()
        surf_df = self.surf_query.df
        pres_df = self.pres_query.df

        clear_df = self._run_series(surf_df, pres_df)
        if self.has_cloud:
            cloud_df = self._run_series(surf_df, pres_df, tcolw_df=self._get_tcolw_df())
        else:
            cloud_df = None

        return (
            self._add_derived_columns(clear_df, baseline_df)
            if clear_df is not None and not clear_df.empty
            else None,
            self._add_derived_columns(cloud_df, baseline_df)
            if cloud_df is not None and not cloud_df.empty
            else None,
        )

    @property
    def forecast_time(self) -> pd.Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.clear_df is not None and self.cloud_df is not None

    def save_data(self, outname: Path | str = "vla_sensitivity") -> None:
        if self.clear_df is not None:
            to_parquet(self.clear_df, self.forecast_dir / Path(outname))
        else:
            logger.warn("Could not save data for model VLA sensitivity forecast.")
        if self.cloud_df is not None:
            to_parquet(self.cloud_df, self.forecast_dir / Path(f"{outname}_cloud"))
        else:
            logger.warn(
                "Could not save data for model VLA sensitivity forecast with clouds."
            )
