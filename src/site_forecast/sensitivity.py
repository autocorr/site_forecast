from typing import TYPE_CHECKING

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
# - Interpolate hourly data to sub-hourly for 15-min cloud TCOLW/TCOLI.
# - Predict range of sensitivities from the ensemble IFS PWV forecast.


class AmModelPredictor:
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
        Configure AM for a set of vertical atmospheric profiles. The layers are
        clipped and interpolated to the given surface pressure. If a PWV is
        provided then the relative humidity profile in the troposphere is
        rescaled to match the given value.
        """
        # FIXME
        # - Add parameters for cloud liquid water mixing ratio and cloud ice
        #   water mixing ratio.
        # - encapsulate information related to a single AM model instance with a
        #   given pressure, temperature, and relative humidity profile.
        # - allow repeated runs by mutating the troposphere scaling factor and
        #   the cloud column density.
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
    ) -> pd.DataFrame:
        if pwv is None:
            pwv_scale = self.pwv_scale
        else:
            pwv_scale = (pwv / self.model_pwv).to("").value
        model = amwrap.Model(
            self.pres,
            self.temp,
            mixing_ratio={"h2o": self.mixr},
            troposphere_h2o_scaling=pwv_scale,
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

    Uses the day of year to select either the "midlatitude_summer" or
    "midlatitude_winter" standard climatologies and then determines seasonal
    PWV and surface pressure values from ERA5 re-analysis estimates.
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
        """
        Seasonal surface pressure at the VLA site from ERA5 re-analysis.
        """
        return float(np.polyval(self.psurf_coeffs, self.day_of_year)) * u.hPa

    @property
    def _seasonal_pwv(self) -> u.Quantity["length"]:
        """
        Seasonal PWV at the VLA stie from ERA5 re-analysis.
        """
        return float(np.polyval(self.pwv_coeffs, self.day_of_year)) * u.mm

    def run(self, **kwargs) -> pd.DataFrame:
        """Run the seasonal AM model and return a frequency-indexed DataFrame."""
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


def _run_one(
    time: pd.Timestamp,
    surf_row: pd.Series,
    pres_slice: pd.DataFrame,
    freq_min: u.Quantity,
    freq_max: u.Quantity,
    freq_step: u.Quantity,
    band_frequencies: dict,
) -> tuple[pd.Timestamp, pd.DataFrame | None]:
    """Run AM for a single timestep; return (time, band-indexed DataFrame) or (time, None)."""
    try:
        freq_df = (
            AmModelPredictor.from_frames(
                surf_row,
                pres_slice,
                freq_min=freq_min,
                freq_max=freq_max,
                freq_step=freq_step,
            )
            .run()
            .droplevel("date")
        )
        return time, _band_averages(freq_df, band_frequencies)
    except Exception:
        logger.warning(f"Error in calculating sensitivity at {time}")
        return time, None


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


class VlaSensitivityEstimator:
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

    def __init__(
        self,
        om_query_surf: "OpenMeteoVlaQuery",
        om_query_pres: "OpenMeteoVlaPressureQuery",
        n_workers: int = 30,
    ):
        if n_workers < 1:
            raise ValueError(f"Invalid number of workers: {n_workers=}")
        self.surf_query = om_query_surf
        self.pres_query = om_query_pres
        self.n_workers = n_workers
        self._time = om_query_surf.forecast_time
        self.df = None
        try:
            self.df = self.compute()
        except:
            logger.exception("Error computing sensitivity estimates.")
            self.df = None

    def _run_series(
        self,
        surf_df: pd.DataFrame,
        pres_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run AM for all timesteps; return a (date, band)-indexed DataFrame."""
        times = pres_df.index.get_level_values("date").intersection(surf_df.index)
        args = [
            (
                t,
                surf_df.loc[t],
                pres_df.xs(t, level="date"),
                self.freq_min,
                self.freq_max,
                self.freq_step,
                self.band_frequencies,
            )
            for t in times
        ]
        if self.n_workers == 1:
            results = [_run_one(*a) for a in args]
        else:
            with multiprocessing.Pool(self.n_workers) as pool:
                results = pool.starmap(_run_one, args)
        frames = [(t, df) for t, df in results if df is not None]
        if not frames:
            return pd.DataFrame()
        else:
            keys, valid_frames = zip(*frames)
            return pd.concat(valid_frames, keys=keys, names=["date", "band"])

    def _compute_seasonal_baseline(self) -> pd.DataFrame:
        """Run the seasonal AM model; return a (band,)-indexed baseline DataFrame."""
        clim_df = VlaSeasonalPredictor(
            self.forecast_time,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            freq_step=self.freq_step,
        ).run()
        return _band_averages(clim_df, self.band_frequencies)

    def compute(self) -> pd.DataFrame | None:
        """Build the sensitivity DataFrame from the forecast queries."""
        series_df = self._run_series(self.surf_query.df, self.pres_query.df)
        if series_df.empty:
            return None
        baseline_df = self._compute_seasonal_baseline()
        df = series_df.copy()
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

    @property
    def forecast_time(self) -> pd.Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.surf_query.okay and self.pres_query.okay and self.df is not None

    def save_data(self, outname: Path | str = "predicted_phase") -> None:
        if not self.okay:
            logger.warn("Could not save data for model phase forecast.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)
