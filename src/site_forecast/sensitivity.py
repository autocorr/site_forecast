
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .query.open_meteo import (OpenMeteoVlaQuery, OpenMeteoVlaPressureQuery)
    from .query.herbie_maps import HerbieQuery

from numbers import Real
from pathlib import Path

import amwrap
import numpy as np
import pandas as pd
from astropy import units as u

from . import logger
from .query import (QueryBase, to_parquet)

# TODO
#   - nominal sensitivity from midlatitude-winter
#   - range of PWV estimates from ensemble PWV
#   - interpolate to sub-hourly/15-min for the cloud data


class AmModelPredictor:
    # Stratospheric water vapor mixing ratio.
    strato_mixing_ratio = 5e-6 * u.dimensionless_unscaled
    strato_pressure_limit = 30 * u.hPa
    mixing_ratio_epsilon = 1e-7 * u.dimensionless_unscaled
    relative_humidity_epsilon = 0.1 / 3  * u.dimensionless_unscaled
    output_columns = [
            "frequency",
            "brightness temperature",
            "opacity",
            "transmittance",
    ]

    @u.quantity_input
    def __init__(
            self,
            pressure: u.Quantity["pressure"],  # noqa: F821
            temperature: u.Quantity["temperature"],  # noqa: F821
            relative_humidity: u.Quantity["dimensionless"],  # noqa: F821
            surface_pressure: u.Quantity["pressure"] | None=None,  # noqa: F821
            pwv: u.Quantity["length"] | None=None,  # noqa: F821
            freq_min: u.Quantity["frequency"]=1.0*u.GHz,  # noqa: F821
            freq_max: u.Quantity["frequency"]=50.0*u.GHz,  # noqa: F821
            freq_step: u.Quantity["frequency"]=0.2*u.GHz,  # noqa: F821
            time: pd.Time | None=None,
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
        self._pwv = pwv
        self.surface_pressure = surface_pressure
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step
        self.time = time if time is not None else pd.NaT
        # Convert values to descending order in order to be used by `interp_by_pressure`
        if (is_ascending := pressure[1] > pressure[0]):
            pressure = pressure[::-1]
            temperature = temperature[::-1]
            relative_humidity = relative_humidity[::-1]
        # Clip the full pressure level arrays to the current surface level.
        if self.surface_pressure is not None:
            self.pres = amwrap.interp_by_pressure(pressure, pressure, surface_pressure)
            self.temp = amwrap.interp_by_pressure(temperature, pressure, surface_pressure)
            self.relh = amwrap.interp_by_pressure(relative_humidity, pressure, surface_pressure)
        else:
            self.pres = pressure
            self.temp = temperature
            self.relh = relative_humidity
        self.mixr = amwrap.mixing_ratio_from_relative_humidity(self.pres, self.temp, self.relh)
        # Clamp stratospheric water values and small values.
        strat_mask = self.pres < self.strato_pressure_limit
        self.mixr[strat_mask] = np.maximum(self.strato_mixing_ratio, self.mixr[strat_mask])
        self.mixr = np.maximum(self.mixing_ratio_epsilon, self.mixr)
        self.relh = np.maximum(self.relative_humidity_epsilon, self.relh)
        # Get the PWV from the vertical profiles for reference scaling.
        self.model_pwv = amwrap.precipitable_water(self.pres, self.temp, self.relh)

    @classmethod
    def from_frames(
            cls,
            time: pd.Timestamp,
            surf_df: pd.DataFrame,
            pres_df: pd.DataFrame,
            **kwargs
        ):
        pwv = surf_df.loc[time, "total_column_integrated_water_vapour"] * u.mm
        surf_pres = surf_df.loc[time, "surface_pressure"] * u.hPa
        this_pres_df = pres_df.xs(time, level="date")
        pressure = this_pres_df.index.get_level_values("pressure").values * u.hPa
        temperature = this_pres_df["temperature"].values * u.deg_C
        relative_humidity = this_pres_df["relative_humidity"].values / 100 * u.dimensionless_unscaled
        return cls(
                pressure,
                temperature,
                relative_humidity,
                surface_pressure=surf_pres,
                pwv=pwv,
                time=time,
                **kwargs
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
            pwv: u.Quantity["length"] | None,
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


class SeasonalPredictor:
    def __init__(
            self,
            site,
            **kwargs,
        ):
        # FIXME
        # - encapsulate the climatologies and then add parameters to pick the
        #   right one based on time of year and the latitude of the site.
        # - use the site altitude to derive the average surface pressure
        # - use the time to get the season.
        #   - split the spring/fall conditions halfway between summer and winter.
        pass


def am_predict_series(
        surf_df: pd.DataFrame,
        pres_df: pd.DataFrame,
    ) -> pd.DataFrame:
    # get overlapping sets of indices here
    times = pres_df.index.get_level_values("date").unique()
    pressure = pres_df.index.get_level_values("pressure").unique().values * u.hPa
    all_dfs = []
    for i, time in enumerate(times):
        try:
            pwv = surf_df.loc[time, "total_column_integrated_water_vapour"] * u.mm
            surf_pres = surf_df.loc[time, "surface_pressure"] * u.hPa
            this_pres_df = pres_df.xs(time, level="date")
            temperature = this_pres_df["temperature"].values * u.deg_C
            relative_humidity = this_pres_df["relative_humidity"].values / 100 * u.dimensionless_unscaled
        except IndexError:
            continue
        df = am_run_model(
                pressure[::-1],
                temperature[::-1],
                relative_humidity[::-1],
                surface_pressure=surf_pres,
                pwv=pwv,
        )
        df["date"] = time
        df.set_index(["date", "frequency"], inplace=True)
        all_dfs.append(df.sort_index())
    return pd.concat(all_dfs)


class VlaSensitivityEstimator:
    bands = list("LSCXUKAQ")
    # Nominal band center frequencies in GHz.
    center_freqs = {
            "L":  1.5,  #  1.0 -  2.0
            "S":  3.0,  #  2.0 -  4.0
            "C":  6.0,  #  4.0 -  8.0
            "X": 10.0,  #  8.0 - 12.0
            "U": 15.0,  # 12.0 - 18.0
            "K": 22.3,  # 18.0 - 26.5
            "A": 33.3,  # 26.5 - 40.0
            "Q": 45.0,  # 40.0 - 50.0
    }
    # The receiver temperatures in K are taken from the files provided by Rich
    # Moeser in:
    #   https://mctest.evla.nrao.edu/rich/tcals/
    # by taking the median across the band for each receiver and then taking
    # the median across all antennas and polarizations.
    receiver_temperatures = {
            "L": 18.5,
            "S": 20.9,
            "C":  7.9,
            "X": 12.1,
            "U": 12.0,
            "K": 14.6,
            "A": 25.1,
            "Q": 36.8,
    }
    def __init__(
            self,
            om_query_surf: OpenMeteoVlaQuery,
            om_query_pres: OpenMeteoVlaPressureQuery,
            hb_query_tcolw: HerbieQuery,
            hb_query_veril: HerbieQuery,
        ):
        self.surf_query = om_query_surf
        self.pres_query = om_query_pres
        self.maps_query_tcolw = hb_query_tcolw
        self.maps_query_veril = hb_query_veril
        self._time = om_query_surf.forecast_time
        # - for each time stamp create model from the interpolated
        #   pressure level data (to the given surface pressure)
        # - predict over the frequency range

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return all([
            self.surf_query.okay,
            self.pres_query.okay,
            self.maps_query_tcolw.okay,
            self.maps_query_veril.okay,
        ])

    def save_data(self, outname: Path | str="predicted_phase") -> None:
        if not self.okay:
            logger.warn("Could not save data for model phase forecast.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)

