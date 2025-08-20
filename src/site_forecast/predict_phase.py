
from pathlib import Path
from typing import Union

import pandas as pd
from pandas import (DataFrame, Timestamp)

from darts import TimeSeries
from darts.models import LightGBMModel

from . import (MODEL_DIR, logger)
from .query import (
        QueryBase,
        to_parquet,
)


def get_model(filen) -> LightGBMModel:
    logger.info(f"Using model: {filen}")
    model = LightGBMModel.load(MODEL_DIR / f"model_{filen}.pkl")
    model.name = filen
    return model


def predict_model(model, series, future, n=48) -> DataFrame:
    # Huge kludge to work around an indexing problem within darts in
    # the `darts.models.forecasting.sklearn_model` module's `predict`
    # function. The issue seems to be in not using an inclusive
    # slice for this index type for `covariate_matrices`, between
    # a `pd.DatetimeIndex` and `pd.RangeIndex`.
    if model.name == "no_weather":
        series._has_datetime_index = False
    prediction = model.predict(
            n=n,
            series=series,
            future_covariates=future,
            verbose=False,
    )
    if model.name == "no_weather":
        series._has_datetime_index = True
    return prediction.to_dataframe().tz_localize("utc")


class ModelPhaseForecast(QueryBase):
    model_types = {
            # (has_weather, has_phase): model_name
            (True,  True):  "full",
            (False, True):  "no_weather",
            (True,  False): "no_phase",
            (False, False): "seasonal",
    }

    def __init__(self, w_query, p_query):
        # FIXME Should fill NaNs in the measured phase data.
        w_ts = w_query.to_model_series()
        p_ts = p_query.to_model_series()
        if p_ts is None:
            when = w_query.forecast_time.tz_localize(None)
            p_ts, _ = w_ts["hour"].split_before(when)
            p_ts._components = pd.Index(["phase_rms"])
            p_ts._time_index.name = "time"
            p_ts._time_dim = "time"
        which_okay = (w_query.okay, p_query.okay_for_model)
        model = get_model(self.model_types[which_okay])
        try:
            self.df = predict_model(model, p_ts, w_ts)
        except:
            logger.exception("Error predicting phase RMS from model.")
            self.df = None
        self.w_ts = w_ts
        self.p_ts = p_ts
        self.model = model
        self._time = w_query.forecast_time
        self._w_query = w_query
        self._p_query = p_query

    @property
    def forecast_time(self) -> Timestamp:
        return self._time

    @property
    def okay(self) -> bool:
        return self.df is not None

    def save_data(self, outname: Union[Path, str]="predicted_phase") -> None:
        if not self.okay:
            logger.warn("Could not save data for model phase forecast.")
            return
        outpath = self.forecast_dir / Path(outname)
        to_parquet(self.df, outpath)

