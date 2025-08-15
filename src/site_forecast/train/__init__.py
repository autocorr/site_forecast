
import warnings

import numpy as np

from darts import TimeSeries, metrics, models
from darts.timeseries import slice_intersect, concatenate
from darts.utils.missing_values import extract_subseries


warnings.filterwarnings(action="ignore", category=UserWarning,
        message=".*sliced data.*")
warnings.filterwarnings(action="ignore", category=UserWarning,
        message=".*X does not have valid feature names.*")

TRAINING_COLUMNS = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "wind_speed_10m",
        "wind_speed_80m",
        "wind_direction_10m",
        "wind_direction_80m",
        "wind_gusts_10m",
        "precipitation",
        "freezing_level_height",
        "cape",
        "visibility",
        "direct_radiation",
        "diffuse_radiation",
        "lifted_index",
        "convective_inhibition",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        #"sensible_heat_flux",
        #"latent_heat_flux",
        "hour",
        "day_of_year",
        "phase_rms",
]


def to_training_subset(df):
    return df[[c for c in df.columns if c in TRAINING_COLUMNS]]


def extract_filtered_subseries(series, min_size=240):
    subseries = extract_subseries(series)
    return [s for s in subseries if s.shape[0] >= min_size]


def concatenate_aligned_subseries(sub_a, sub_b):
    sub_a_int = []
    sub_b_int = []
    for sub_a_sel, sub_b_sel in zip(sub_a, sub_b):
        _sub_a, _sub_b = slice_intersect([sub_a_sel, sub_b_sel])
        sub_a_int.append(_sub_a)
        sub_b_int.append(_sub_b)
    return (
            concatenate(sub_a_int, ignore_time_axis=True),
            concatenate(sub_b_int, ignore_time_axis=True),
    )


class DataSet:
    def __init__(
            self,
            series,
            split_frac,
        ):
        train, valid = series.split_before(split_frac)
        self.series  = series
        self.train   = extract_filtered_subseries(train)
        self.valid   = extract_filtered_subseries(valid)


class TrainingData:
    def __init__(
            self,
            df,
            split_frac=0.85,
            min_subs_size=192,
            drop_cols=["mjd"],
            target_cols=["phase_rms"],
        ):
        self.df = df
        self.split_frac = split_frac
        self.min_subs_size = min_subs_size
        self.drop_cols = drop_cols
        self.target_cols = target_cols
        series = (
                TimeSeries
                .from_dataframe(self.df, fill_missing_dates=True)
                .drop_columns(drop_cols)
                .astype(np.float32)
        )
        # Target series
        y = series[target_cols]
        self.y = DataSet(y, split_frac)
        # Future covariates
        X = series.drop_columns(target_cols)
        self.X = DataSet(X, split_frac)
        self.future_cols = X.components.tolist()


def fit_model(
        td,
        input_chunk=48,
        output_chunk=48,
        use_target_lags=True,
    ):
    if use_target_lags:
        lags = input_chunk
        y_valid = td.y.valid
    else:
        lags = None
        y_valid = None
    model = models.LightGBMModel(
            lags=lags,
            lags_future_covariates=(input_chunk, output_chunk),
            output_chunk_length=output_chunk,
            force_col_wise=True,
            verbose=1,
    )
    model.fit(
            td.y.train,
            future_covariates=td.X.train,
            val_series=y_valid,
            val_future_covariates=td.X.valid,
    )
    return model


def backtest_model(
        model,
        td,
        forecast_horizon=24,
    ):
    backtest = model.historical_forecasts(
            series=td.y.valid,
            future_covariates=td.X.valid,
            forecast_horizon=forecast_horizon,
            retrain=False,
            verbose=True,
    )
    losses = metrics.rmse(td.y.valid, backtest)
    aligned = concatenate_aligned_subseries(td.y.valid, backtest)
    mae = metrics.mae(*aligned)
    rms = metrics.rmse(*aligned)
    return backtest, losses, (mae, rms)


def train_models(w_df):
    t_df = w_df[["phase_rms", "mjd", "hour", "day_of_year"]]
    td_full = TrainingData(w_df)
    td_time = TrainingData(t_df)
    def run_and_save(td, name, use_target_lags):
        model = fit_model(td, use_target_lags=use_target_lags)
        model.save(f"{name}.pkl", clean=True)
        _, _, (mae, rms) = backtest_model(model, td)
        print(f"-- {name}: (MAE: {mae}, RMS: {rms})")
    # Both measured phase RMS and weather data.
    run_and_save(td_full, "model_full", True)
    run_and_save(td_time, "model_no_weather", True)
    run_and_save(td_full, "model_no_phase", False)
    run_and_save(td_time, "model_seasonal", False)

