
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import interpolate


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def get_gaps(df, delta="0.25D", col="mjd"):
    ix_gaps = (df.index[1:] - df.index[:-1]) > pd.Timedelta(delta)
    values = df.iloc[:-1][col]
    gap_start = values[ix_gaps]
    gap_end = values[np.roll(ix_gaps, 1)]
    return gap_start, gap_end


def angle_diff(x, y):
    rad_to_deg = 360 / (2 * np.pi)
    c1 = np.exp(np.deg2rad(x) * 1j)
    c2 = np.exp(np.deg2rad(y) * 1j)
    return np.rad2deg(np.angle(c1 / c2))


def concat_dataframe(a_df, b_df, clip=True):
    m_df = pd.concat([a_df, b_df], axis="columns")
    if clip:
        m_df = m_df[
                (m_df.index > a_df.index.min()) &
                (m_df.index < a_df.index.max())
        ]
    m_df.interpolate(inplace=True, limit=6, limit_area="inside")
    return m_df


def add_interpolated_phase(a_df, w_df):
    # FIXME probably needs some filtering NaN values for longer time series.
    # The weather data will act as the input to the network, so interpolate
    # the API data onto the 15-minutely time grid used by the weather data
    # rather than vice-versa.
    mjd = w_df.mjd.values
    interpolator = interpolate.interp1d(
            a_df.mjd.values,
            a_df.rms_phase_med.values,  # Could be multiple columns
    )
    w_df["phase_rms"] = interpolator(mjd)
    return w_df


def preprocess_hrrr(a_df, w_df):
    # Remove LST column
    w_df.drop(columns=["lst"], inplace=True)
    # Remove duplicate entries on the first of the year.
    w_df = w_df[~w_df.index.duplicated(keep="first")]
    # Sub-select time-range to exclude the times at the beginning and
    # end of the months where the log-files are invalid.
    gap_start, gap_end = get_gaps(a_df)
    mjd = w_df.mjd.values
    gap_mask = np.ones_like(mjd, dtype=bool)
    for start, end in zip(gap_start, gap_end):
        gap_mask &= (mjd < start) | (end < mjd)
    w_df = w_df[gap_mask]
    # Interpolate to remove any existing NaN values from the hourly->15-min
    # resampling if they weren't taken care of in the download steps.
    w_df.interpolate(inplace=True)
    return w_df


def get_data_hrrr(
        api_path=DATA_DIR/"api_data.parquet",
        weather_path=DATA_DIR/"weather_data.parquet",
        smooth_phase=3,
    ):
    a_df = pd.read_parquet(api_path)
    # This will introduce a small amount of information leakage (due to the
    # centered window), but because it is only a single 15-min sample, should
    # be minor for the overall 12-hour forecast.
    if smooth_phase is not None:
        a_df = a_df.rolling(smooth_phase, center=True).mean()
    w_df = pd.read_parquet(weather_path)
    w_df = preprocess_hrrr(a_df, w_df)
    w_df = w_df[
            ((w_df.mjd < 58280) | (58346 < w_df.mjd)) &
            (w_df.index < a_df.index.max().isoformat())
    ]
    w_df = add_interpolated_phase(a_df, w_df)
    w_df = w_df[w_df.phase_rms.notnull()].dropna(how="any")
    return w_df

