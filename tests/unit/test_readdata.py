import numpy as np
import pandas as pd
import pytest

from site_forecast.train.readdata import (
    add_interpolated_phase,
    angle_diff,
    concat_dataframe,
    get_gaps,
    preprocess_hrrr,
)


def test_get_gaps_single_gap():
    idx = pd.DatetimeIndex(
        ["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 12:00", "2025-01-01 12:10"]
    )
    df = pd.DataFrame({"mjd": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    gap_start, gap_end = get_gaps(df, delta="0.25D", col="mjd")
    assert gap_start.tolist() == [2.0]
    assert gap_end.tolist() == [3.0]


def test_get_gaps_no_gap():
    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    df = pd.DataFrame({"mjd": np.arange(5, dtype=float)}, index=idx)
    gap_start, gap_end = get_gaps(df, delta="0.25D", col="mjd")
    assert gap_start.tolist() == []
    assert gap_end.tolist() == []


def test_get_gaps_uses_named_column():
    idx = pd.DatetimeIndex(
        ["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-02 00:00", "2025-01-02 00:10"]
    )
    df = pd.DataFrame(
        {"mjd": [1.0, 2.0, 3.0, 4.0], "other": [10.0, 20.0, 30.0, 40.0]},
        index=idx,
    )
    gap_start, gap_end = get_gaps(df, col="other")
    assert gap_start.tolist() == [20.0]
    assert gap_end.tolist() == [30.0]


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (10, 350, 20.0),
        (350, 10, -20.0),
        (180, 0, 180.0),
        (0, 0, 0.0),
        (90, 0, 90.0),
    ],
)
def test_angle_diff_scalars(x, y, expected):
    assert angle_diff(x, y) == pytest.approx(expected, abs=1e-9)


def test_angle_diff_vectorized():
    x = np.array([0.0, 90.0, 180.0])
    y = np.array([10.0, 80.0, 170.0])
    np.testing.assert_allclose(angle_diff(x, y), [-10.0, 10.0, 10.0], atol=1e-9)


def test_concat_dataframe_clipped_interpolates_inside_range():
    idx_a = pd.date_range("2025-01-01", periods=4, freq="1h")
    idx_b = pd.date_range("2025-01-01 00:30", periods=4, freq="1h")
    a = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]}, index=idx_a)
    b = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0]}, index=idx_b)
    m = concat_dataframe(a, b, clip=True)
    assert m.index.min() > a.index.min()
    assert m.index.max() < a.index.max()
    assert "x" in m.columns and "y" in m.columns
    # 'y' has values at every 30-min mark within the kept range, so it
    # gets interpolated onto the hourly 'x' grid.
    assert m.loc["2025-01-01 01:00", "y"] == pytest.approx(15.0)


def test_concat_dataframe_unclipped_preserves_full_span():
    idx_a = pd.date_range("2025-01-01", periods=4, freq="1h")
    idx_b = pd.date_range("2025-01-01 00:30", periods=4, freq="1h")
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=idx_a)
    b = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0]}, index=idx_b)
    m = concat_dataframe(a, b, clip=False)
    assert m.index.min() == a.index.min()
    assert m.index.max() == b.index.max()


def test_add_interpolated_phase_writes_phase_rms_column():
    idx_a = pd.date_range("2025-01-01", periods=5, freq="1h")
    mjd_a = np.linspace(60310.0, 60310.1667, 5)
    a = pd.DataFrame(
        {"mjd": mjd_a, "rms_phase_med": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx_a
    )
    idx_w = pd.date_range("2025-01-01 00:30", periods=3, freq="1h")
    mjd_w = np.linspace(60310.0208, 60310.1042, 3)
    w = pd.DataFrame({"mjd": mjd_w}, index=idx_w)
    result = add_interpolated_phase(a, w)
    assert "phase_rms" in result.columns
    # Middle value lies exactly between rms_phase_med = 2.0 and 3.0.
    assert result["phase_rms"].iloc[1] == pytest.approx(2.5, abs=1e-3)
    # Endpoints fall inside the a_df mjd range, so interpolation is valid.
    assert result["phase_rms"].notna().all()


def test_preprocess_hrrr_drops_lst_and_dedups_and_masks_gaps():
    idx_a = pd.DatetimeIndex(
        [
            "2025-01-01 00:00",
            "2025-01-01 06:00",
            "2025-01-02 00:00",
            "2025-01-02 06:00",
        ]
    )
    a = pd.DataFrame({"mjd": [60310.0, 60310.25, 60311.0, 60311.25]}, index=idx_a)
    idx_w = pd.date_range("2025-01-01", "2025-01-02 06:00", freq="3h")
    w = pd.DataFrame(
        {
            "mjd": np.linspace(60310.0, 60311.25, len(idx_w)),
            "lst": np.ones(len(idx_w)),
            "val": np.linspace(0.0, 1.0, len(idx_w)),
        },
        index=idx_w,
    )
    # Duplicate the first row to exercise the dedup branch.
    w = pd.concat([w.iloc[[0]], w])

    result = preprocess_hrrr(a, w)

    assert "lst" not in result.columns
    assert not result.index.has_duplicates
    # The 06:00 day-1 → 00:00 day-2 jump in `a` (18 h > 0.25 D) defines a gap.
    # Rows of `w` whose mjd lies inside the open interval (60310.25, 60311.0)
    # must be dropped.
    # The mask `(mjd < start) | (end < mjd)` is strict on both sides, so
    # rows at exactly the gap boundaries are also dropped.
    inside_gap = (result["mjd"] >= 60310.25) & (result["mjd"] <= 60311.0)
    assert not inside_gap.any()
    # Rows outside the gap are kept.
    assert (result["mjd"] < 60310.25).any()
    assert (result["mjd"] > 60311.0).any()
