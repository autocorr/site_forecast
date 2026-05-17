import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from plotly import graph_objects as go

from site_forecast.plotting.plot_mpl import splice_colormaps, truncate_colormap
from site_forecast.plotting.plot_plotly import (
    get_multi_site_stack_layout,
    get_two_stack_agg_layout,
)


def test_truncate_colormap_returns_linear_segmented():
    cmap = truncate_colormap("magma", vmin=0.4, vmax=1.0, n_color=64)
    assert isinstance(cmap, LinearSegmentedColormap)


def test_truncate_colormap_endpoints_match_source_subrange():
    """The truncated cmap at 0.0 / 1.0 should match the source cmap at
    vmin / vmax respectively."""
    import matplotlib.pyplot as plt

    source = plt.get_cmap("magma")
    cmap = truncate_colormap("magma", vmin=0.4, vmax=1.0, n_color=256)
    np.testing.assert_allclose(cmap(0.0)[:3], source(0.4)[:3], atol=1e-3)
    np.testing.assert_allclose(cmap(1.0)[:3], source(1.0)[:3], atol=1e-3)


def test_splice_colormaps_returns_listed_with_n_color():
    cmap = splice_colormaps("gray_r", "magma", pivot=0.5, n_color=64)
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 64


def test_splice_colormaps_left_half_comes_from_first_cmap():
    import matplotlib.pyplot as plt

    gray = plt.get_cmap("gray_r", 64)
    spliced = splice_colormaps("gray_r", "magma", pivot=0.5, n_color=64)
    # Color at the lowest sample should match the lowest gray_r sample.
    np.testing.assert_allclose(spliced(0)[:3], gray(0)[:3], atol=1e-3)


@pytest.mark.parametrize("pivot", [0.0, 1.0])
def test_splice_colormaps_clamps_extreme_pivots(pivot):
    # Both 0 and 1 should produce a valid colormap rather than raise.
    cmap = splice_colormaps("gray_r", "magma", pivot=pivot, n_color=32)
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 32


def test_multi_site_stack_layout_default_yaxes():
    layout = get_multi_site_stack_layout(n_rows=10)
    assert isinstance(layout, go.Layout)
    payload = layout.to_plotly_json()
    for i in range(1, 11):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        assert key in payload
    assert layout.hovermode == "x unified"
    assert layout.hoversubplots == "axis"


def test_multi_site_stack_layout_domains_stack_within_unit_interval():
    n_rows = 5
    v_delta = 0.01
    layout = get_multi_site_stack_layout(n_rows=n_rows, v_delta=v_delta)
    payload = layout.to_plotly_json()
    for i in range(1, n_rows + 1):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        lo, hi = payload[key]["domain"]
        assert lo == pytest.approx((i - 1) / n_rows)
        assert hi == pytest.approx(i / n_rows - v_delta)
        assert 0.0 <= lo < hi <= 1.0


def test_multi_site_stack_layout_applies_limits():
    layout = get_multi_site_stack_layout(n_rows=3, limits=(0, 10, -1, 1))
    payload = layout.to_plotly_json()
    assert payload["xaxis"]["range"] == [0, 10]
    assert payload["yaxis"]["range"] == [-1, 1]


def test_multi_site_stack_layout_rejects_zero_rows():
    with pytest.raises(ValueError):
        get_multi_site_stack_layout(n_rows=0)


def test_multi_site_stack_layout_rejects_inverted_limits():
    with pytest.raises(ValueError):
        get_multi_site_stack_layout(n_rows=2, limits=(10, 0, -1, 1))


def test_two_stack_agg_layout_overrides_hovermode_and_uses_two_rows():
    layout = get_two_stack_agg_layout()
    payload = layout.to_plotly_json()
    assert layout.hovermode == "x"
    assert "yaxis2" in payload
    assert "yaxis3" not in payload
