
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from plotly import graph_objects as go

from .. import (CONFIG, KMHOUR_TO_MS, TPW_TO_PWV, VLBA_SITES, VLBA_SITE_NAMES, logger, _now_dir)

COLORS = px.colors.qualitative.G10


def savefig(fig, outname, t_forecast=None, overwrite=True):
    now_dir = _now_dir(t_forecast)
    out_dir = Path(CONFIG.get("Paths", "plots", fallback="./plots")).expanduser() / now_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / outname
    if path.exists() and not overwrite:
        logger.info(f"Figure exists, continuing: {path}")
    else:
        fig.write_html(f"{path}.html", include_plotlyjs="cdn", auto_open=False)
        print_path = Path(*path.parts[-6:])
        logger.info(f"Figure saved: {print_path}")


def get_multi_site_stack_layout(
        n_rows=10,
        v_delta=0.005,
        limits=None,
    ):
    if limits is None:
        limits = (None, None, None, None)
    xmin, xmax, ymin, ymax = limits
    if n_rows < 1:
        raise ValueError(f"Invalid number of sites: {n_rows=}")
    if any(limits) and (xmin > xmax or ymin > ymax):
        raise ValueError(f"Invalid axis limits: {limits=}")
    yaxes_kwargs = {
            f"yaxis{i}": {
                "domain": [(i-1)/n_rows, i/n_rows-v_delta],
                "range": [ymin, ymax],
                "hoverformat": ".3r",
            }
            for i in range(1, n_rows+1)
    }
    layout = go.Layout(
            legend=dict(traceorder="reversed"),
            height=1000,
            width=1400,
            showlegend=True,
            hoversubplots="axis",
            hovermode="x unified",
            margin=dict(b=20, l=20, r=20, t=20),
            font=dict(size=14),
            xaxis=dict(
                range=[xmin, xmax],
                showspikes=True,
                spikemode="across+marker",
                spikesnap="data",
                spikethickness=1,
                spikedash="dash",
                spikecolor="black",
                minor=dict(
                    ticklen=6,
                    tickcolor="black",
                    tickmode="auto",
                    nticks=6,
                    showgrid=True,

                ),
            ),
            **yaxes_kwargs
    )
    return layout


def get_two_stack_agg_layout(**kwargs):
    layout = get_multi_site_stack_layout(n_rows=2, v_delta=0.01, **kwargs)
    layout.update(hovermode="x")
    return layout


def get_now_trace(x, y_lim, yaxis=None):
    return go.Scatter(
            x=[x, x],
            y=y_lim,
            mode="lines",
            line_width=1.5,
            line_dash="dash",
            line_color="magenta",
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
    )


def get_freezing_trace(xmin, xmax, yaxis=None):
    return go.Scatter(
            x=[xmin, xmax],
            y=[0, 0],
            mode="lines",
            line_width=1.5,
            line_dash="dash",
            line_color="mediumblue",
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
    )


def get_wind_limit_graph_objects(xmin, xmax, yaxis=None, vlba=True):
    stow_limit = 25 if vlba else 20  # m/s
    labels = ["Q", "A", "K", "U", "X", " !"]
    limits = [  5,   6,   7,  10,  15, stow_limit]  # m/s
    graph_objects = []
    for label, limit in zip(labels, limits):
        if limit == stow_limit:
            color = "black"
            line_width = 2.5
        else:
            color = "dimgray"
            line_width = 1.0
        scatter = go.Scatter(
                x=[xmin, xmax],
                y=[limit, limit],
                mode="lines",
                line_width=line_width,
                line_dash="dot",
                line_color=color,
                showlegend=False,
                hoverinfo="skip",
                yaxis=yaxis,
        )
        graph_objects.append(scatter)
    return graph_objects


def get_annotation_trace(x, y, text, yaxis=None, x_offset=pd.Timedelta("1.5h")):
    return go.Scatter(
            x=[x+x_offset],
            y=[y],
            mode="text",
            text=[text],
            textposition="top center",
            yaxis=yaxis,
            showlegend=False,
            hoverinfo="skip",
    )


def get_sun_rise_set_by_site(times, station, t_offset=pd.Timedelta("1d")):
    match times:
        case (t_min, t_max) if isinstance(times, tuple):
            t_center = (t_max - t_min) / 2 + t_min
            t_delta  = (t_max - t_min) / 2 + t_offset
        case _:
            t_center = times.mean()
            t_delta  = (times.max() - t_center) + t_offset
    return station.sun_rise_and_sets(t_center, delta=t_delta)


def get_sun_rise_set_patches(rises, sets, y_lim, yaxis="y1", opacity=0.2):
    ymin, ymax = y_lim
    x_vertices = []
    y_vertices = []
    for t_rise, t_set in zip(rises, sets):
        t_rise = t_rise.to_datetime()
        t_set  = t_set.to_datetime()
        x_vertices.extend([t_rise, t_rise, t_set, t_set, t_rise, None])
        y_vertices.extend([ymin, ymax, ymax, ymin, ymin, None])
    return go.Scatter(
            x=x_vertices,
            y=y_vertices,
            fill="toself",
            opacity=opacity,
            line_width=0,
            mode="lines",
            line_color="gray",
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
    )


class PlotlyFigureBase:
    n_rows = 1
    x_offset_lo = pd.Timedelta("12h")
    x_offset_hi = pd.Timedelta("4.5d")

    def __init__(self, series, x_lim=None, y_lim=None, label=None, t_forecast=None):
        if x_lim is not None:
            self.x_lim = x_lim
        elif t_forecast is not None:
            self.x_lim = (t_forecast - self.x_offset_lo, t_forecast + self.x_offset_hi)
        else:
            times = series.index.get_level_values("date")
            self.x_lim = (times.min(), times.max())
        if y_lim is not None:
            self.y_lim = y_lim
        else:
            self.y_lim = (series.min(), series.max())
        self.series = series
        self.label = series.name if label is None else label
        self.data = []
        self.t_forecast = t_forecast
        self.layout = None

    @property
    def xmin(self):
        return self.x_lim[0]

    @property
    def xmax(self):
        return self.x_lim[1]

    @property
    def ymin(self):
        return self.y_lim[0]

    @property
    def ymax(self):
        return self.y_lim[1]

    @property
    def limits(self):
        return (*self.x_lim, *self.y_lim)

    @property
    def is_wind_speed(self):
        return "wind_speed" in self.series.name

    @property
    def is_temperature(self):
        return "temperature" in self.series.name

    @property
    def scale_factor(self):
        if self.is_wind_speed:
            return KMHOUR_TO_MS
        else:
            return 1

    def add_now_line(self) -> None:
        self.data.extend([
                get_now_trace(self.t_forecast, self.y_lim, f"y{i}")
                for i in range(1, self.n_rows+1)
        ])

    def add_freezing_line(self) -> None:
        for i in range(1, self.n_rows+1):
            self.data.append(get_freezing_trace(*self.x_lim, f"y{i}"))

    def add_wind_limits(self) -> None:
        for i in range(1, self.n_rows+1):
            self.data.extend(get_wind_limit_graph_objects(*self.x_lim, f"y{i}"))

    def add_value_traces(self, yaxis=None) -> None:
        for i, site in enumerate(self.sites):
            ser = self.series.xs(site.name, level="site")
            self.data.append(go.Scatter(
                    x=ser.index,
                    y=ser * self.scale_factor,
                    name=site.name,
                    line=dict(color=COLORS[i]),
                    mode="lines",
                    yaxis=f"y{i+1}" if yaxis is None else yaxis,
                    legendgroup="value",
            ))

    def add_quantile_traces(
            self,
            yaxis=None,
            antenna_counts=(3, 4, 5, 6, 7, 8, 9, 10),
            highlight_antenna=8,
            max_count=10,
        ) -> None:
        for n_s in antenna_counts:
            q = n_s / max_count
            ser = (
                    self.series[self.series.index.get_level_values("site").isin(self.site_names)]
                    .groupby(level="date")
                    .quantile(min(q, 1.0))
            )
            if n_s == highlight_antenna:
                line = dict(color="mediumblue", width=4.0)
                marker = dict(size=8)
            else:
                line = dict(color="black", width=1.75)
                marker = dict(size=0.1)
            self.data.append(go.Scatter(
                    x=ser.index,
                    y=ser * self.scale_factor,
                    name=f"{q:.1f}",
                    line=line,
                    marker=marker,
                    mode="lines",
                    yaxis=yaxis,
                    showlegend=False,
                    legendgroup="quantile",
            ))

    def save(self, outname, **kwargs):
        fig = go.Figure(data=self.data, layout=self.layout)
        savefig(fig, outname, t_forecast=self.t_forecast, **kwargs)
        return fig


class PlotlyFigureVlbaBase(PlotlyFigureBase):
    sites = VLBA_SITES
    site_names = VLBA_SITE_NAMES
    n_sites = len(VLBA_SITE_NAMES)

    def __init__(self, *args, sun_rise_sets=None, **kwargs):
        super().__init__(*args, **kwargs)
        if sun_rise_sets is None:
            self.sun_rise_sets_by_site = {
                    site.name: get_sun_rise_set_by_site((self.xmin, self.xmax), site)
                    for site in self.sites
            }
        else:
            self.sun_rise_sets_by_site = sun_rise_sets


class PlotlyFigureVlbaStack(PlotlyFigureVlbaBase):
    n_rows = 10
    suffix = "stack"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = get_multi_site_stack_layout(n_rows=self.n_rows, limits=self.limits)

    def add_site_name_annotations(self, annotation_offset_scale=0.75):
        for i, name in enumerate(VLBA_SITE_NAMES):
            trace = get_annotation_trace(
                    self.xmin,
                    self.ymax*annotation_offset_scale,
                    name,
                    yaxis=f"y{i+1}",
            )
            self.data.append(trace)

    def add_sun_rise_set(self, opacity=0.2):
        for i, name in enumerate(self.site_names):
            rises, sets = self.sun_rise_sets_by_site[name]
            patches = get_sun_rise_set_patches(rises, sets, self.y_lim, yaxis=f"y{i+1}", opacity=opacity)
            self.data.append(patches)

    def add_traces(self):
        self.add_sun_rise_set()
        if self.is_wind_speed:
            self.add_wind_limits()
        elif self.is_temperature:
            self.add_freezing_line()
        self.add_now_line()
        self.add_value_traces()
        self.add_site_name_annotations()


class PlotlyFigureVlbaAgg(PlotlyFigureVlbaBase):
    n_rows = 2
    suffix = "agg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = get_two_stack_agg_layout(limits=self.limits)

    def add_sun_rise_set(self, opacity=0.075):
        for name in self.site_names:
            rises, sets = self.sun_rise_sets_by_site[name]
            for i in (1, 2):
                patches = get_sun_rise_set_patches(rises, sets, self.y_lim, yaxis=f"y{i}", opacity=opacity)
                self.data.append(patches)

    def add_traces(self):
        self.add_sun_rise_set()
        if self.is_wind_speed:
            self.add_wind_limits()
        elif self.is_temperature:
            self.add_freezing_line()
        self.add_now_line()
        self.add_value_traces(yaxis="y1")
        self.add_quantile_traces(yaxis="y2")


def plot_vlba_multisite(fc, prefix=None):
    if not fc.weather_ms.okay:
        logger.warn("Skipping multi-site VLBA plots")
        return
    df = fc.weather_ms.df.copy()
    times = df.index.get_level_values("date").unique()
    df["temp_dewpoint_diff"] = df.temperature_2m - df.dew_point_2m
    df["cloud_cover_midlow"] = df[["cloud_cover_mid", "cloud_cover_low"]].max(axis=1)
    items = (
            ("wind_speed_10m", "wind_speed", (0, 30)),
            ("total_column_integrated_water_vapour", "pwv", (0, 40)),
            ("precipitation", "precip", (0, 5)),
            ("precipitation_probability", "precip_prob", (0, 102)),
            ("temperature_2m", "temperature", (-25, 45)),
            ("temp_dewpoint_diff", "temp_dewp_diff", (-5, 50)),
            ("boundary_layer_height", "pbl_height", (0, 5)),
            ("cloud_cover_midlow", "cloud_cover", (0, 102)),
    )
    sun_rise_sets = {
            site.name: get_sun_rise_set_by_site((times.min(), times.max()), site)
            for site in VLBA_SITES
    }
    for col, label, y_lim in items:
        try:
            series = df[col]
            if col == "boundary_layer_height":
                series = series / 1e3  # m to km
            for cls in (PlotlyFigureVlbaStack, PlotlyFigureVlbaAgg):
                fig = cls(
                        series,
                        y_lim=y_lim,
                        t_forecast=fc.forecast_time,
                        sun_rise_sets=sun_rise_sets,
                )
                fig.add_traces()
                outname = f"{label}_ms_{fig.suffix}"
                if prefix is not None:
                    fig.save(f"{prefix}_{outname}")
                else:
                    fig.save(outname)
        except:
            logger.exception(f"Unhandled exception for {col}")

